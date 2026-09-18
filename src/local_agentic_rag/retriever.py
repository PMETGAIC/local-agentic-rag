from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, List

from chromadb import Client
from sentence_transformers import SentenceTransformer

class RAGRetriever:
    """
    RAG minimale:
      1. carica PDF/Markdown
      2. chunking in token
      3. embedding con Nomic
      4. indicizzazione in ChromaDB in-memory
      5. retrieval tramite similarità coseno
    """

    EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        embedding_device: str = "cuda",
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap deve essere < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        print(f"[RAG] Caricamento embedding model: {self.EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(
            self.EMBEDDING_MODEL,
            trust_remote_code=True,
            device=embedding_device,
        )
        self.chroma = Client()
        self.collection = self.chroma.get_or_create_collection(
            name="rag_documents",
            configuration={"hnsw": {"space": "cosine"}},
        )
        print("[RAG] Retriever pronto.")

    def load_file(self, path: str | Path) -> List[dict[str, Any]]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._load_pdf(path)
        if suffix in {".md", ".markdown", ".txt"}:
            return self._load_text(path)
        raise ValueError(
            f"Formato non supportato: {suffix}. "
            f"Usa PDF, MD, MARKDOWN o TXT."
        )

    def load_directory(self, directory: str | Path) -> List[dict[str, Any]]:
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(directory)
        supported = {".pdf", ".md", ".markdown", ".txt"}
        files = [p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in supported]
        documents = []
        for path in sorted(files):
            print(f"[RAG] Carico: {path}")
            documents.extend(self.load_file(path))
        return documents

    def _load_text(self, path: Path) -> List[dict[str, Any]]:
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            return []
        return [{
            "text": text,
            "source": str(path),
            "page": None,
        }]

    def _load_pdf(self, path: Path) -> List[dict[str, Any]]:
        reader = PdfReader(str(path))
        documents = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():
                continue
            documents.append({
                "text": text,
                "source": str(path),
                "page": page_number,
            })
        return documents

    def chunk_text(self, text: str) -> List[str]:
        tokenizer = self.embedding_model.tokenizer
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            return []
        step = self.chunk_size - self.chunk_overlap
        chunks = []
        for start in range(0, len(token_ids), step):
            end = start + self.chunk_size
            chunk_ids = token_ids[start:end]
            chunk = tokenizer.decode(chunk_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(token_ids):
                break
        return chunks

    def index_documents(self, documents: List[dict[str, Any]]) -> int:
        all_chunks = []
        all_ids = []
        all_metadatas = []
        for document in documents:
            chunks = self.chunk_text(document["text"])
            for chunk_index, chunk in enumerate(chunks):
                source = document["source"]
                page = document["page"]
                embedding_text = f"search_document: {chunk}"
                chunk_id_raw = f"{source}|{page}|{chunk_index}|{chunk}"
                chunk_id = hashlib.sha256(chunk_id_raw.encode("utf-8")).hexdigest()
                all_chunks.append(embedding_text)
                all_ids.append(chunk_id)
                metadata = {
                    "source": source,
                    "chunk_index": chunk_index,
                }
                if page is not None:
                    metadata["page"] = int(page)
                all_metadatas.append(metadata)
        if not all_chunks:
            print("[RAG] Nessun chunk da indicizzare.")
            return 0
        print(f"[RAG] Chunk totali: {len(all_chunks)}")
        print("[RAG] Generazione embedding...")
        embeddings = self.embedding_model.encode(
            all_chunks,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        print("[RAG] Inserimento in ChromaDB...")
        self.collection.upsert(
            ids=all_ids,
            documents=all_chunks,
            embeddings=embeddings.tolist(),
            metadatas=all_metadatas,
        )
        print(f"[RAG] Indicizzati {len(all_chunks)} chunk.")
        return len(all_chunks)

    def retrieve(self, query: str, top_k: int = 5) -> List[dict[str, Any]]:
        if not query.strip():
            return []
        query_text = f"search_query: {query}"
        query_embedding = self.embedding_model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        output = []
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        for document, metadata, distance in zip(documents, metadatas, distances):
            output.append({
                "text": document,
                "metadata": metadata,
                "distance": float(distance),
                "similarity": 1.0 - float(distance),
            })
        return output

    def count(self) -> int:
        return self.collection.count()