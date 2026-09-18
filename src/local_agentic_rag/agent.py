import json
from pathlib import Path
from llama_cpp import Llama
from .tools import parse_llm_action, TOOLS_SCHEMA
from .retriever import RAGRetriever

class Agent:
    def __init__(self, llm_path: str, n_ctx: int, n_batch: int, retriever: RAGRetriever):
        self.llm = Llama(model_path=llm_path, n_gpu_layers=-1, n_ctx=n_ctx, n_batch=n_batch, verbose=False)
        self.retriever = retriever
        self.history = []
        self.sys_prompt = (
            f"Available tools: {TOOLS_SCHEMA}\n"
            "Directory context: files are in './documents/'.\n"
            "RULES:\n"
            "1. Use 'query_rag' for general questions about document content.\n"
            "2. Use 'read_file' ONLY when the user explicitly asks to read a specific file path.\n"
            "3. Output ONLY valid JSON to use a tool: {\"name\": \"read_file\", \"parameters\": {\"filepath\": \"./documents/test.md\"}}\n"
            "4. Output plain text to answer directly."
        )

    def run(self, user_query: str, retries: int = 3) -> str:
        self.history.append({"role": "user", "content": user_query})
        
        if len(self.history) > 6:
            self.history = self.history[-6:]

        msgs = [{"role": "system", "content": self.sys_prompt}] + self.history

        for _ in range(retries):
            out = self.llm.create_chat_completion(messages=msgs, max_tokens=512, temperature=0.1)
            txt = out["choices"][0]["message"]["content"].strip()
            
            action = parse_llm_action(txt)
            if not action:
                self.history.append({"role": "assistant", "content": txt})
                return txt
            
            if action.name == "query_rag":
                docs = self.retriever.retrieve(action.parameters.get("query", ""), top_k=3)
                res = "\n".join(d["text"] for d in docs)
            elif action.name == "read_file":
                p = Path(action.parameters.get("filepath", ""))
                res = p.read_text(encoding="utf-8") if p.exists() else "File not found."
            else:
                res = "Unknown tool."

            msgs.extend([
                {"role": "assistant", "content": txt},
                {"role": "user", "content": f"Tool result:\n{res}\nAnswer the query."}
            ])
            
        return "Task failed."