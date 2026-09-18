import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import cfg
from local_agentic_rag.retriever import RAGRetriever
from local_agentic_rag.agent import Agent

def main():
    hw = cfg.active
    rag = RAGRetriever(chunk_size=500, chunk_overlap=50)
    
    docs = rag.load_directory("./documents")
    if docs:
        rag.index_documents(docs)

    agent = Agent(llm_path=hw["llm_path"], n_ctx=hw["n_ctx"], n_batch=hw["n_batch"], retriever=rag)

    while True:
        try:
            query = input("\nDomanda (exit per uscire): ")
            if query.strip().lower() in ["exit", "quit"]:
                break
            res = agent.run(query)
            print(f"\nRISPOSTA AGENTE:\n{res}")
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()