"""
Interactive testing script for the LanceDB RAG pipeline.
"""
import sys
from pathlib import Path

# Add implementation directory to path so we can import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.retriever import RAGRetriever

def main():
    print("Initializing RAG Retriever and loading Sentence Transformers...")
    try:
        rag = RAGRetriever()
        # Ping the DB to ensure connection is valid
        _ = rag.table
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize RAG: {e}")
        print("Did you build the knowledge base first? (Run build_knowledge_base.py)")
        return

    print("\n[SUCCESS] RAG Retriever initialized and connected to LanceDB.")
    print("Type your debate premise/query below to see what evidence the RAG pulls.")
    print("(Type 'quit' or 'exit' to stop)\n")
    
    while True:
        try:
            query = input("Query > ")
            if query.strip().lower() in ["quit", "exit"]:
                break
            if not query.strip():
                continue
                
            print("\n" + "="*60)
            print("SEARCHING LANCEDB...")
            print("="*60)
            
            result = rag.query(query, n_results=5)
            print(result)
            print("="*60 + "\n")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}\n")

if __name__ == "__main__":
    main()
