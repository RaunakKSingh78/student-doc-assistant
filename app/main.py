import os
import sys
from dotenv import load_dotenv

# Add the project root directory to the python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# pyrefly: ignore [missing-import]
from src.ingestion import load_all_documents, save_chunks_to_json, load_chunks_from_json, EmbeddingPipeline
# pyrefly: ignore [missing-import]
from src.retriever import ChromaVectorStore
# pyrefly: ignore [missing-import]
from src.generator import RAGSearch

load_dotenv()

def main():
    chroma_path = os.path.join("chroma_store", "chroma.sqlite3")
    processed_dir = os.path.join("data", "processed")

    if not os.path.exists(chroma_path):
        print("[INFO] No existing Chroma store found. Initiating document ingestion and build...")
        
        # 1. Ingest raw files to JSON chunks if processed folder is empty/missing
        raw_dir = os.path.join("data", "raw")
        if not os.path.exists(processed_dir) or not os.listdir(processed_dir):
            print("[INFO] Processed chunks directory is empty. Running ingestion...")
            docs = load_all_documents(raw_dir)
            if not docs:
                print("[ERROR] No raw documents found in data/raw to ingest!")
                return
            emb_pipe = EmbeddingPipeline()
            chunks = emb_pipe.chunk_documents(docs)
            save_chunks_to_json(chunks, processed_dir)
            
        # 2. Load pre-computed JSON chunks
        chunks = load_chunks_from_json(processed_dir)
        if not chunks:
            print("[ERROR] No chunk files loaded. Cannot build vector store.")
            return
            
        # 3. Build Chroma vector store
        store = ChromaVectorStore("chroma_store")
        store.build_from_chunks(chunks)
    else:
        print("[INFO] Existing Chroma store found. Skipping rebuild.")

    rag_search = RAGSearch()
    
    while True:
        query = input("You: ")
        print(f"\n[INFO] Running RAG query: '{query}'")
        summary = rag_search.search_and_summarize(query, top_k=5)['answer']

        print("\n" + "="*60)
        print("Query:", query)
        print("="*60)
        print("Answer:", summary)
        print("="*60)

        cont = input("Do you want to continue? (y/n): ")
        if cont.lower() != "y":
            break


if __name__ == "__main__":
    main()


