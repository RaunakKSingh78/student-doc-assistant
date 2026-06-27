import os
import sys
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Add parent directory to sys.path to support running this file directly
if __name__ == "__main__" or __package__ is None:
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

# pyrefly: ignore [missing-import]
from src.ingestion import EmbeddingPipeline
# pyrefly: ignore [missing-import]
from src.ingestion import load_chunks_from_json
from dotenv import load_dotenv

load_dotenv()

class ChromaVectorStore:

    def __init__(self, persist_dir: str = "chroma_store", embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the Chroma Vector Store.
        Args:
            persist_dir (str): Directory to persist the vector store.
            embedding_model (str): Model to use for embeddings.
        """
        self.persist_dir = persist_dir
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"local_files_only": True}
        )
        self.db = None

    def build_from_chunks(self, chunks: List[Document]):
        """
        Build the vector store from pre-computed chunks.
        Args:
            chunks: List of LangChain Document objects to build the store from.
        """
        if not chunks:
            raise ValueError("[ERROR] No chunks provided. Cannot build vector store.")
        
        print(f"[INFO] Building vector store from {len(chunks)} pre-computed chunks...")
        self.db = Chroma.from_documents(
            chunks,  
            self.embeddings,
            persist_directory=self.persist_dir
        )

    def build_from_documents(self, documents: List[Document]):
        # Fallback helper: chunk documents using ingestion and index them
        pipeline = EmbeddingPipeline()
        chunks = pipeline.chunk_documents(documents)
        self.build_from_chunks(chunks)

    def save(self):
        print(f"[INFO] Chroma automatically persists; save() called for compatibility.")

    def load(self):
        """
        Load the vector store from disk.
        """
        self.db = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings
        )
        print(f"[INFO] Loaded Chroma store from '{self.persist_dir}'")

    def as_retriever(self, search_kwargs: dict = None):
        """ Return a LangChain-compatible retriever (used in RAG chains). """
        if not self.db:
            self.load()
        return self.db.as_retriever(search_kwargs=search_kwargs or {"k": 5})

    def retrieve_context(self, query: str, top_k: int = 5) -> List[Document]:
        """Convert the user query to embeddings and retrieve relevant context documents from the store."""
        if not self.db:
            self.load()
        retriever = self.db.as_retriever(search_kwargs={"k": top_k})
        return retriever.invoke(query)


if __name__ == "__main__":

    # 1. Load the pre-computed JSON chunks from data/processed
    processed_dir = os.path.join("data", "processed")
    chunks = load_chunks_from_json(processed_dir)

    if not chunks:
        print("[WARNING] No chunks found in processed directory. Please run ingestion.py first.")
    else:
        # 2. Build the vector store from those chunks
        store = ChromaVectorStore("chroma_store")
        store.build_from_chunks(chunks)
        store.load()

        # 3. Retrieve context for a query
        query = input("Enter your query:")
        print(f"\n[INFO] Retrieving context for query: '{query}'")
        
        results = store.retrieve_context(query, top_k=3)
        for i, doc in enumerate(results):
            print(f"\n--- Result {i+1} ---\n{doc.page_content[:300]}")


