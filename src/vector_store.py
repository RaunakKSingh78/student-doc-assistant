"""
src/vector_store.py
===================
Dense Vector Store implementation using ChromaDB and HuggingFace Embeddings.
"""

import os
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


class ChromaVectorStore:
    """
    Manages vector embeddings and similarity search using ChromaDB.
    """

    def __init__(
        self,
        persist_dir: str = "chroma_store",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.persist_dir = persist_dir
        self.embedding_model = embedding_model
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.db: Optional[Chroma] = None

    def build_from_chunks(self, chunks: List[Document]) -> None:
        """
        Build and persist vector store from Document chunks.
        """
        if not chunks:
            raise ValueError("[ERROR] No chunks provided. Cannot build vector store.")

        print(f"[INFO] Building vector store from {len(chunks)} chunks...")
        self.db = Chroma.from_documents(
            chunks,
            self.embeddings,
            persist_directory=self.persist_dir,
        )

    def load(self) -> None:
        """
        Load existing Chroma store from disk.
        """
        self.db = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
        )
        print(f"[INFO] Loaded Chroma store from '{self.persist_dir}'")

    def as_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None):
        """
        Return a LangChain compatible retriever interface.
        """
        if not self.db:
            self.load()
        return self.db.as_retriever(search_kwargs=search_kwargs or {"k": 5})

    def retrieve_context(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Retrieve top_k documents matching query with optional metadata filter.
        """
        if not self.db:
            self.load()

        search_kwargs: Dict[str, Any] = {"k": top_k}
        if metadata_filter:
            search_kwargs["filter"] = metadata_filter

        retriever = self.db.as_retriever(search_kwargs=search_kwargs)
        return retriever.invoke(query)


if __name__ == "__main__":
    import sys
    print("[TEST] Running vector_store.py individually...")
    store = ChromaVectorStore(persist_dir="chroma_store")
    if os.path.exists(os.path.join("chroma_store", "chroma.sqlite3")):
        store.load()
        results = store.retrieve_context("Data Structures and Algorithms", top_k=2)
        print(f"[SUCCESS] Retrieved {len(results)} docs from vector store.")
    else:
        print("[INFO] Vector store sqlite file not present yet. Skipping load test.")

