"""
src/hybrid_retriever.py
========================
Hybrid search combiner using Dense (Chroma Vector) + Sparse (BM25 Keyword) search
and Reciprocal Rank Fusion (RRF).
"""

import os
import sys
from typing import List, Dict, Optional, Any
from langchain_core.documents import Document

if __name__ == "__main__" or __package__ is None:
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

# pyrefly: ignore [missing-import]
from src.vector_store import ChromaVectorStore


class HybridRetriever:
    """
    Combines dense semantic search (ChromaVectorStore) with sparse keyword search (BM25)
    using Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, store: ChromaVectorStore, chunks: List[Document]):
        self.store = store
        self._chunks = chunks

        if not BM25_AVAILABLE:
            print("[WARNING] rank_bm25 is not installed. HybridRetriever will fall back to dense search.")
            self._bm25 = None
        else:
            tokenized = [doc.page_content.lower().split() for doc in chunks]
            self._bm25 = BM25Okapi(tokenized)
            print(f"[INFO] BM25 index built over {len(chunks)} chunks.")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Retrieve top_k documents using RRF hybrid search.
        alpha = weight for semantic search (0.5 means equal weight for semantic & BM25).
        """
        if not BM25_AVAILABLE or self._bm25 is None or not self._chunks:
            return self.store.retrieve_context(query, top_k=top_k, metadata_filter=metadata_filter)

        fetch_k = min(top_k * 3, len(self._chunks))
        semantic_docs = self.store.retrieve_context(query, top_k=fetch_k, metadata_filter=metadata_filter)

        tokenized_query = query.lower().split()
        bm25_scores = self._bm25.get_scores(tokenized_query)

        # Apply metadata filter to candidate BM25 pool if requested
        candidate_indices = range(len(self._chunks))
        if metadata_filter:
            filtered_indices = []
            for idx, chunk in enumerate(self._chunks):
                match = True
                for k, v in metadata_filter.items():
                    if chunk.metadata.get(k) != v:
                        match = False
                        break
                if match:
                    filtered_indices.append(idx)
            if filtered_indices:
                candidate_indices = filtered_indices

        bm25_top_indices = sorted(
            candidate_indices,
            key=lambda i: bm25_scores[i],
            reverse=True,
        )[:fetch_k]
        bm25_docs = [self._chunks[i] for i in bm25_top_indices]

        # Reciprocal Rank Fusion (RRF) with constant K=60
        K = 60
        rrf_scores: Dict[str, float] = {}

        for rank, doc in enumerate(semantic_docs):
            key = doc.page_content
            rrf_scores[key] = rrf_scores.get(key, 0.0) + alpha * (1.0 / (K + rank + 1))

        for rank, doc in enumerate(bm25_docs):
            key = doc.page_content
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 - alpha) * (1.0 / (K + rank + 1))

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        content_to_doc: Dict[str, Document] = {
            doc.page_content: doc for doc in semantic_docs + bm25_docs
        }

        return [
            content_to_doc[content]
            for content, _ in ranked
            if content in content_to_doc
        ]


if __name__ == "__main__":
    import os
    import sys

    if __name__ == "__main__" or __package__ is None:
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

    # pyrefly: ignore [missing-import]
    from src.ingestion import load_chunks_from_json

    print("[TEST] Running hybrid_retriever.py individually...")
    chunks = load_chunks_from_json("data/processed")
    store = ChromaVectorStore(persist_dir="chroma_store")
    if os.path.exists(os.path.join("chroma_store", "chroma.sqlite3")):
        store.load()
    else:
        if chunks:
            store.build_from_chunks(chunks)

    hybrid = HybridRetriever(store, chunks)
    results = hybrid.retrieve("Data Structures", top_k=2)
    print(f"[SUCCESS] HybridRetriever retrieved {len(results)} docs.")

