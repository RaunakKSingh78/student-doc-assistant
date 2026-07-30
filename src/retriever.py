"""
src/retriever.py
================
Restructured Advanced RAG Retriever Facade.

Integrates:
1. ChromaVectorStore (Vector Search)
2. HybridRetriever (Dense + BM25 RRF Search)
3. QueryTransformer (Query Expansion, Rewriting, Multi-Query)
4. Reranker (Cross-Encoder refinement)
5. ParentDocumentRetriever (Child-to-Parent context expansion)
6. Metrics & Quality Evaluation (Latency, Accuracy, MRR)
"""

import os
import sys
import time
from typing import List, Dict, Optional, Any
from langchain_core.documents import Document

# Support running directly from command line
if __name__ == "__main__" or __package__ is None:
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

# pyrefly: ignore [missing-import]
from src.vector_store import ChromaVectorStore
# pyrefly: ignore [missing-import]
from src.hybrid_retriever import HybridRetriever
# pyrefly: ignore [missing-import]
from src.metrics import (
    evaluate_query_retrieval,
    measure_speed,
    measure_accuracy,
    print_benchmark_report,
)
# pyrefly: ignore [missing-import]
from src.query_transform import QueryTransformer
# pyrefly: ignore [missing-import]
from src.reranker import Reranker
# pyrefly: ignore [missing-import]
from src.parent_retriever import ParentDocumentRetriever

try:
    # pyrefly: ignore [missing-import]
    from src.ingestion import load_chunks_from_json
    INGESTION_AVAILABLE = True
except ImportError:
    load_chunks_from_json = None
    INGESTION_AVAILABLE = False


class AdvancedRAGRetriever:
    """
    Facade class that orchestrates the entire Advanced RAG Retrieval Pipeline:
    Query Transformation -> Hybrid Search -> Reranking -> Parent Context Retrieval.
    """

    def __init__(
        self,
        persist_dir: str = "chroma_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunks: Optional[List[Document]] = None,
        use_reranker: bool = True,
        use_query_transform: bool = True,
    ):
        self.vector_store = ChromaVectorStore(persist_dir, embedding_model)

        if not os.path.exists(os.path.join(persist_dir, "chroma.sqlite3")) and chunks:
            self.vector_store.build_from_chunks(chunks)

        self.vector_store.load()

        self.chunks = chunks or []
        self.hybrid_retriever: Optional[HybridRetriever] = None
        if self.chunks:
            self.hybrid_retriever = HybridRetriever(self.vector_store, self.chunks)

        self.query_transformer = QueryTransformer() if use_query_transform else None
        self.reranker = Reranker() if use_reranker else None
        self.parent_retriever = ParentDocumentRetriever()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Execute full Advanced RAG pipeline:
        1. Query Transformation (Multi-query generation)
        2. Hybrid Search (Vector + BM25) across query variations
        3. Reranking candidates using Cross-Encoder
        4. Resolving child chunks to Parent Documents
        """
        # Step 1: Query Transformation
        search_queries = [query]
        if self.query_transformer is not None:
            search_queries = self.query_transformer.generate_multi_queries(query, num_queries=3)

        # Step 2: Retrieval across search queries
        candidate_docs: List[Document] = []
        seen_texts = set()

        for q in search_queries:
            if self.hybrid_retriever is not None:
                docs = self.hybrid_retriever.retrieve(
                    query=q,
                    top_k=top_k * 3,
                    alpha=alpha,
                    metadata_filter=metadata_filter,
                )
            else:
                docs = self.vector_store.retrieve_context(
                    query=q,
                    top_k=top_k * 3,
                    metadata_filter=metadata_filter,
                )

            for d in docs:
                if d.page_content not in seen_texts:
                    seen_texts.add(d.page_content)
                    candidate_docs.append(d)

        # Step 3: Reranking candidate documents
        if self.reranker is not None and candidate_docs:
            refined_docs = self.reranker.rerank(query=query, documents=candidate_docs, top_n=top_k)
        else:
            refined_docs = candidate_docs[:top_k]

        # Step 4: Parent Document Resolution
        final_docs = self.parent_retriever.resolve_parents(refined_docs)

        return final_docs


# ──────────────────────────────────────────────────────────────────────────────
#  CLI Entry point for quick testing & benchmarking
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    processed_dir = os.path.join("data", "processed")
    chunks = load_chunks_from_json(processed_dir) if INGESTION_AVAILABLE else []

    if not chunks:
        print("[WARNING] No chunks found in data/processed. Build store first.")

    pipeline = AdvancedRAGRetriever(
        persist_dir="chroma_store",
        chunks=chunks,
    )

    query = input("\nEnter query to test Advanced RAG pipeline: ")
    start_time = time.perf_counter()

    results = pipeline.retrieve(query, top_k=5)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    report = evaluate_query_retrieval(
        query=query,
        retrieved_docs=results,
        latency_ms=elapsed_ms,
        mode="Advanced RAG (Multi-Query + Hybrid + Reranker)",
        top_k=5,
    )

    print("\n" + "=" * 60)
    print(report["summary_text"])
    print("=" * 60)

    print("\nTop Retrieved Passages:")
    for i, doc in enumerate(results, 1):
        src = doc.metadata.get("source") or doc.metadata.get("filename") or "unknown"
        print(f"\n[{i}] Source: {os.path.basename(str(src))}")
        print(doc.page_content[:300] + "...")
