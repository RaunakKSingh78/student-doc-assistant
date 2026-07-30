"""
src/reranker.py
===============
Cross-Encoder Reranking module to refine candidate passages.
Passes candidate chunks (e.g., top 20) through a CrossEncoder model
to compute precise query-document relevance scores and pick the top N passages.
"""

from typing import List, Tuple
from langchain_core.documents import Document

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False


class Reranker:
    """
    Reranks retrieved candidate documents using a Cross-Encoder model.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None

        if CROSS_ENCODER_AVAILABLE:
            try:
                self.model = CrossEncoder(model_name)
                print(f"[INFO] Loaded CrossEncoder reranker model: {model_name}")
            except Exception as e:
                print(f"[WARNING] Failed to load CrossEncoder model '{model_name}': {e}. Using score fallback.")
        else:
            print("[WARNING] sentence_transformers CrossEncoder not available. Using term overlap fallback.")

    def rerank(self, query: str, documents: List[Document], top_n: int = 5) -> List[Document]:
        """
        Score query-document pairs and return top_n most relevant documents.
        """
        if not documents:
            return []

        if len(documents) <= top_n:
            top_candidate_limit = len(documents)
        else:
            top_candidate_limit = min(20, len(documents))

        candidates = documents[:top_candidate_limit]

        if self.model is not None:
            try:
                pairs = [[query, doc.page_content] for doc in candidates]
                scores = self.model.predict(pairs)

                scored_docs: List[Tuple[float, Document]] = list(zip(scores, candidates))
                scored_docs.sort(key=lambda x: x[0], reverse=True)

                return [doc for _, doc in scored_docs[:top_n]]
            except Exception as e:
                print(f"[WARNING] CrossEncoder predict error: {e}. Falling back to keyword rank.")

        # Fallback term-matching reranker
        query_words = set(query.lower().split())
        fallback_scored = []
        for doc in candidates:
            words = doc.page_content.lower().split()
            overlap = sum(1 for w in words if w in query_words)
            fallback_scored.append((overlap, doc))

        fallback_scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in fallback_scored[:top_n]]


if __name__ == "__main__":
    print("[TEST] Running reranker.py individually...")
    reranker = Reranker()
    docs = [
        Document(page_content="Introductory Data Structures and Algorithms course covers trees and graphs."),
        Document(page_content="Hostel mess rules and fee structure guidelines.")
    ]
    reranked = reranker.rerank("Data Structures syllabus", docs, top_n=1)
    print(f"[SUCCESS] Top reranked doc: {reranked[0].page_content[:60]}...")

