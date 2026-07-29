"""
src/retriever.py
================

"""

import os
import sys
from typing import List, Optional, Dict

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__" or __package__ is None:
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)


try:
    from src.ingestion import EmbeddingPipeline, load_chunks_from_json
    INGESTION_AVAILABLE = True
except ImportError:
    EmbeddingPipeline = None
    load_chunks_from_json = None
    INGESTION_AVAILABLE = False
    print("[WARNING] src.ingestion not found. Run ingestion.py first.")

# BM25 is optional — only needed for HybridRetriever
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False


class ChromaVectorStore:
    def __init__(
        self,
        persist_dir: str = "chroma_store",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize the Chroma Vector Store.

        """
        self.persist_dir = persist_dir

        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.db = None

    def build_from_chunks(self, chunks: List[Document]) -> None:
        """
        Build the vector store from pre-computed chunks.

        """
        if not chunks:
            raise ValueError("[ERROR] No chunks provided. Cannot build vector store.")

        print(f"[INFO] Building vector store from {len(chunks)} pre-computed chunks...")
        self.db = Chroma.from_documents(
            chunks,
            self.embeddings,
            persist_directory=self.persist_dir,
        )

    def build_from_documents(self, documents: List[Document]) -> None:
        """Fallback helper: chunk documents using ingestion and index them."""
        if not INGESTION_AVAILABLE:
            raise ImportError("src.ingestion not available. Use build_from_chunks().")
        pipeline = EmbeddingPipeline()
        chunks = pipeline.chunk_documents(documents)
        self.build_from_chunks(chunks)

    def save(self) -> None:
        """Chroma automatically persists. save() kept for compatibility."""
        print("[INFO] Chroma automatically persists; save() called for compatibility.")

    def load(self) -> None:
        """Load the vector store from disk."""
        self.db = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
        )
        print(f"[INFO] Loaded Chroma store from '{self.persist_dir}'")

    def as_retriever(self, search_kwargs: dict = None):
        """Return a LangChain-compatible retriever (used in RAG chains)."""
        if not self.db:
            self.load()
        return self.db.as_retriever(search_kwargs=search_kwargs or {"k": 5})

    def retrieve_context(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Convert the user query to embeddings and retrieve relevant
        context documents from the store.
        """
        if not self.db:
            self.load()
        retriever = self.db.as_retriever(search_kwargs={"k": top_k})
        return retriever.invoke(query)


# ──────────────────────────────────────────────────────────────────────────────
#  HybridRetriever
#
#  WHY ADD THIS?
#  ChromaVectorStore uses semantic (meaning-based) search only.
#  It works well for conceptual queries but misses exact terms:
#    → "CGPA 6.0", "October 15", "Rs. 50,000"
#
#  BM25 is a keyword search algorithm — it finds chunks that contain
#  the exact words in the query. Combining both with RRF (Reciprocal
#  Rank Fusion) covers far more query types than either method alone.
#
#  INSTALL: pip install rank-bm25
# ──────────────────────────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Combines ChromaVectorStore (semantic) + BM25 (keyword) search
    using Reciprocal Rank Fusion (RRF).

    """

    def __init__(self, store: ChromaVectorStore, chunks: List[Document]):

        self.store  = store
        self._chunks = chunks

        if not BM25_AVAILABLE:
            print("[HybridRetriever] rank_bm25 not installed.")
            print("  Run: pip install rank-bm25")
            print("  Falling back to semantic-only search until installed.")
            self._bm25 = None
        else:
            # Tokenize each chunk by splitting into lowercase words
            tokenized = [doc.page_content.lower().split() for doc in chunks]
            self._bm25 = BM25Okapi(tokenized)
            print(f"[HybridRetriever] BM25 index built over {len(chunks)} chunks.")

    def retrieve(
        self,
        query:  str,
        top_k:  int   = 5,
        alpha:  float = 0.5,
    ) -> List[Document]:
        """
        Retrieve top_k most relevant chunks using hybrid search.

        """
        # If BM25 not installed, fall back to pure semantic search
        if not BM25_AVAILABLE or self._bm25 is None:
            return self.store.retrieve_context(query, top_k=top_k)

        
        # Fetch more than top_k so RRF has enough candidates to re-rank
        fetch_k         = min(top_k * 3, len(self._chunks))
        semantic_docs   = self.store.retrieve_context(query, top_k=fetch_k)

        
        tokenized_query = query.lower().split()
        bm25_scores     = self._bm25.get_scores(tokenized_query)

        # Sort all chunk indices by BM25 score, take top fetch_k
        bm25_top_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )[:fetch_k]
        bm25_docs = [self._chunks[i] for i in bm25_top_indices]

        # Reciprocal Rank Fusion (RRF) ──────────────────────────────
        # For each chunk:
        #   score = alpha × 1/(60 + semantic_rank)
        #         + (1-alpha) × 1/(60 + bm25_rank)
        # K=60 is the standard RRF constant — don't change it
        K = 60
        rrf_scores: Dict[str, float] = {}

        for rank, doc in enumerate(semantic_docs):
            key = doc.page_content
            rrf_scores[key] = rrf_scores.get(key, 0.0) + alpha * (1.0 / (K + rank + 1))

        for rank, doc in enumerate(bm25_docs):
            key = doc.page_content
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 - alpha) * (1.0 / (K + rank + 1))

        # Sort by combined RRF score, take top_k
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Map content strings back to their Document objects
        content_to_doc: Dict[str, Document] = {
            doc.page_content: doc
            for doc in semantic_docs + bm25_docs
        }

        return [
            content_to_doc[content]
            for content, _ in ranked
            if content in content_to_doc
        ]


# ──────────────────────────────────────────────────────────────────────────────
#  PER-QUERY EVALUATION REPORT GENERATOR
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_query_retrieval(
    query: str,
    retrieved_docs: List[Document],
    latency_ms: float,
    mode: str = "Hybrid (Semantic + BM25)",
    alpha: float = 0.5,
    top_k: int = 5,
) -> Dict:
    """
    Generate an evaluation report for a single query retrieval execution.

    """
    retrieved_count = len(retrieved_docs)
    total_chars = sum(len(doc.page_content) for doc in retrieved_docs)
    est_tokens = int(total_chars / 4)

    # Collect source files
    sources = []
    for doc in retrieved_docs:
        src = doc.metadata.get("source") or doc.metadata.get("filename") or "unknown"
        if isinstance(src, str):
            src = os.path.basename(src)
        sources.append(src)

    source_counts: Dict[str, int] = {}
    for s in sources:
        source_counts[s] = source_counts.get(s, 0) + 1

    # Check keyword alignment / relevance estimate
    query_words = set(w.lower() for w in query.split() if len(w) > 3)
    matching_chunks = 0
    for doc in retrieved_docs:
        content_lower = doc.page_content.lower()
        if any(qw in content_lower for qw in query_words):
            matching_chunks += 1

    keyword_alignment_pct = round((matching_chunks / retrieved_count * 100), 1) if retrieved_count > 0 else 0.0

    if keyword_alignment_pct >= 60 and retrieved_count >= min(3, top_k):
        confidence = "High"
    elif keyword_alignment_pct >= 30 or retrieved_count > 0:
        confidence = "Medium"
    else:
        confidence = "Low"

    summary_text = (
        f"Retrieval Evaluation Report:\n"
        f"- Strategy: {mode} (alpha={alpha})\n"
        f"- Retrieval Latency: {latency_ms:.2f} ms\n"
        f"- Passages Retrieved: {retrieved_count} of top_k={top_k}\n"
        f"- Unique Document Sources: {len(source_counts)} ({', '.join(source_counts.keys()) if source_counts else 'None'})\n"
        f"- Context Volume: {total_chars} chars (~{est_tokens} tokens)\n"
        f"- Relevance Confidence: {confidence} (Keyword alignment: {keyword_alignment_pct}%)"
    )

    return {
        "query": query,
        "latency_ms": round(latency_ms, 2),
        "mode": mode,
        "alpha": alpha,
        "top_k": top_k,
        "retrieved_count": retrieved_count,
        "total_characters": total_chars,
        "estimated_tokens": est_tokens,
        "unique_sources": len(source_counts),
        "source_distribution": source_counts,
        "sources": list(source_counts.keys()),
        "keyword_alignment_pct": keyword_alignment_pct,
        "confidence": confidence,
        "summary_text": summary_text,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Quick test — run directly:  python src/retriever.py
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    if not INGESTION_AVAILABLE:
        print("[ERROR] src.ingestion not found. Run ingestion.py first.")
        sys.exit(1)

    # Load chunks produced by ingestion pipeline
    processed_dir = os.path.join("data", "processed")
    chunks = load_chunks_from_json(processed_dir)

    if not chunks:
        print("[WARNING] No chunks in data/processed/. Run ingestion.py first.")
        sys.exit(1)

    # Build and load the vector store
    store = ChromaVectorStore("chroma_store")
    store.build_from_chunks(chunks)
    store.load()

    query = input("\nEnter your query: ")

    import time
    start_time = time.perf_counter()

    if BM25_AVAILABLE:
        hybrid = HybridRetriever(store, chunks)
        docs = hybrid.retrieve(query, top_k=5, alpha=0.5)
        mode = "Hybrid (Semantic + BM25)"
    else:
        docs = store.retrieve_context(query, top_k=5)
        mode = "Semantic (ChromaDB)"

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    report = evaluate_query_retrieval(query, docs, elapsed_ms, mode=mode, top_k=5)

    print("\n" + "=" * 60)
    print(report["summary_text"])
    print("=" * 60)
    print("\nTop Retrieved Passages:")
    for i, doc in enumerate(docs, 1):
        src = doc.metadata.get('source') or doc.metadata.get('filename') or 'unknown'
        print(f"\n[{i}] {os.path.basename(src)}")
        print(doc.page_content[:200] + "...")

"""
notebooks/benchmark_retriever.py
=================================
Retrieval Benchmarking Script

Measures two things for both ChromaVectorStore and HybridRetriever:
  1. SPEED   — how many milliseconds each query takes
  2. ACCURACY — hit rate and MRR on your test query set

HOW TO RUN:
    python notebooks/benchmark_retriever.py

BEFORE RUNNING:
    - ingestion.py must have been run (data/processed/ must have JSON files)
    - retriever.py must be at src/retriever.py
    - pip install rank-bm25  (for hybrid search)

WHAT YOU'LL SEE:
    A side-by-side table comparing Semantic vs Hybrid on speed and accuracy.
    Failed queries are printed so you know exactly which ones to fix.
"""

import os
import sys
import time
from typing import List, Dict, Optional

# ── Path setup ────────────────────────────────────────────────────────────────
# Makes sure Python can find src/ from this notebooks/ location
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.retriever import ChromaVectorStore, HybridRetriever
from src.ingestion import load_chunks_from_json


# ──────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION — edit these before running
# ──────────────────────────────────────────────────────────────────────────────

PROCESSED_DIR = os.path.join("data", "processed")
CHROMA_DIR    = "chroma_store"
EMBED_MODEL   = "all-MiniLM-L6-v2"
TOP_K         = 8     # how many results to retrieve per query
ALPHA         = 0.5   # hybrid weight: 0.5 = equal semantic + BM25

# ── TEST QUERIES ──────────────────────────────────────────────────────────────
# These are the queries used to measure accuracy.
# "expected_keyword" must appear somewhere in the correct chunk's text.
# Edit these to match YOUR domain's actual documents.
# Add all 20 of these for the final evaluation report (Week 7).

TEST_QUERIES = [
    {
        "question": "What are the charges to be paid by students to stay in hostel during vacations for 15 days?",
        "expected_keyword": "Rs. 1500",
    },
    {
        "question": "What is the minimum CPI required for branch change of a GEN student?",
        "expected_keyword": "6.5",
    },
    {
        "question": "What minimum percentage of attendance is required to get full attendance marks?",
        "expected_keyword": "80",
    },
    {
        "question": "What could be the fine for ragging at IIT Indore?",
        "expected_keyword": "Rs. 25000",
    },
    {
        "question": "How much percentage overall is the weightage of attendance?",
        "expected_keyword": "10%",
    },
    {
        "question": "Can I attend interviews after accepting an offer?",
        "expected_keyword": "de-registered",
    },
    {
        "question": "What happens if I miss the registration deadline?",
        "expected_keyword": "september 30",
    },
    {
        "question": "What documents must I carry to a placement drive?",
        "expected_keyword": "id card",
    },
    # ── Add more queries below to match your actual documents ──────────────
    # {
    #     "question": "your question here",
    #     "expected_keyword": "keyword that must appear in the correct chunk",
    # },
]


def measure_speed(retriever_fn, queries: List[str], top_k: int) -> Dict:
    """
    Run each query through the retriever and record how long it takes.

    """
    times = []
    for query in queries:
        start  = time.perf_counter()
        retriever_fn(query, top_k)
        end    = time.perf_counter()
        elapsed_ms = (end - start) * 1000
        times.append(round(elapsed_ms, 2))

    return {
        "times_ms":  times,
        "avg_ms":    round(sum(times) / len(times), 2),
        "min_ms":    round(min(times), 2),
        "max_ms":    round(max(times), 2),
        "total_ms":  round(sum(times), 2),
    }


def measure_accuracy(retriever_fn, test_queries: List[Dict], top_k: int) -> Dict:
    """
    Measure hit rate and MRR for a set of labelled test queries.

    """
    hits             = 0
    reciprocal_ranks = []
    details          = []

    for item in test_queries:
        question = item["question"]
        keyword  = item["expected_keyword"].lower().strip()
        results  = retriever_fn(question, top_k)

        hit_rank: Optional[int] = None
        for rank, doc in enumerate(results, start=1):
            if keyword in doc.page_content.lower():
                hit_rank = rank
                break

        if hit_rank:
            hits += 1
            reciprocal_ranks.append(1.0 / hit_rank)
        else:
            reciprocal_ranks.append(0.0)

        details.append({
            "question":    question,
            "keyword":     keyword,
            "found":       hit_rank is not None,
            "rank":        hit_rank,
            "top_preview": results[0].page_content[:80].replace("\n", " ") if results else "",
        })

    total = len(test_queries)
    return {
        "hit_rate": round(hits / total, 3) if total else 0,
        "mrr":      round(sum(reciprocal_ranks) / total, 3) if total else 0,
        "hits":     hits,
        "total":    total,
        "details":  details,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  REPORT PRINTING
# ──────────────────────────────────────────────────────────────────────────────

def print_speed_report(name: str, speed: Dict, queries: List[str]) -> None:
    print(f"\n  Per-query breakdown:")
    for i, (q, t) in enumerate(zip(queries, speed["times_ms"]), 1):
        bar = "█" * int(t / 5)   # 1 block per 5ms
        print(f"    Q{i:02d} {t:6.1f}ms  {bar}  {q[:55]}...")


def print_report(
    semantic_speed:    Dict,
    hybrid_speed:      Dict,
    semantic_accuracy: Dict,
    hybrid_accuracy:   Dict,
    queries:           List[str],
    has_bm25:          bool,
) -> None:

    W = 62
    print("\n" + "=" * W)
    print("  RETRIEVAL BENCHMARK REPORT")
    print("=" * W)

    # ── Speed comparison ──────────────────────────────────────────────────────
    print("\n  SPEED")
    print(f"  {'Metric':<22} {'Semantic':>12} {'Hybrid':>12}")
    print(f"  {'-'*22} {'-'*12} {'-'*12}")

    metrics = [
        ("Avg per query",  f"{semantic_speed['avg_ms']}ms",   f"{hybrid_speed['avg_ms']}ms"),
        ("Fastest query",  f"{semantic_speed['min_ms']}ms",   f"{hybrid_speed['min_ms']}ms"),
        ("Slowest query",  f"{semantic_speed['max_ms']}ms",   f"{hybrid_speed['max_ms']}ms"),
        ("Total all qs",   f"{semantic_speed['total_ms']}ms", f"{hybrid_speed['total_ms']}ms"),
    ]
    for label, sem, hyb in metrics:
        print(f"  {label:<22} {sem:>12} {hyb:>12}")

    overhead = round(hybrid_speed["avg_ms"] - semantic_speed["avg_ms"], 2)
    sign     = "+" if overhead >= 0 else ""
    print(f"\n  Hybrid overhead: {sign}{overhead}ms per query avg")
    if overhead < 50:
        print("  → Acceptable for a demo (under 50ms extra)")
    else:
        print("  → Consider reducing top_k to speed up")

    # ── Accuracy comparison ───────────────────────────────────────────────────
    print(f"\n  ACCURACY  (top_k={TOP_K})")
    print(f"  {'Metric':<22} {'Semantic':>12} {'Hybrid':>12} {'Target':>10}")
    print(f"  {'-'*22} {'-'*12} {'-'*12} {'-'*10}")

    hit_s    = f"{semantic_accuracy['hit_rate']*100:.1f}%"
    hit_h    = f"{hybrid_accuracy['hit_rate']*100:.1f}%"
    mrr_s    = f"{semantic_accuracy['mrr']:.3f}"
    mrr_h    = f"{hybrid_accuracy['mrr']:.3f}"
    count_s  = f"{semantic_accuracy['hits']}/{semantic_accuracy['total']}"
    count_h  = f"{hybrid_accuracy['hits']}/{hybrid_accuracy['total']}"

    print(f"  {'Hit Rate @ k':<22} {hit_s:>12} {hit_h:>12} {'> 80%':>10}")
    print(f"  {'MRR':<22} {mrr_s:>12} {mrr_h:>12} {'> 0.60':>10}")
    print(f"  {'Correct queries':<22} {count_s:>12} {count_h:>12}")

    # ── Which is better ───────────────────────────────────────────────────────
    print(f"\n  VERDICT")
    sem_hr = semantic_accuracy["hit_rate"]
    hyb_hr = hybrid_accuracy["hit_rate"]

    if not has_bm25:
        print("  Hybrid not fully tested — install rank-bm25 first")
    elif hyb_hr > sem_hr:
        diff = round((hyb_hr - sem_hr) * 100, 1)
        print(f"  Hybrid is better by {diff} percentage points on hit rate ✓")
        print(f"  Recommendation: use HybridRetriever in generator.py")
    elif hyb_hr == sem_hr:
        print(f"  Both methods tied on accuracy")
        print(f"  Use Hybrid anyway — it's more robust on edge cases")
    else:
        diff = round((sem_hr - hyb_hr) * 100, 1)
        print(f"  Semantic wins by {diff} pp — try adjusting alpha in CONFIG")
        print(f"  Try alpha=0.7 (trust semantic more) and re-run")

    # ── Failed queries ────────────────────────────────────────────────────────
    sem_fails = [d for d in semantic_accuracy["details"] if not d["found"]]
    hyb_fails = [d for d in hybrid_accuracy["details"]  if not d["found"]]

    if sem_fails or hyb_fails:
        print(f"\n  FAILED QUERIES")

        all_failed_qs = set(
            d["question"] for d in sem_fails + hyb_fails
        )
        for q in all_failed_qs:
            sem_ok = q not in [d["question"] for d in sem_fails]
            hyb_ok = q not in [d["question"] for d in hyb_fails]
            sem_tag = "✓ pass" if sem_ok else "✗ miss"
            hyb_tag = "✓ pass" if hyb_ok else "✗ miss"
            # find keyword
            kw = next(
                (d["keyword"] for d in semantic_accuracy["details"] if d["question"] == q),
                "?"
            )
            print(f"\n    Q: {q}")
            print(f"       keyword expected: '{kw}'")
            print(f"       Semantic: {sem_tag}   Hybrid: {hyb_tag}")

            if not sem_ok:
                preview = next(
                    (d["top_preview"] for d in semantic_accuracy["details"] if d["question"] == q),
                    ""
                )
                print(f"       Top result was: {preview}...")

        print(f"\n  HOW TO FIX FAILED QUERIES:")
        print(f"    1. The chunk for this answer may be missing — check data/raw/")
        print(f"    2. The chunk size may be too large — ask to reduce chunk_size")
        print(f"    3. Try alpha=0.3 (more keyword weight) for exact-term queries")

    else:
        print(f"\n  All {semantic_accuracy['total']} queries passed on both methods ✓")

    # ── Per-query speed breakdown ─────────────────────────────────────────────
    print(f"\n  PER-QUERY SPEED (Semantic vs Hybrid)")
    print(f"  {'#':<4} {'Semantic':>10} {'Hybrid':>10}  Query")
    print(f"  {'-'*4} {'-'*10} {'-'*10}  {'-'*40}")
    for i, (q, st, ht) in enumerate(
        zip(queries, semantic_speed["times_ms"], hybrid_speed["times_ms"]), 1
    ):
        print(f"  {i:<4} {st:>8.1f}ms {ht:>8.1f}ms  {q[:45]}...")

    print("\n" + "=" * W + "\n")


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "=" * 62)
    print("  Retriever Benchmark ")
    print("=" * 62)

    # ── 1. Load chunks ────────────────────────────────────────────────────────
    print(f"\n[1/4] Loading chunks from '{PROCESSED_DIR}' ...")
    chunks = load_chunks_from_json(PROCESSED_DIR)

    if not chunks:
        print("[ERROR] No chunks found. Run ingestion.py first.")
        sys.exit(1)

    print(f"      Loaded {len(chunks)} chunks.")

    # ── 2. Build / load retrievers ────────────────────────────────────────────
    print(f"\n[2/4] Setting up retrievers ...")

    store = ChromaVectorStore(CHROMA_DIR, EMBED_MODEL)
    if not os.path.exists(os.path.join(CHROMA_DIR, "chroma.sqlite3")):
        print("      Building ChromaDB (first time — may take ~1 min) ...")
        store.build_from_chunks(chunks)
    store.load()

    try:
        from rank_bm25 import BM25Okapi
        hybrid  = HybridRetriever(store, chunks)
        has_bm25 = True
    except ImportError:
        print("      rank_bm25 not installed — hybrid will mirror semantic results")
        print("      Run: pip install rank-bm25")
        hybrid   = None
        has_bm25 = False

    # Wrap retrieve methods into simple callables (query, top_k) → List[Document]
    def semantic_fn(query, top_k):
        return store.retrieve_context(query, top_k=top_k)

    def hybrid_fn(query, top_k):
        if hybrid:
            return hybrid.retrieve(query, top_k=top_k, alpha=ALPHA)
        return store.retrieve_context(query, top_k=top_k)

    # ── 3. Run speed benchmark ────────────────────────────────────────────────
    print(f"\n[3/4] Measuring speed ({len(TEST_QUERIES)} queries × 2 methods) ...")

    queries = [item["question"] for item in TEST_QUERIES]

    # Warm-up run (first call is always slower due to model loading)
    store.retrieve_context(queries[0], top_k=TOP_K)

    semantic_speed = measure_speed(semantic_fn, queries, TOP_K)
    hybrid_speed   = measure_speed(hybrid_fn,   queries, TOP_K)

    print(f"      Semantic avg: {semantic_speed['avg_ms']}ms/query")
    print(f"      Hybrid avg:   {hybrid_speed['avg_ms']}ms/query")

    # ── 4. Run accuracy benchmark ─────────────────────────────────────────────
    print(f"\n[4/4] Measuring accuracy (hit rate + MRR) ...")

    semantic_accuracy = measure_accuracy(semantic_fn, TEST_QUERIES, TOP_K)
    hybrid_accuracy   = measure_accuracy(hybrid_fn,   TEST_QUERIES, TOP_K)

    print(f"      Semantic hit rate: {semantic_accuracy['hit_rate']*100:.1f}%")
    print(f"      Hybrid hit rate:   {hybrid_accuracy['hit_rate']*100:.1f}%")

    # ── 5. Print full report ──────────────────────────────────────────────────
    print_report(
        semantic_speed,
        hybrid_speed,
        semantic_accuracy,
        hybrid_accuracy,
        queries,
        has_bm25,
    )
