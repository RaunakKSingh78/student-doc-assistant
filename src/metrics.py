"""
src/metrics.py
==============
Quality and performance evaluation functions for RAG retrieval.
Measures retrieval latency (speed), keyword alignment, hit rate, MRR (accuracy), and generates summary reports.
"""

import os
import time
from typing import List, Dict, Optional, Any
from langchain_core.documents import Document


def evaluate_query_retrieval(
    query: str,
    retrieved_docs: List[Document],
    latency_ms: float,
    mode: str = "Hybrid (Semantic + BM25)",
    alpha: float = 0.5,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Generate a detailed quality evaluation report for a single query retrieval step.
    """
    retrieved_count = len(retrieved_docs)
    total_chars = sum(len(doc.page_content) for doc in retrieved_docs)
    est_tokens = int(total_chars / 4)

    sources = []
    for doc in retrieved_docs:
        src = doc.metadata.get("source") or doc.metadata.get("filename") or "unknown"
        if isinstance(src, str):
            src = os.path.basename(src)
        sources.append(src)

    source_counts: Dict[str, int] = {}
    for s in sources:
        source_counts[s] = source_counts.get(s, 0) + 1

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


def measure_speed(retriever_fn: Any, queries: List[str], top_k: int) -> Dict[str, Any]:
    """
    Measure retrieval execution time (latency in ms) across multiple queries.
    """
    times = []
    for query in queries:
        start = time.perf_counter()
        retriever_fn(query, top_k)
        end = time.perf_counter()
        elapsed_ms = (end - start) * 1000
        times.append(round(elapsed_ms, 2))

    return {
        "times_ms": times,
        "avg_ms": round(sum(times) / len(times), 2) if times else 0.0,
        "min_ms": round(min(times), 2) if times else 0.0,
        "max_ms": round(max(times), 2) if times else 0.0,
        "total_ms": round(sum(times), 2) if times else 0.0,
    }


def measure_accuracy(retriever_fn: Any, test_queries: List[Dict[str, str]], top_k: int) -> Dict[str, Any]:
    """
    Measure retrieval accuracy (Hit Rate @ K and Mean Reciprocal Rank MRR) against ground truth expected keywords.
    """
    hits = 0
    reciprocal_ranks = []
    details = []

    for item in test_queries:
        question = item["question"]
        keyword = item["expected_keyword"].lower().strip()
        results = retriever_fn(question, top_k)

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
            "question": question,
            "keyword": keyword,
            "found": hit_rank is not None,
            "rank": hit_rank,
            "top_preview": results[0].page_content[:80].replace("\n", " ") if results else "",
        })

    total = len(test_queries)
    return {
        "hit_rate": round(hits / total, 3) if total else 0,
        "mrr": round(sum(reciprocal_ranks) / total, 3) if total else 0,
        "hits": hits,
        "total": total,
        "details": details,
    }


def print_benchmark_report(
    semantic_speed: Dict[str, Any],
    hybrid_speed: Dict[str, Any],
    semantic_accuracy: Dict[str, Any],
    hybrid_accuracy: Dict[str, Any],
    queries: List[str],
    top_k: int = 5,
) -> None:
    """
    Print a side-by-side comparative evaluation report for speed and accuracy.
    """
    W = 64
    print("\n" + "=" * W)
    print("  RAG RETRIEVAL BENCHMARK & QUALITY REPORT")
    print("=" * W)

    print("\n  1. SPEED BENCHMARK")
    print(f"  {'Metric':<22} {'Semantic':>12} {'Hybrid':>12}")
    print(f"  {'-'*22} {'-'*12} {'-'*12}")

    metrics = [
        ("Avg Latency / Query", f"{semantic_speed['avg_ms']}ms", f"{hybrid_speed['avg_ms']}ms"),
        ("Fastest Query", f"{semantic_speed['min_ms']}ms", f"{hybrid_speed['min_ms']}ms"),
        ("Slowest Query", f"{semantic_speed['max_ms']}ms", f"{hybrid_speed['max_ms']}ms"),
        ("Total Duration", f"{semantic_speed['total_ms']}ms", f"{hybrid_speed['total_ms']}ms"),
    ]
    for label, sem, hyb in metrics:
        print(f"  {label:<22} {sem:>12} {hyb:>12}")

    print(f"\n  2. ACCURACY BENCHMARK (top_k={top_k})")
    print(f"  {'Metric':<22} {'Semantic':>12} {'Hybrid':>12} {'Target':>10}")
    print(f"  {'-'*22} {'-'*12} {'-'*12} {'-'*10}")

    hit_s = f"{semantic_accuracy['hit_rate']*100:.1f}%"
    hit_h = f"{hybrid_accuracy['hit_rate']*100:.1f}%"
    mrr_s = f"{semantic_accuracy['mrr']:.3f}"
    mrr_h = f"{hybrid_accuracy['mrr']:.3f}"
    count_s = f"{semantic_accuracy['hits']}/{semantic_accuracy['total']}"
    count_h = f"{hybrid_accuracy['hits']}/{hybrid_accuracy['total']}"

    print(f"  {'Hit Rate @ k':<22} {hit_s:>12} {hit_h:>12} {'> 80%':>10}")
    print(f"  {'MRR':<22} {mrr_s:>12} {mrr_h:>12} {'> 0.60':>10}")
    print(f"  {'Correct Queries':<22} {count_s:>12} {count_h:>12}")

    print("\n" + "=" * W + "\n")


if __name__ == "__main__":
    print("[TEST] Running metrics.py individually...")
    sample_doc = Document(page_content="Data Structures and Algorithms syllabus includes arrays, stacks, queues, trees, and graphs.")
    report = evaluate_query_retrieval("Data Structures syllabus", [sample_doc], latency_ms=12.5)
    print(f"[SUCCESS] Evaluation report generated successfully:\n{report['summary_text']}")

