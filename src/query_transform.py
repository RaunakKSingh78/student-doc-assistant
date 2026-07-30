"""
src/query_transform.py
======================
Query Transformation module for Advanced RAG.
Includes:
- Query Expansion (synonyms / domain expansion)
- Query Rewriting (clarifying vague user questions)
- Multi-Query Retrieval (generating multiple search variants)
"""

import os
from typing import List, Optional, Any
from dotenv import load_dotenv

load_dotenv()


class QueryTransformer:
    """
    Handles Query Expansion, Query Rewriting, and Multi-Query Generation
    to optimize raw student questions before retrieval.
    """

    def __init__(self, llm: Optional[Any] = None):
        self.llm = llm

    def expand_query(self, query: str) -> str:
        """
        Expand terms and abbreviations in the question (e.g. AI -> Artificial Intelligence, Machine Learning, etc.).
        """
        expansions = {
            "ai": "Artificial Intelligence, Machine Learning, Deep Learning, Neural Networks",
            "ml": "Machine Learning, Statistical Learning, Data Mining",
            "cse": "Computer Science and Engineering",
            "cpi": "Cumulative Performance Index CGPA grade requirement",
            "gpa": "Grade Point Average CPI marks",
            "fee": "hostel fee tuition fee charges dues payment",
            "coding": "Computer Science programming algorithms introductory course CS101",
            "robots": "Robotics, Automation, Mechanical Engineering, Electronics",
        }

        words = query.lower().split()
        expanded_parts = [query]

        for w in words:
            clean_word = w.strip("?,.!")
            if clean_word in expansions:
                expanded_parts.append(expansions[clean_word])

        return " | ".join(expanded_parts)

    def rewrite_query(self, query: str) -> str:
        """
        Rewrite vague or informal questions into explicit academic/administrative search queries.
        """
        if self.llm is not None:
            try:
                prompt = (
                    f"Rewrite the following student question into a clear, precise academic query for document search:\n"
                    f"Original: {query}\n"
                    f"Rewritten Query:"
                )
                res = self.llm.invoke(prompt)
                rewritten = getattr(res, "content", str(res)).strip()
                if rewritten:
                    return rewritten
            except Exception as e:
                print(f"[WARNING] LLM query rewriting failed, using rule-based fallback: {e}")

        # Rule-based fallback rewriting heuristics
        q_lower = query.lower()
        if "coding class" in q_lower or "start coding" in q_lower:
            return "prerequisites and registration steps for Computer Science introductory programming courses"
        if "robots" in q_lower or "robotics" in q_lower:
            return "introductory robotics mechanical engineering and electronics courses for beginners"
        if "hostel" in q_lower and "vacation" in q_lower:
            return "hostel charges fee per day stay during vacation"
        if "branch change" in q_lower:
            return "minimum CPI required for branch change criteria"

        return query

    def generate_multi_queries(self, query: str, num_queries: int = 3) -> List[str]:
        """
        Generate multiple search query variations for Multi-Query Retrieval.
        """
        queries = [query]

        rewritten = self.rewrite_query(query)
        if rewritten not in queries:
            queries.append(rewritten)

        expanded = self.expand_query(query)
        if expanded not in queries:
            queries.append(expanded)

        if self.llm is not None and len(queries) < num_queries:
            try:
                prompt = (
                    f"Generate {num_queries - 1} different perspectives or search query variations for the question: '{query}'. "
                    f"Provide one query per line."
                )
                res = self.llm.invoke(prompt)
                lines = getattr(res, "content", str(res)).strip().split("\n")
                for line in lines:
                    clean_line = line.strip(" 123456789.-*")
                    if clean_line and clean_line not in queries:
                        queries.append(clean_line)
            except Exception as e:
                print(f"[WARNING] Multi-query LLM generation fallback: {e}")

        return queries[:num_queries]


if __name__ == "__main__":
    print("[TEST] Running query_transform.py individually...")
    qt = QueryTransformer()
    q = "tell me about coding class"
    expanded = qt.expand_query(q)
    rewritten = qt.rewrite_query(q)
    multi = qt.generate_multi_queries(q)
    print(f"[SUCCESS] Expanded: {expanded}")
    print(f"[SUCCESS] Rewritten: {rewritten}")
    print(f"[SUCCESS] Multi-Queries: {multi}")

