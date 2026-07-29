import os
import sys
import time
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Add parent directory to sys.path to support running this file directly
if __name__ == "__main__" or __package__ is None:
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

# pyrefly: ignore [missing-import]
from src.retriever import ChromaVectorStore, HybridRetriever, evaluate_query_retrieval
# pyrefly: ignore [missing-import]
from src.ingestion import load_chunks_from_json

load_dotenv()

if not os.environ.get("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY environment variable is missing. Check your .env file.")

class RAGSearch:

    def __init__(self, persist_dir: str = "chroma_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "llama-3.3-70b-versatile"):
        self.vectorstore = ChromaVectorStore(persist_dir, embedding_model)
        
        processed_dir = os.path.join("data", "processed")
        chunks = load_chunks_from_json(processed_dir) if os.path.exists(processed_dir) else []

        # Load the store (or build it from JSON chunks if sqlite file doesn't exist)
        if not os.path.exists(os.path.join(persist_dir, "chroma.sqlite3")):
            if not chunks:
                raise ValueError("No chunks found to build vector store.")
            self.vectorstore.build_from_chunks(chunks)
        
        self.vectorstore.load()
        
        # Instantiate HybridRetriever if chunks are available
        self.hybrid_retriever: Optional[HybridRetriever] = None
        if chunks:
            try:
                self.hybrid_retriever = HybridRetriever(self.vectorstore, chunks)
                print("[INFO] HybridRetriever initialized successfully.")
            except Exception as e:
                print(f"[WARNING] HybridRetriever initialization skipped: {e}")

        self.llm = ChatGroq(model_name=llm_model)
        print(f"[INFO] Groq LLM initialized: {llm_model}")

        # Define the prompt template for QA, incorporating the per-query Evaluation Report
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system",
             """You are an assistant for question-answering tasks. 
             You have contexts of Rules, Policies and Guidelines along with a Retrieval Evaluation Report. 
             Use the following pieces of retrieved context and retrieval evaluation report to answer the question. 
             Answer only based on the context provided.
             If the retrieved context has bullet points and new lines, preserve them in the output.
             Try to answer in concise points or as short sentences as possible.
             Do not provide any additional information on your own.
             If the answer is not in the context, simply say "We don't have any information about this question.".
             Provide a concise, accurate and helpful answer.

Evaluation Report:
{evaluation_report}

Context:
{context}"""),
            ("human", "{question}"),
        ])

    def search_and_summarize(self, query: str, top_k: int = 5) -> dict:
        start_time = time.perf_counter()

        # 1. Retrieve relevant context from retriever (Hybrid if available, else Semantic)
        if self.hybrid_retriever is not None:
            retrieved_docs = self.hybrid_retriever.retrieve(query, top_k=top_k, alpha=0.5)
            mode = "Hybrid (Semantic + BM25)"
        else:
            retrieved_docs = self.vectorstore.retrieve_context(query, top_k=top_k)
            mode = "Semantic (ChromaDB)"

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # 2. Generate per-query Evaluation Report
        eval_report = evaluate_query_retrieval(
            query=query,
            retrieved_docs=retrieved_docs,
            latency_ms=elapsed_ms,
            mode=mode,
            alpha=0.5,
            top_k=top_k
        )

        passages = []
        for doc in retrieved_docs:
            source = doc.metadata.get("source") or doc.metadata.get("filename") or "unknown_source"
            if source and isinstance(source, str):
                source = Path(source).name
            passages.append({
                "source": source,
                "text": doc.page_content,
                "metadata": dict(doc.metadata) if hasattr(doc, "metadata") else {},
            })
        
        # 3. Format the retrieved context
        context_str = "\n\n".join(doc.page_content for doc in retrieved_docs)
        
        # 4. Combine with prompt template including the evaluation report summary
        prompt_value = self.prompt_template.format_messages(
            evaluation_report=eval_report["summary_text"],
            context=context_str,
            question=query
        )
        
        # 5. Feed combined prompt into LLM and return the generated answer, passages, and evaluation report
        response = self.llm.invoke(prompt_value)
        return {
            "answer": response.content,
            "passages": passages,
            "evaluation_report": eval_report,
        }

if __name__ == "__main__":
    rag_search = RAGSearch()
    query = input("Enter your query: ")
    result = rag_search.search_and_summarize(query)
    print("\n" + "=" * 60)
    print("ANSWER:")
    print(result["answer"])
    print("=" * 60)
    print("\nEVALUATION REPORT:")
    print(result["evaluation_report"]["summary_text"])

