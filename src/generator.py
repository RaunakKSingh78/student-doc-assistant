import os
import sys
from pathlib import Path
from typing import List
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
from src.retriever import ChromaVectorStore
# pyrefly: ignore [missing-import]
from src.ingestion import load_chunks_from_json

load_dotenv()

if not os.environ.get("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY environment variable is missing. Check your .env file.")

class RAGSearch:

    def __init__(self, persist_dir: str = "chroma_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "llama-3.3-70b-versatile"):
        self.vectorstore = ChromaVectorStore(persist_dir, embedding_model)
        
        # Load the store (or build it from JSON chunks if sqlite file doesn't exist)
        if not os.path.exists(os.path.join(persist_dir, "chroma.sqlite3")):

            # Load chunks from data/processed
            chunks = load_chunks_from_json("data/processed")
            self.vectorstore.build_from_chunks(chunks)
        
        self.vectorstore.load()
        self.llm = ChatGroq(model_name=llm_model)
        print(f"[INFO] Groq LLM initialized: {llm_model}")

        # Define the prompt template for QA
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system",
             """You are an assistant for question-answering tasks. 
             You generally have contexts of Rules, Policies and Guidelines . 
             Use the following pieces of retrieved context to answer the question. 
             Answer only based on the context provided. 
             Do not provide any additional information.
             If the answer is not in the context, say "I don't know".
             Provide a concise, accurate and helpful answer.\n\nContext:\n{context}"""),
            ("human", "{question}"),
        ])

    def search_and_summarize(self, query: str, top_k: int = 5) -> dict:
        # 1. Retrieve relevant context from the Vector Store (converting query to embeddings under the hood)
        retrieved_docs = self.vectorstore.retrieve_context(query, top_k=top_k)

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
        
        # 2. Format the retrieved context
        context_str = "\n\n".join(doc.page_content for doc in retrieved_docs)
        
        # 3. Combine with prompt template
        prompt_value = self.prompt_template.format_messages(context=context_str, question=query)
        
        # 4. Feed combined prompt into LLM and return the generated answer plus retrieved passages
        response = self.llm.invoke(prompt_value)
        return {
            "answer": response.content,
            "passages": passages,
        }

if __name__ == "__main__":
    rag_search = RAGSearch()
    query = input("Enter your query: ")
    print(rag_search.search_and_summarize(query))
