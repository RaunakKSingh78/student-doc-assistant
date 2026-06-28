import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add the project root directory to the python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# pyrefly: ignore [missing-import]
from src.ingestion import load_all_documents, save_chunks_to_json, load_chunks_from_json, EmbeddingPipeline
# pyrefly: ignore [missing-import]
from src.generator import RAGSearch

load_dotenv()

API_PREFIX = "/api"
PERSIST_DIR = os.path.join("chroma_store")
PROCESSED_DIR = os.path.join("data", "processed")
RAW_DIR = os.path.join("data", "raw")

app = FastAPI(title="Student Document Assistant Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_search: Optional[RAGSearch] = None


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5


class SourceItem(BaseModel):
    source: str
    text: str
    metadata: dict = {}


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[SourceItem] = []


def ensure_processed_chunks() -> None:
    if not os.path.exists(PROCESSED_DIR) or not os.listdir(PROCESSED_DIR):
        if not os.path.exists(RAW_DIR):
            raise FileNotFoundError(f"Raw data directory not found: {RAW_DIR}")

        documents = load_all_documents(RAW_DIR)
        if not documents:
            raise FileNotFoundError(f"No documents found in raw data directory: {RAW_DIR}")

        emb_pipeline = EmbeddingPipeline()
        chunks = emb_pipeline.chunk_documents(documents)
        save_chunks_to_json(chunks, PROCESSED_DIR)


def ensure_vector_store() -> None:
    chroma_path = os.path.join(PERSIST_DIR, "chroma.sqlite3")
    if not os.path.exists(chroma_path):
        ensure_processed_chunks()
        chunks = load_chunks_from_json(PROCESSED_DIR)
        if not chunks:
            raise RuntimeError("No chunks loaded from processed data.")

        os.makedirs(PERSIST_DIR, exist_ok=True)
        from src.retriever import ChromaVectorStore

        store = ChromaVectorStore(PERSIST_DIR)
        store.build_from_chunks(chunks)


@app.on_event("startup")
async def startup_event() -> None:
    global rag_search

    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY environment variable is not set.")

    ensure_vector_store()
    rag_search = RAGSearch(persist_dir=PERSIST_DIR)


@app.get(f"{API_PREFIX}/health")
async def health() -> dict:
    return {"status": "ok", "ready": rag_search is not None}


@app.get(f"{API_PREFIX}/status")
async def status() -> dict:
    processed_files = list(Path(PROCESSED_DIR).glob("*.json")) if os.path.exists(PROCESSED_DIR) else []
    return {
        "ready": rag_search is not None,
        "vector_store_exists": os.path.exists(os.path.join(PERSIST_DIR, "chroma.sqlite3")),
        "processed_chunk_files": len(processed_files),
    }


@app.post(f"{API_PREFIX}/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    if rag_search is None:
        raise HTTPException(status_code=503, detail="RAG engine is not initialized yet.")

    result = rag_search.search_and_summarize(request.query, top_k=request.top_k)
    return QueryResponse(
        query=request.query,
        answer=result["answer"],
        sources=result["passages"],
    )


@app.post(f"{API_PREFIX}/rebuild")
async def rebuild_store() -> dict:
    ensure_processed_chunks()
    from src.retriever import ChromaVectorStore

    chunks = load_chunks_from_json(PROCESSED_DIR)
    if not chunks:
        raise HTTPException(status_code=500, detail="No chunks available for rebuild.")

    os.makedirs(PERSIST_DIR, exist_ok=True)
    store = ChromaVectorStore(PERSIST_DIR)
    store.build_from_chunks(chunks)

    global rag_search
    rag_search = RAGSearch(persist_dir=PERSIST_DIR)

    return {"status": "rebuild_complete", "chunk_count": len(chunks)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.backend:app", host="0.0.0.0", port=8000, reload=True)
