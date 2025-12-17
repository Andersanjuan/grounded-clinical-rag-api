import json
from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.routers.qa import router as qa_router
from app.state import state
from app.config import settings

from app.data_ingest.embedding import LocalEmbeddingModel
from app.db.vector_store import ChromaVectorStore
from langchain_ollama import ChatOllama

import logging
from app.middleware import PrivacyAwareLoggingMiddleware

logging.basicConfig(level=logging.INFO)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Startup: initializing embedder...")
    state.embedder = LocalEmbeddingModel()
    print("Startup: embedder ready")

    print("Startup: initializing vector store...")
    state.vector_store = ChromaVectorStore(
        collection_name=settings.chroma_collection,
        persist_directory=settings.chroma_dir,
    )
    print("Startup: vector store ready")

    print("Startup: initializing LLM client...")
    state.llm = ChatOllama(
        model=settings.ollama_model,
        temperature=settings.llm_temperature,
        base_url=settings.ollama_base_url,
    )
    print("Startup: LLM client ready")

    print("Startup complete")
    yield
    print("Shutdown complete")

app = FastAPI(title="MedRAG Clinical Assistant API", lifespan=lifespan)

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "MedRAG API is running"}

app.include_router(qa_router)

app.add_middleware(PrivacyAwareLoggingMiddleware)

