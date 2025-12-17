from dataclasses import dataclass
from typing import Optional

from app.data_ingest.embedding import LocalEmbeddingModel
from app.db.vector_store import ChromaVectorStore
from langchain_ollama import ChatOllama


@dataclass
class AppState:
    embedder: Optional[LocalEmbeddingModel] = None
    vector_store: Optional[ChromaVectorStore] = None
    llm: Optional[ChatOllama] = None


state = AppState()
