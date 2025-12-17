from typing import List, Dict, Any, Optional
import chromadb
from app.config import settings

class ChromaVectorStore:
    
    def __init__(
        self, 
        collection_name: str | None = None, 
        persist_directory: str | None = None
    ):
        collection_name = collection_name or settings.chroma_collection
        persist_directory = persist_directory or settings.chroma_dir
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(
        self,
        docs: List[Dict[str, Any]],
        embeddings: List[List[float]],
        ids: Optional[List[str]] = None,
    ):
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(docs))]

        metadatas = [doc.get("metadata", {}) for doc in docs]
        texts = [doc["content"] for doc in docs]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=texts,
        )

    def query(self, query_embedding: List[float], top_k: int = 5):
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
    
    