from typing import List
from app.data_ingest.embedding import LocalEmbeddingModel
from app.db.vector_store import ChromaVectorStore
from app.models.schemas import RetrievedChunk
from app.state import state


def retrieve_context(question: str, top_k: int = 3) -> tuple[List[RetrievedChunk], List[str]]:
    if state.embedder is None or state.vector_store is None:
        raise RuntimeError("App state not initialized. Startup hook did not run.")

    q_emb = state.embedder.embed_texts([question])[0]
    results = state.vector_store.query(q_emb, top_k=top_k)

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]

    chunks: List[RetrievedChunk] = []
    citations: List[str] = []

    for rank, (text, meta, cid, dist) in enumerate(zip(docs, metas, ids, distances), start=1):
        source_file = meta.get("filename", "unknown_file")
        chunk_id = meta.get("chunk_uid", cid)

        chunks.append(
            RetrievedChunk(
                rank=rank,
                chunk_id=chunk_id,
                source_file=source_file,
                metadata=meta,
                text=text,
                distance=dist,
            )
        )
        citations.append(chunk_id)

    return chunks, citations
