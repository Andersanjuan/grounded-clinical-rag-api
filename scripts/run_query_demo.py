from app.data_ingest.embedding import LocalEmbeddingModel
from app.db.vector_store import ChromaVectorStore


def main():
    question = input("Enter your question: ").strip()
    if not question:
        print("No question entered.")
        return

    embedder = LocalEmbeddingModel()
    store = ChromaVectorStore()

    q_emb = embedder.embed_texts([question])[0]
    results = store.query(q_emb, top_k=3)

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    ids = results.get("ids", [[]])[0] if "ids" in results else [None] * len(docs)

    print("\n--- Retrieved Context (Top 3) ---\n")
    for rank, (text, meta, cid) in enumerate(zip(docs, metas, ids), start=1):
        filename = meta.get("filename", "unknown_file")
        chunk_id = meta.get("chunk_id", "NA")
        chunk_uid = meta.get("chunk_uid", cid)

        print(f"[{rank}] {filename} (chunk {chunk_id}) | id={chunk_uid}")
        print(text.strip())
        print()

    print("--- Suggested Citations ---")
    for rank, meta in enumerate(metas, start=1):
        print(f"[{rank}] {meta.get('chunk_uid', 'unknown_chunk')}")

    print()


if __name__ == "__main__":
    main()
