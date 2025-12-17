from app.data_ingest.text_loader import load_text_files
from app.data_ingest.chunking import chunk_documents
from app.data_ingest.embedding import LocalEmbeddingModel
from app.db.vector_store import ChromaVectorStore


def main():
    # 1. Load raw text docs
    docs = load_text_files("data/sample_docs")
    print(f"Loaded {len(docs)} documents")

    # 2. Chunk them
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")

    # 3. Embed chunks
    embedder = LocalEmbeddingModel()
    texts = [c["content"] for c in chunks]
    embeddings = embedder.embed_texts(texts)
    print(f"Generated {len(embeddings)} embeddings")

    # 4. Store in Chroma
    store = ChromaVectorStore()
    ids = [c["metadata"]["chunk_uid"] for c in chunks]
    store.add_documents(chunks, embeddings, ids=ids)
    print("Stored chunks in ChromaDB")

    # 5. Quick retrieval test
    test_query = "What does this document say about diet or treatment?"
    q_emb = embedder.embed_texts([test_query])[0]
    results = store.query(q_emb, top_k=3)
    print("Top matches:", results.get("documents", [[]])[0])
    print("Metadatas:", results.get("metadatas", [[]])[0])


if __name__ == "__main__":
    main()
