from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    docs: List[Dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    
) -> List[Dict]:
    """
    Take a list of docs with 'content' and 'metadata',
    return a list of smaller chunk dicts.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    all_chunks: List[Dict] = []

    for doc in docs:
        content = doc["content"]
        metadata = doc.get("metadata", {})
        source = metadata.get("filename", metadata.get("source", "unknown_source"))

        chunks = text_splitter.split_text(content)
        for i, chunk in enumerate(chunks):
            chunk_meta = metadata.copy()
            chunk_meta["chunk_id"] = i
            chunk_meta["chunk_uid"] = f"{source}::chunk_{i}"
            
            all_chunks.append(
                {
                    "content": chunk,
                    "metadata": chunk_meta,
                }
            )

    return all_chunks
