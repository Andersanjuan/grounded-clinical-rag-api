from typing import List
from sentence_transformers import SentenceTransformer


class LocalEmbeddingModel:
    """
    Simple wrapper around a SentenceTransformers model.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]):
        """
        Return a list of embedding vectors for the given texts.
        """
        return self.model.encode(texts, convert_to_numpy=True).tolist()
