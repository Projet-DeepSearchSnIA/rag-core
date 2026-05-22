from sentence_transformers import SentenceTransformer
from typing import List


class LocalEmbedder:
    """embedder local avec SentenceTransformer"""

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        embeddings = self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True)
        return embeddings.tolist()
