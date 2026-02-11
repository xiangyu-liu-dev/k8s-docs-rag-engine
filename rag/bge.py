from sentence_transformers import SentenceTransformer
import numpy as np


class BGEEmbedder:
    def __init__(self, model_name="BAAI/bge-large-en"):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=16,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return np.asarray(vecs, dtype="float32")

    def encode_query(self, query: str) -> np.ndarray:
        q = f"Represent this question for retrieving relevant passages: {query}"
        v = self.model.encode(
            q,
            normalize_embeddings=True,
        )
        return np.asarray(v, dtype="float32")
