"""Transparent dense retrieval using sentence-transformers plus NumPy ranking."""

import numpy as np
from sentence_transformers import SentenceTransformer

from findocs.types import Chunk, RetrievedChunk


class DenseRetriever:
    """Encode chunks once, then rank them by cosine similarity."""

    def __init__(self, chunks: list[Chunk], model_name: str = "all-MiniLM-L6-v2") -> None:
        self.chunks = chunks
        self.model = SentenceTransformer(model_name)
        self.vectors = self._normalise(self.model.encode([c.text for c in chunks], show_progress_bar=True))

    @staticmethod
    def _normalise(vectors: np.ndarray) -> np.ndarray:
        """Unit-normalise rows so a dot product equals cosine similarity."""

        values = np.asarray(vectors, dtype="float32")
        return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)

    def search(self, query: str, k: int = 10) -> list[RetrievedChunk]:
        """Return the highest-scoring chunks, with rank starting at one."""

        query_vector = self._normalise(self.model.encode([query]))[0]
        scores = self.vectors @ query_vector
        indices = np.argsort(-scores)[:k]
        return [RetrievedChunk(self.chunks[i], float(scores[i]), rank + 1, "dense") for rank, i in enumerate(indices)]

