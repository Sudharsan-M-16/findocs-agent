"""
dense.py — Dense Semantic Retrieval
=====================================
CORE CONCEPT: Embedding + Cosine Similarity
--------------------------------------------
An embedding model (here: sentence-transformers' all-MiniLM-L6-v2) maps any
piece of text to a fixed-length vector (384 numbers for this model) such that
semantically similar text produces vectors that are "close together" in that
384-dimensional space.

"Close together" is measured by COSINE SIMILARITY:
    cos(A, B) = (A · B) / (|A| × |B|)

If both vectors are unit-normalised (length = 1), then the dot product A · B
equals the cosine similarity directly. That's why this code normalises first
and then does a matrix multiplication — it's a fast batch cosine similarity.

WHY all-MiniLM-L6-v2?
- Only 22M parameters → fast enough to run on CPU.
- Produces 384-dimension vectors → small memory footprint.
- Trained on sentence similarity tasks → good for semantic search.
- No API key, no cost, no network call after first download.

WHAT "SEMANTIC" MEANS:
"Revenue declined due to headwinds" and "sales dropped because of weak economy"
produce vectors that are very close. They share almost no words but the
MEANING is the same. BM25 (the keyword retriever) would NOT connect these.
That's why we need both dense and sparse retrieval (see hybrid.py).

WHERE DENSE FAILS:
"Apple Q3 2024 revenue was $85.8B" — the exact number $85.8B doesn't carry
semantic meaning by itself. If you query "what was Apple's Q3 revenue", dense
search will return chunks that discuss revenue generally, but might rank the
exact figure chunk lower than a generic revenue discussion chunk. BM25 handles
this better (exact term matching).
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from findocs.types import Chunk, RetrievedChunk


class DenseRetriever:
    """
    Encode chunks once, then rank them by cosine similarity at query time.

    TWO-PHASE DESIGN:
    __init__: build the index (slow, done once per session)
    search:   query the index (fast, done per question)

    WHY NOT REBUILD ON EVERY QUERY?
    Encoding 181 Apple chunks takes ~5 seconds on CPU. Building once and
    caching the result matrix means each query runs in <100ms.

    MEMORY: 181 chunks × 384 dimensions × 4 bytes = ~278 KB. Trivial.
    For 10,000 chunks this becomes ~15 MB. Still fine with NumPy.
    For 10 million chunks you'd need an ANN index (FAISS, HNSW) instead.
    """

    def __init__(self, chunks: list[Chunk], model_name: str = "all-MiniLM-L6-v2") -> None:
        """
        Load the model and pre-compute normalised embeddings for all chunks.

        ARGUMENTS:
        chunks     : The list of Chunk objects whose .text fields get embedded.
        model_name : HuggingFace model identifier. First call downloads ~90MB
                     to ~/.cache/torch/sentence_transformers/.

        WHAT _normalise() DOES:
        Each embedding vector from the model has an arbitrary L2 norm (length).
        Dividing by the norm makes every vector have length 1 (unit vector).
        After normalisation, dot_product(A, B) == cosine_similarity(A, B).
        This lets us use simple matrix multiplication for retrieval instead of
        the slower full cosine formula.
        """

        self.chunks = chunks
        # SentenceTransformer downloads and caches the model on first call.
        self.model = SentenceTransformer(model_name)
        # show_progress_bar=True prints a progress bar when encoding many chunks.
        self.vectors = self._normalise(
            self.model.encode([c.text for c in chunks], show_progress_bar=True)
        )

    @staticmethod
    def _normalise(vectors: np.ndarray) -> np.ndarray:
        """
        Unit-normalise rows so a dot product equals cosine similarity.

        STEP BY STEP:
        1. Cast to float32 (half the memory of float64, sufficient precision).
        2. Compute L2 norm of each row: sqrt(x₁² + x₂² + ... + x₃₈₄²).
           keepdims=True keeps shape (N, 1) so broadcasting works.
        3. np.maximum(..., 1e-12) prevents division-by-zero for zero vectors
           (a zero vector has no meaningful direction, but we avoid NaN).
        4. Divide each row by its norm → all rows now have length 1.
        """

        values = np.asarray(vectors, dtype="float32")
        # axis=1 means compute norm per row (one norm per chunk vector)
        return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)

    def search(self, query: str, k: int = 10) -> list[RetrievedChunk]:
        """
        Return the highest-scoring chunks, with rank starting at one.

        HOW THE SEARCH WORKS:
        1. Encode the query string into a 384-dim vector.
        2. Normalise it (same unit-length transform as the stored vectors).
        3. Dot product: self.vectors @ query_vector
           Because both are unit-normalised, this computes cosine similarity
           for ALL chunks simultaneously in one matrix multiply — very fast.
        4. argsort(-scores): negate scores so argsort gives descending order
           (highest similarity first). Take the top k indices.
        5. Build RetrievedChunk objects with the original Chunk, its score, its
           1-indexed rank, and the source label "dense".

        INTERVIEW QUESTION: "Why argsort(-scores) instead of argsort(scores)?"
        argsort returns indices sorted ascending (smallest first). Negating the
        scores makes the highest similarity become the most negative, so it
        appears first. Alternatively: np.argsort(scores)[::-1][:k].
        """

        # Encode and normalise the query vector
        query_vector = self._normalise(self.model.encode([query]))[0]  # shape: (384,)
        # Dot product with all chunk vectors simultaneously → shape: (N,)
        scores = self.vectors @ query_vector
        # Get indices of top-k scores in descending order
        indices = np.argsort(-scores)[:k]
        # Build results with 1-indexed ranks
        return [
            RetrievedChunk(self.chunks[i], float(scores[i]), rank + 1, "dense")
            for rank, i in enumerate(indices)
        ]
