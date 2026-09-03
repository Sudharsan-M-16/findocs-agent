"""Optional cross-encoder stage: expensive precision after cheap recall."""

from sentence_transformers import CrossEncoder

from findocs.types import RetrievedChunk


class CrossEncoderReranker:
    """Rerank only candidates already found by hybrid retrieval."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[RetrievedChunk], k: int = 5) -> list[RetrievedChunk]:
        """Score query-document pairs jointly and return the best candidates."""

        pairs = [(query, candidate.chunk.text) for candidate in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda pair: float(pair[1]), reverse=True)[:k]
        return [RetrievedChunk(item.chunk, float(score), rank + 1, "reranker") for rank, (item, score) in enumerate(ranked)]

