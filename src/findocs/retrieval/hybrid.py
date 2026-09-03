"""Reciprocal Rank Fusion: the project's first genuinely custom retrieval logic."""

from findocs.types import Chunk, RetrievedChunk


def reciprocal_rank_fusion(*result_lists: list[RetrievedChunk], k: int = 60, limit: int = 10) -> list[RetrievedChunk]:
    """Fuse ranked lists without pretending their raw score scales are comparable."""

    fused: dict[str, tuple[Chunk, float, list[str]]] = {}
    for results in result_lists:
        for result in results:
            old = fused.get(result.chunk.chunk_id)
            contribution = 1.0 / (k + result.rank)
            if old is None:
                fused[result.chunk.chunk_id] = (result.chunk, contribution, [result.source])
            else:
                fused[result.chunk.chunk_id] = (old[0], old[1] + contribution, old[2] + [result.source])
    ordered = sorted(fused.values(), key=lambda item: item[1], reverse=True)[:limit]
    return [RetrievedChunk(chunk, score, rank + 1, "+".join(sources)) for rank, (chunk, score, sources) in enumerate(ordered)]


class HybridRetriever:
    """Run dense and BM25 independently, then combine their rankings."""

    def __init__(self, dense, sparse) -> None:
        self.dense = dense
        self.sparse = sparse

    def search(self, query: str, k: int = 10) -> list[RetrievedChunk]:
        """Retrieve twice and fuse the two ranked lists."""

        return reciprocal_rank_fusion(self.dense.search(query, k), self.sparse.search(query, k), limit=k)

