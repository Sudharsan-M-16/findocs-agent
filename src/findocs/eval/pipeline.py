"""Build complete retrieval stages for Day 10 ablation experiments."""

from findocs.retrieval.bm25 import BM25Retriever
from findocs.retrieval.dense import DenseRetriever
from findocs.retrieval.hybrid import HybridRetriever
from findocs.retrieval.reranker import CrossEncoderReranker
from findocs.types import Chunk, RetrievedChunk


def build_retrieval_stages(chunks: list[Chunk], use_reranker: bool = False) -> dict[str, object]:
    """Create every retrieval stage from one shared corpus for fair comparison."""

    dense = DenseRetriever(chunks)
    bm25 = BM25Retriever(chunks)
    hybrid = HybridRetriever(dense, bm25)
    stages: dict[str, object] = {
        "dense_only": lambda query: dense.search(query, k=10),
        "bm25_only": lambda query: bm25.search(query, k=10),
        "dense_bm25_rrf": lambda query: hybrid.search(query, k=10),
    }
    if use_reranker:
        reranker = CrossEncoderReranker()

        def hybrid_reranker(query: str) -> list[RetrievedChunk]:
            return reranker.rerank(query, hybrid.search(query, k=20), k=10)

        stages["hybrid_reranker"] = hybrid_reranker
    return stages
