"""
pipeline.py — Retrieval Stage Factory for Ablation
=====================================================
PURPOSE: Build all four retrieval stages from ONE shared chunk list.

WHY A SHARED CHUNK LIST MATTERS:
If you built separate chunks for dense vs BM25 retrieval (e.g. different
chunk sizes or sources), the comparison would be confounded. You couldn't
tell whether BM25 outperformed dense because of the retrieval algorithm OR
because it got better chunks.

By building ALL stages from identical chunks:
    - dense_only uses the exact same 181 chunks as bm25_only
    - hybrid uses both of the above
    - hybrid+reranker adds cross-encoding on top of hybrid

This isolation is the statistical validity of the ablation study.

FUNCTION: build_retrieval_stages()
Returns a dict of {stage_name: callable}. Each callable takes a query string
and returns list[RetrievedChunk]. This interface matches the Stage type in
ablation.py exactly.

LAMBDA FUNCTIONS:
The lambdas capture the retriever objects via closure. When ablation.py calls
stages["dense_only"](query), it's calling:
    lambda query: dense.search(query, k=10)
which calls the already-built DenseRetriever's search method.

WHY LAMBDAS AND NOT METHOD REFERENCES?
Method references like dense.search would work, but they don't let us fix k=10
or add the reranker wrapping. Lambdas let us partial-apply k and add the
intermediate reranker step cleanly.
"""

from findocs.retrieval.bm25 import BM25Retriever
from findocs.retrieval.dense import DenseRetriever
from findocs.retrieval.hybrid import HybridRetriever
from findocs.retrieval.reranker import CrossEncoderReranker
from findocs.types import Chunk, RetrievedChunk


def build_retrieval_stages(
    chunks: list[Chunk], use_reranker: bool = False
) -> dict[str, object]:
    """
    Create every retrieval stage from one shared corpus for fair comparison.

    STAGE DEFINITIONS:
    ──────────────────
    "dense_only":
        Only use sentence-transformer cosine similarity.
        Baseline: what does embedding-only retrieval look like?

    "bm25_only":
        Only use BM25 keyword matching.
        Baseline: what does keyword-only retrieval look like?

    "dense_bm25_rrf":
        Run both, fuse with Reciprocal Rank Fusion.
        Key question: Does hybrid beat either alone?

    "hybrid_reranker" (optional):
        Run hybrid (dense+BM25+RRF) to get top-20 candidates,
        then apply cross-encoder to reorder to top-10.
        Key question: Does cross-encoder reranking improve over RRF?

    WHY RERANKER IS OPTIONAL:
    Loading the CrossEncoder model takes ~5s and requires downloading the model.
    During development/debugging, you want fast iterations without the reranker.
    The use_reranker=True flag is only passed for final evaluation runs.

    THE RERANKER LAMBDA EXPLAINED:
        lambda query: reranker.rerank(query, hybrid.search(query, k=20), k=10)
    1. hybrid.search(query, k=20): get top-20 hybrid candidates (wide pool)
    2. reranker.rerank(query, ..., k=10): cross-encode all 20 pairs,
       return the top-10 by cross-encoder score.

    WHY k=20 FOR HYBRID INPUT AND k=10 FOR RERANKER OUTPUT?
    The reranker can only improve over what it's given. Giving it 20 candidates
    vs 10 means it has more to choose from — it might promote the "true" best
    chunk from rank 15 to rank 1 after seeing the query alongside it.
    Restricting final output to 10 keeps the evidence manageable.
    """

    # Build all base retrievers from the same chunk list
    dense = DenseRetriever(chunks)          # Embeds all chunks once
    bm25 = BM25Retriever(chunks)            # Builds BM25 inverted index
    hybrid = HybridRetriever(dense, bm25)   # Stores references to both

    stages: dict[str, object] = {
        # Dense-only: pure cosine similarity, no keyword matching
        "dense_only": lambda query: dense.search(query, k=10),
        # BM25-only: pure keyword matching, no semantic understanding
        "bm25_only": lambda query: bm25.search(query, k=10),
        # Hybrid: RRF fusion of dense + BM25, no reranking
        "dense_bm25_rrf": lambda query: hybrid.search(query, k=10),
    }

    if use_reranker:
        # Load the cross-encoder model (slow first time, cached after)
        reranker = CrossEncoderReranker()

        # Hybrid + reranker: wide retrieve then precise rerank
        def hybrid_reranker(query: str) -> list[RetrievedChunk]:
            # Step 1: get top-20 candidates from hybrid search
            candidates = hybrid.search(query, k=20)
            # Step 2: cross-encode all 20 (query, chunk) pairs and return top-10
            return reranker.rerank(query, candidates, k=10)

        stages["hybrid_reranker"] = hybrid_reranker

    return stages
