"""
hybrid.py — Reciprocal Rank Fusion (RRF)
==========================================
CORE IDEA:
Dense retrieval and BM25 retrieval produce two separate ranked lists of chunks.
We need to merge them into one ranked list that benefits from both.

NAIVE APPROACH (WRONG): Average the scores.
WHY IT FAILS: Dense cosine similarity lives in [0, 1]. BM25 scores can be
anywhere from 0 to 20+. Averaging them gives BM25 a much larger influence
just because of its numerical scale — not because it's better. This would
make BM25 dominate on every query regardless of query type.

CORRECT APPROACH: Reciprocal Rank Fusion (RRF).
KEY INSIGHT: Instead of using raw scores, use rank positions. Rank position
is comparable across any retrieval method — "rank 1" means "best result"
whether it's dense, BM25, or reranking.

THE RRF FORMULA:
    RRF_score(chunk) = Σ 1 / (k + rank_in_list_i)

For each ranked list (dense, BM25), a chunk gets a contribution of 1/(k+rank).
These contributions are summed across lists. Chunks appearing high in BOTH
lists score highest — they're the consensus best results.

WHY k=60?
k is a smoothing constant. It prevents rank-1 from being astronomically
better than rank-2 (1/61 vs 1/62 are close). The original RRF paper (Cormack
et al., 2009) found k=60 works well empirically across many search tasks.

EXAMPLE:
Chunk "section-42" appears at rank 3 in dense and rank 1 in BM25:
  dense contribution:  1 / (60 + 3) = 0.01587
  BM25 contribution:   1 / (60 + 1) = 0.01639
  Total:               0.03226

Chunk "section-7" appears at rank 1 in dense but not in BM25 top-k:
  dense contribution:  1 / (60 + 1) = 0.01639
  bm25 contribution:   0 (not in BM25 results)
  Total:               0.01639

"section-42" wins because it ranked high in BOTH — that's a stronger signal.
"""

from findocs.types import Chunk, RetrievedChunk


def reciprocal_rank_fusion(
    *result_lists: list[RetrievedChunk],
    k: int = 60,
    limit: int = 10,
) -> list[RetrievedChunk]:
    """
    Fuse ranked lists without pretending their raw score scales are comparable.

    ARGUMENTS:
    *result_lists : Any number of ranked lists (dense results, BM25 results, etc.)
    k             : Smoothing constant. k=60 is the standard value from the paper.
    limit         : How many results to return after fusion.

    HOW THE CODE WORKS — line by line:

    fused dict:
        Maps chunk_id → (Chunk, accumulated_rrf_score, [source_list])
        We use chunk_id as the key because the same chunk can appear in
        multiple result lists and needs to be identified as the same document.

    For each list, for each result:
        contribution = 1.0 / (k + result.rank)
        If this chunk hasn't been seen yet: create an entry with this score.
        If it's already in fused: add the new contribution to the existing score.
        Also accumulate the source labels (["dense", "bm25"]) for traceability.

    Sort by score descending, take top `limit`.

    Build final RetrievedChunk objects:
        source = "+".join(sources) e.g. "dense+bm25" or just "dense"
        This tells you whether a chunk appeared in one or both lists.

    WHY USE dict.get() WITH None CHECK?
    Python's defaultdict would work too, but explicit None checking makes the
    "first time seen" vs "already seen" logic clearer to read.

    INTERVIEW QUESTION: "Could you use more than two lists?"
    YES — *result_lists accepts any number. Add a third list from a different
    embedding model and it just works. The formula naturally handles N lists.
    """

    # chunk_id → (Chunk object, cumulative RRF score, list of source labels)
    fused: dict[str, tuple[Chunk, float, list[str]]] = {}

    for results in result_lists:
        for result in results:
            # RRF score contribution from this chunk's rank in this list
            contribution = 1.0 / (k + result.rank)
            old = fused.get(result.chunk.chunk_id)

            if old is None:
                # First time we see this chunk: create its entry
                fused[result.chunk.chunk_id] = (result.chunk, contribution, [result.source])
            else:
                # Already seen: accumulate score and add new source label
                fused[result.chunk.chunk_id] = (
                    old[0],
                    old[1] + contribution,
                    old[2] + [result.source],
                )

    # Sort by accumulated RRF score (highest first), take top `limit`
    ordered = sorted(fused.values(), key=lambda item: item[1], reverse=True)[:limit]

    # Assign fresh 1-indexed ranks to the fused results
    return [
        RetrievedChunk(chunk, score, rank + 1, "+".join(sources))
        for rank, (chunk, score, sources) in enumerate(ordered)
    ]


class HybridRetriever:
    """
    Run dense and BM25 independently, then combine their rankings.

    WHY A CLASS AND NOT JUST A FUNCTION?
    Having a class with a .search() interface means the rest of the code
    (graph.py, cli.py, eval/pipeline.py) can treat hybrid retrieval exactly
    the same as pure dense or pure BM25 retrieval. One interface → easy to
    swap in any evaluation harness.

    USAGE:
        dense = DenseRetriever(chunks)
        bm25 = BM25Retriever(chunks)
        hybrid = HybridRetriever(dense, bm25)
        results = hybrid.search("What are Apple's risk factors?", k=10)
    """

    def __init__(self, dense, sparse) -> None:
        """
        Store references to pre-built retrievers.
        Both retrievers must already be initialised (indexes built).
        Dense and sparse are injected — this class doesn't build its own.
        """

        self.dense = dense
        self.sparse = sparse

    def search(self, query: str, k: int = 10) -> list[RetrievedChunk]:
        """
        Retrieve from both stages and fuse the two ranked lists.

        WHY k FOR BOTH AND THEN k FOR FUSION?
        Each retriever returns k results. Some of those results will overlap
        (same chunk appears in both). After fusion we keep the top k from the
        merged set. This means the final result has at most k chunks, but often
        fewer unique chunks than 2k because of overlaps.

        NOTE: Passing k=k to both retrievers means we only consider each method's
        top-k candidates for fusion. If a chunk is rank 11 in dense and rank 1 in
        BM25, it won't appear in the dense list at all. A larger initial k (e.g.,
        k=20 for retrieval, k=10 for final output) would catch more such cases —
        this is exactly what pipeline.py does when building the reranker stage.
        """

        return reciprocal_rank_fusion(
            self.dense.search(query, k),   # dense top-k results
            self.sparse.search(query, k),  # BM25 top-k results
            limit=k,                        # final output size
        )
