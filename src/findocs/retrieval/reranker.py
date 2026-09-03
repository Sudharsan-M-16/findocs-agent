"""
reranker.py — Cross-Encoder Reranking
=======================================
CORE CONCEPT: Bi-Encoder vs Cross-Encoder
------------------------------------------
EVERYTHING SO FAR IS A BI-ENCODER:
- The query is encoded separately → query vector.
- Each chunk is encoded separately → chunk vector.
- Similarity = dot product of the two vectors.

WHY BI-ENCODERS ARE FAST BUT IMPRECISE:
The two encodings never "see" each other. The model can't know that the word
"net" in the query "net sales" specifically matches "net" in a chunk that says
"net revenue" — because they were encoded in isolation.

A CROSS-ENCODER PROCESSES QUERY + CHUNK TOGETHER:
- Input: "[CLS] {query} [SEP] {chunk} [SEP]"
- The transformer's self-attention lets every query token attend to every chunk
  token simultaneously.
- Output: a single relevance score (not a vector).

WHY NOT USE CROSS-ENCODERS FOR EVERYTHING?
Speed. Cross-encoding 181 chunks takes ~10 seconds because each pair requires
a full transformer forward pass. That's fine for 20 candidates but not for
an entire corpus. The two-stage architecture:
  1. BI-ENCODER: retrieve top 20 cheaply (fast, approximate)
  2. CROSS-ENCODER: rerank those 20 precisely (slow, accurate)
...gives you cross-encoder accuracy at bi-encoder cost for the corpus.

MODEL CHOICE: cross-encoder/ms-marco-MiniLM-L-6-v2
- Trained on the MS MARCO passage retrieval dataset (real web search queries).
- MiniLM-L-6 is a distilled 6-layer model — good accuracy/speed tradeoff.
- This is NOT trained on financial text, so reranking helps most on generic
  semantic queries and less on finance-specific jargon.

INTERVIEW QUESTION: "When does reranking hurt more than it helps?"
When the bi-encoder retrieval already found the right chunks and the cross-
encoder re-orders them based on surface-level patterns the MARCO training
doesn't capture well. For finance-specific jargon, the reranker may actually
prefer a "sounds like an answer" chunk over a "contains the exact number" chunk.
This is why we measure before/after reranking in the ablation.
"""

from sentence_transformers import CrossEncoder

from findocs.types import RetrievedChunk


class CrossEncoderReranker:
    """
    Rerank only candidates already found by hybrid retrieval.

    WHY INJECT CANDIDATES FROM OUTSIDE?
    This class doesn't call any retriever itself. It receives the top-20
    candidates from HybridRetriever and reorders them. This keeps the two
    stages independently testable and swappable.

    USAGE PATTERN (in graph.py and pipeline.py):
        hybrid_results = hybrid.search(query, k=20)  # retrieve 20
        reranked = reranker.rerank(query, hybrid_results, k=5)  # keep best 5

    WHY k=20 → k=5?
    You want the reranker to see enough candidates to find the truly best ones
    (20 gives it a reasonable pool), but the final answer only needs 3-5 chunks
    of evidence to avoid overwhelming the generator or citation verifier.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        """
        Load the cross-encoder model.

        CrossEncoder() downloads the model from HuggingFace on first call and
        caches it. Unlike bi-encoders, there's no pre-computation step because
        cross-encoders are inherently query-dependent.
        """

        # CrossEncoder from sentence-transformers handles the "[CLS] Q [SEP] D [SEP]"
        # formatting internally. We just pass (query, chunk) pairs.
        self.model = CrossEncoder(model_name)

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], k: int = 5
    ) -> list[RetrievedChunk]:
        """
        Score query-document pairs jointly and return the best candidates.

        HOW IT WORKS:
        1. Build pairs: [(query, chunk1.text), (query, chunk2.text), ...]
           The cross-encoder needs both in a single input.
        2. model.predict(pairs): runs one forward pass per pair (or batched).
           Returns a list of relevance scores (floats, arbitrary scale).
        3. Sort by score descending, take top k.
        4. Build RetrievedChunk objects with source="reranker" and fresh ranks.

        WHY zip(candidates, scores)?
        model.predict() returns scores in the same order as the input pairs,
        so zipping pairs the original candidate object with its new score.

        WHY source="reranker" (not "dense" or "bm25")?
        The reranker changes the ranks from what hybrid search produced. The
        source label reflects the LAST stage that determined this chunk's
        position — which is the reranker. This matters for ablation analysis:
        you can see which chunks moved up or down after reranking.
        """

        # Create (query, chunk_text) pairs for cross-encoder scoring
        pairs = [(query, candidate.chunk.text) for candidate in candidates]

        # Get relevance scores — one float per pair
        scores = self.model.predict(pairs)

        # Sort (candidate, score) pairs by score descending, take top k
        ranked = sorted(
            zip(candidates, scores),
            key=lambda pair: float(pair[1]),
            reverse=True,
        )[:k]

        # Re-assign ranks 1..k with updated scores from cross-encoder
        return [
            RetrievedChunk(item.chunk, float(score), rank + 1, "reranker")
            for rank, (item, score) in enumerate(ranked)
        ]
