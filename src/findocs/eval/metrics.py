"""
metrics.py — Retrieval and Answer Evaluation Metrics
=====================================================
WHY HAND-WRITE METRICS INSTEAD OF USING A LIBRARY?
Libraries like RAGAS or trulens abstract the math away. For an interview,
you MUST be able to derive and explain each metric from first principles.
If you use a library you cannot explain what "MRR" means without notes, an
interviewer will notice immediately.

These five functions are ~50 lines total. Reading them once and understanding
what they compute is far more valuable than importing a library.

THE FOUR RETRIEVAL METRICS:
---------------------------
Recall@K:
    "Is at least one relevant chunk in the top K results?"
    Binary: 1.0 if yes, 0.0 if no.
    WHY: The most important retrieval question for RAG is "did we retrieve
    ANYTHING relevant?" Not "how many relevant did we get", because the
    generator only needs one good source. Recall@5 = 0.8 means 80% of
    questions had at least one relevant chunk in the top 5.

Precision@K:
    "What fraction of the top K results are relevant?"
    Range: [0, 1]. Precision@5 = 0.4 means 2 out of 5 returned chunks were relevant.
    WHY: Precision measures noise. A retriever that returns 5 chunks where 4
    are irrelevant has Precision@5 = 0.2 — the generator has to filter a lot.
    High Recall + Low Precision = retriever casts wide net but includes junk.
    High Precision + Low Recall = retriever is selective but misses things.

MRR (Mean Reciprocal Rank):
    "Where is the FIRST relevant chunk?"
    Score = 1/rank_of_first_relevant_chunk. Average this over all questions.
    MRR = 1.0: every question's first result is relevant.
    MRR = 0.5: the first relevant chunk is at rank 2 on average.
    MRR = 0.0: no question ever had a relevant result.
    WHY: If you only show the top-1 result to the user (or generator), you
    care about rank 1 a lot. MRR weights rank 1 most heavily.

THE TWO ANSWER METRICS:
-----------------------
Answer Accuracy:
    "Does the answer contain any of the accepted phrases?"
    Binary: 1.0 if yes. These phrases are hand-written by you in eval_questions.json.
    WHY: Simple but honest. "Revenue increased" being in the answer is a clear
    signal that the agent answered the question correctly, even if the exact
    wording differs.

Citation Correctness:
    "What fraction of answer claims are supported by retrieved evidence?"
    Range: [0, 1]. Computed from verify.py's output (which is stored in state["verification"]).
    WHY: Measures hallucination rate. If the agent claims "R&D was $29.9B" but
    that figure isn't in the retrieved chunks, citation_correctness drops.
"""

from collections.abc import Iterable

from findocs.types import RetrievedChunk


# ── Internal helper ──────────────────────────────────────────────────────────

def _relevant_ids(
    results: Iterable[RetrievedChunk], relevant_ids: set[str], k: int
) -> list[str]:
    """
    Return retrieved IDs in rank order, limited to the requested cutoff.

    This is a helper that computes the intersection between:
    - The top-k retrieved chunk IDs (in rank order)
    - The set of hand-labeled relevant chunk IDs

    Used by recall_at_k() but factored out so it can be reused.
    """

    return [
        item.chunk.chunk_id
        for item in list(results)[:k]
        if item.chunk.chunk_id in relevant_ids
    ]


# ── Retrieval metrics ────────────────────────────────────────────────────────

def recall_at_k(
    results: Iterable[RetrievedChunk], relevant_ids: set[str], k: int
) -> float:
    """
    Measure whether at least one relevant chunk was retrieved by rank k.

    Returns 1.0 (success) or 0.0 (failure).

    WHY BINARY AND NOT COUNT-BASED?
    For RAG, you need at least ONE relevant chunk in the top k so the generator
    has something to work with. Whether there are 2 or 5 relevant chunks is less
    important than whether there are any at all. Binary recall@k captures this.

    EXAMPLE:
    Relevant chunk IDs: {"section-42"}
    Top-5 results:      ["section-7", "section-42", "section-1", "section-3", "section-9"]
    recall_at_5 = 1.0 (section-42 is at rank 2, within the top 5)
    recall_at_3 = 0.0 would be WRONG here since rank 2 <= 3. Actually 1.0.
    recall_at_1 = 0.0 (section-42 is at rank 2, not in top 1)
    """

    return float(bool(_relevant_ids(results, relevant_ids, k)))


def precision_at_k(
    results: Iterable[RetrievedChunk], relevant_ids: set[str], k: int
) -> float:
    """
    Measure the fraction of the top-k results that are relevant.

    FORMULA: |{retrieved_top_k} ∩ {relevant}| / k

    WHY max(len(top), 1)?
    If no results were returned at all (empty list), we'd divide by zero.
    The max guard returns 0.0 precision for empty results, which is correct.
    """

    top = list(results)[:k]
    return sum(item.chunk.chunk_id in relevant_ids for item in top) / max(len(top), 1)


def reciprocal_rank(
    results: Iterable[RetrievedChunk], relevant_ids: set[str]
) -> float:
    """
    Return 1/rank for the first relevant result, or zero when absent.

    FORMULA: 1 / rank_of_first_relevant (rank is 1-indexed)

    WHY 1/RANK AND NOT JUST RANK?
    1/rank is bounded in [0, 1], which makes averaging meaningful.
    rank 1 → score 1.0 (perfect)
    rank 2 → score 0.5
    rank 5 → score 0.2
    rank 10 → score 0.1
    Not found → score 0.0

    MEAN RECIPROCAL RANK (MRR):
    Average reciprocal_rank across all questions.
    The average_metric_rows() function below handles this averaging.
    """

    for rank, item in enumerate(results, start=1):  # 1-indexed rank
        if item.chunk.chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0  # Relevant chunk not found in any retrieved result


def summarise_retrieval(
    results: Iterable[RetrievedChunk], relevant_ids: set[str]
) -> dict[str, float]:
    """
    Produce the core table row for one question and one retrieval stage.

    Computes all four metrics in one call so the ablation runner can store
    a single dict per (question, stage) pair.

    USAGE IN ablation.py:
        row = summarise_retrieval(stage(question), relevant_ids_for_question)
        rows.append({"question_id": q_id, "stage": stage_name, **row})
    """

    ordered = list(results)  # Materialise the iterator once, reuse for all metrics
    return {
        "recall_at_5": recall_at_k(ordered, relevant_ids, 5),
        "recall_at_10": recall_at_k(ordered, relevant_ids, 10),
        "mrr": reciprocal_rank(ordered, relevant_ids),
        "precision_at_5": precision_at_k(ordered, relevant_ids, 5),
    }


# ── Answer quality metrics ───────────────────────────────────────────────────

def answer_accuracy(answer: str, accepted_phrases: list[str]) -> float:
    """
    A transparent baseline: one if any hand-written accepted phrase appears.

    WHY SO SIMPLE?
    "Exact match" evaluation is the most honest automated metric when you
    don't have an LLM judge. The accepted_phrases in eval_questions.json are
    SHORT phrases (e.g. "risk factors", "net sales") that MUST appear in a
    correct answer. If the answer says "The company's primary risk factors
    include...", the phrase "risk factors" is present → score 1.0.

    NORMALISATION:
    " ".join(answer.lower().split()) collapses all whitespace variants into
    single spaces. This handles filing text that uses non-breaking spaces or
    multiple consecutive spaces.

    LIMITATION:
    A correct answer that uses perfect synonyms ("risk disclosures" instead of
    "risk factors") would score 0.0. This is a known limitation and is why
    accepted_phrases should include multiple acceptable formulations.
    """

    normalised = " ".join(answer.lower().split())
    return float(
        any(
            " ".join(phrase.lower().split()) in normalised
            for phrase in accepted_phrases
        )
    )


def citation_correctness(
    verification: dict, expected_claim_count: int | None = None
) -> float:
    """
    Measure the fraction of answer claims marked supported by evidence.

    INPUTS:
    verification : The dict returned by verify_claims() — stored in graph state
                   as state["verification"].
    expected_claim_count : Optional sanity check. If the answer should have
                           exactly N claims and verify_claims() found a different
                           number (e.g. it split wrong), score 0.0.

    FORMULA: supported_claims / total_claims

    WHY NOT BINARY?
    Unlike recall@k (where partial retrieval is a failure), partial citation
    support is meaningful. If 3 out of 4 claims are supported, that's much
    better than 0/4 even if it's not perfect.

    INTERPRETATION:
    citation_correctness = 1.0 → all claims in the answer are grounded in evidence.
    citation_correctness = 0.5 → half the claims have no evidence support → potential hallucination.
    citation_correctness = 0.0 → no evidence for any claim (likely hallucination or bad retrieval).
    """

    claims = verification.get("claims", [])
    if expected_claim_count is not None and len(claims) != expected_claim_count:
        return 0.0
    return sum(bool(claim.get("supported")) for claim in claims) / max(len(claims), 1)


# ── Aggregation helper ────────────────────────────────────────────────────────

def average_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    """
    Average per-question metrics so one easy question cannot hide failures.

    WHY AVERAGE AND NOT TOTAL?
    If one question is easy (recall@5 = 1.0) and nine are hard (recall@5 = 0.0),
    the total recall@5 is 1. The average is 0.1 — correctly indicating the
    retriever fails on most questions.

    USAGE:
        rows = [{"recall_at_5": 1.0, "mrr": 1.0}, {"recall_at_5": 0.0, "mrr": 0.0}]
        average_metric_rows(rows) → {"recall_at_5": 0.5, "mrr": 0.5}
    """

    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}
