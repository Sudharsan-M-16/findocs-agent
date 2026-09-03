"""Small, auditable retrieval metrics; no black-box evaluation framework yet."""

from collections.abc import Iterable

from findocs.types import RetrievedChunk


def _relevant_ids(results: Iterable[RetrievedChunk], relevant_ids: set[str], k: int) -> list[str]:
    """Return retrieved IDs in rank order, limited to the requested cutoff."""

    return [item.chunk.chunk_id for item in list(results)[:k] if item.chunk.chunk_id in relevant_ids]


def recall_at_k(results: Iterable[RetrievedChunk], relevant_ids: set[str], k: int) -> float:
    """Measure whether at least one relevant chunk was retrieved by rank k."""

    return float(bool(_relevant_ids(results, relevant_ids, k)))


def precision_at_k(results: Iterable[RetrievedChunk], relevant_ids: set[str], k: int) -> float:
    """Measure the fraction of the top-k results that are relevant."""

    top = list(results)[:k]
    return sum(item.chunk.chunk_id in relevant_ids for item in top) / max(len(top), 1)


def reciprocal_rank(results: Iterable[RetrievedChunk], relevant_ids: set[str]) -> float:
    """Return 1/rank for the first relevant result, or zero when absent."""

    for rank, item in enumerate(results, start=1):
        if item.chunk.chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def summarise_retrieval(results: Iterable[RetrievedChunk], relevant_ids: set[str]) -> dict[str, float]:
    """Produce the core table row for one question and one retrieval stage."""

    ordered = list(results)
    return {
        "recall_at_5": recall_at_k(ordered, relevant_ids, 5),
        "recall_at_10": recall_at_k(ordered, relevant_ids, 10),
        "mrr": reciprocal_rank(ordered, relevant_ids),
        "precision_at_5": precision_at_k(ordered, relevant_ids, 5),
    }


def answer_accuracy(answer: str, accepted_phrases: list[str]) -> float:
    """A transparent baseline: one if any hand-written accepted phrase appears."""

    normalised = " ".join(answer.lower().split())
    return float(any(" ".join(phrase.lower().split()) in normalised for phrase in accepted_phrases))


def citation_correctness(verification: dict, expected_claim_count: int | None = None) -> float:
    """Measure the fraction of answer claims marked supported by evidence."""

    claims = verification.get("claims", [])
    if expected_claim_count is not None and len(claims) != expected_claim_count:
        return 0.0
    return sum(bool(claim.get("supported")) for claim in claims) / max(len(claims), 1)


def average_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    """Average per-question metrics so one easy question cannot hide failures."""

    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}
