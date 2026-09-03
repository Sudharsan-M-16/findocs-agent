"""Create transparent query-chunk labels for the Day 7-8 relevance grader."""

import csv
from pathlib import Path

from findocs.types import Chunk, RetrievedChunk


def candidate_rows(question: dict, results: list[RetrievedChunk]) -> list[dict[str, str | int]]:
    """Turn retrieved candidates into rows a human can label relevant or not."""

    rows: list[dict[str, str | int]] = []
    for result in results:
        rows.append({
            "question_id": question["id"],
            "question": question["question"],
            "chunk_id": result.chunk.chunk_id,
            "company": result.chunk.company,
            "section": result.chunk.section,
            "rank": result.rank,
            "label": "",
            "chunk_text": result.chunk.text[:1800],
        })
    return rows


def write_label_sheet(rows: list[dict[str, str | int]], path: str) -> None:
    """Write a CSV where the only manual column is label: 1 relevant, 0 irrelevant."""

    if not rows:
        raise ValueError("No rows to write.")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_labeled_pairs(path: str) -> list[dict[str, str | int]]:
    """Load only completed rows so unfinished labels do not enter training."""

    with Path(path).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if str(row.get("label", "")).strip() in {"0", "1"}]


def prompt_for_pair(question: str, chunk: Chunk) -> str:
    """Format the exact supervised task the small grader will learn."""

    return (
        "Decide whether the SEC filing chunk contains enough evidence to help answer the question.\n"
        f"Question: {question}\n"
        f"Chunk: {chunk.text}\n"
        "Answer with only relevant or irrelevant."
    )
