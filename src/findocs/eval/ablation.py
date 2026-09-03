"""Reusable experiment runner for the resume-facing retrieval comparison."""

import csv
from pathlib import Path
from typing import Callable

from findocs.eval.metrics import average_metric_rows, summarise_retrieval
from findocs.types import RetrievedChunk


Stage = Callable[[str], list[RetrievedChunk]]


def evaluate_stages(questions: list[dict], stages: dict[str, Stage]) -> list[dict[str, float | str]]:
    """Run every question through every stage using identical labels."""

    rows: list[dict[str, float | str]] = []
    for question in questions:
        relevant = set(question["relevant_chunk_ids"])
        if not relevant:
            raise ValueError(f"Question {question['id']} has no hand-labelled relevant chunk IDs.")
        for stage_name, stage in stages.items():
            metrics = summarise_retrieval(stage(question["question"]), relevant)
            rows.append({"question_id": question["id"], "stage": stage_name, **metrics})
    return rows


def aggregate_by_stage(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    """Create one table row per stage from the per-question result rows."""

    output = []
    for stage in dict.fromkeys(str(row["stage"]) for row in rows):
        stage_rows = [{key: float(value) for key, value in row.items() if key not in {"stage", "question_id"}} for row in rows if row["stage"] == stage]
        output.append({"stage": stage, **average_metric_rows(stage_rows)})
    return output


def write_csv(rows: list[dict], path: str) -> None:
    """Persist results so claims in the README can be regenerated."""

    if not rows:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

