"""
ablation.py — Retrieval Stage Ablation Harness
===============================================
WHAT IS AN ABLATION STUDY?
An ablation study measures the contribution of each component by removing it
(or replacing it with a baseline) and measuring the performance drop.

In this project:
    Stage 1: dense_only          ← "What if we had no BM25?"
    Stage 2: bm25_only           ← "What if we had no dense retrieval?"
    Stage 3: dense_bm25_rrf      ← "What does hybrid fusion buy us over either alone?"
    Stage 4: hybrid_reranker     ← "What does reranking buy us over hybrid fusion?"

Each stage runs on the SAME questions with the SAME labeled ground truth.
The difference in metrics between stages = the contribution of the added component.

RESULT TABLE (once you have labels):
    stage              recall@5  recall@10  mrr   precision@5
    dense_only         0.XX      0.XX       0.XX  0.XX
    bm25_only          0.XX      0.XX       0.XX  0.XX
    dense_bm25_rrf     0.XX      0.XX       0.XX  0.XX
    hybrid_reranker    0.XX      0.XX       0.XX  0.XX

That table is your resume bullet made concrete.

CRITICAL: This only runs if eval_questions.json has non-empty relevant_chunk_ids.
See the data labeling workflow in the README for how to fill those in.
"""

import csv
from pathlib import Path
from typing import Callable

from findocs.eval.metrics import average_metric_rows, summarise_retrieval
from findocs.types import RetrievedChunk


# Type alias: a "stage" is any callable that takes a query string and returns
# a ranked list of RetrievedChunk objects. This interface is satisfied by
# dense.search, bm25.search, hybrid.search, and the reranker wrapper in pipeline.py.
Stage = Callable[[str], list[RetrievedChunk]]


def evaluate_stages(
    questions: list[dict], stages: dict[str, Stage]
) -> list[dict[str, float | str]]:
    """
    Run every question through every stage using identical labels.

    ARGUMENTS:
    questions : The list loaded from eval_questions.json. Each entry must have
                a non-empty "relevant_chunk_ids" list — this is the ground truth.
    stages    : Dict mapping stage name → callable. e.g.:
                {"dense_only": dense.search, "bm25_only": bm25.search, ...}

    WHAT IT PRODUCES:
    One row per (question, stage) pair, shaped as:
    {
        "question_id": "q001",
        "stage": "dense_only",
        "recall_at_5": 1.0,
        "recall_at_10": 1.0,
        "mrr": 1.0,
        "precision_at_5": 0.4
    }

    WHY RAISE IF relevant_chunk_ids IS EMPTY?
    The eval harness refuses to produce metrics for unlabeled questions.
    If it silently used empty labels, every metric would be 0.0 for every
    question — producing a table that looks like "everything failed" when in
    reality labels are just missing. An exception is more honest.

    WHY RUN THE SAME STAGE ACROSS ALL QUESTIONS IN ONE LOOP?
    Fair comparison: the retriever's index is built once and stays constant.
    Running question 1 through dense_only, then question 1 through bm25_only,
    then question 2 through dense_only, etc. ensures no ordering effects.
    """

    # Filter to questions that have non-empty ground truth labels
    labeled_questions = [q for q in questions if q.get("relevant_chunk_ids")]

    if not labeled_questions:
        raise ValueError(
            "No questions have hand-labelled relevant chunk IDs in eval_questions.json. "
            "Run label-candidates, fill the label column, then run sync-labels."
        )

    rows: list[dict[str, float | str]] = []

    for question in labeled_questions:
        relevant = set(question["relevant_chunk_ids"])

        for stage_name, stage in stages.items():
            # Run this question through this retrieval stage
            metrics = summarise_retrieval(stage(question["question"]), relevant)
            rows.append({
                "question_id": question["id"],
                "stage": stage_name,
                **metrics,
            })

    return rows


def aggregate_by_stage(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    """
    Create one table row per stage from the per-question result rows.

    Takes the output of evaluate_stages() (one row per question-stage pair)
    and averages all the metrics per stage — giving you one summary row per
    stage that you can paste into your README as the ablation table.

    HOW IT WORKS:
    1. Collect all unique stage names in their original order (dict.fromkeys
       preserves insertion order while deduplicating, unlike set()).
    2. For each stage: filter rows for that stage, convert to floats, average.
    3. Return list of {"stage": ..., "recall_at_5": ..., ...} dicts.

    WHY dict.fromkeys() INSTEAD OF set()?
    set() doesn't preserve order. The ablation table should show stages in
    the order they appear in the data (dense_only first, reranker last),
    which matches the logical pipeline order.
    """

    output = []
    for stage in dict.fromkeys(str(row["stage"]) for row in rows):
        # Get all metric dicts for this stage (exclude the non-numeric columns)
        stage_rows = [
            {
                key: float(value)
                for key, value in row.items()
                if key not in {"stage", "question_id"}
            }
            for row in rows
            if row["stage"] == stage
        ]
        output.append({"stage": stage, **average_metric_rows(stage_rows)})

    return output


def write_csv(rows: list[dict], path: str) -> None:
    """
    Persist results so claims in the README can be regenerated.

    WHY SAVE TO CSV?
    Results CSVs serve as the audit trail: if someone questions your metrics,
    you can show the exact per-question results. The summary CSV (one row per
    stage) is what goes in the README table. The per-question CSV lets you
    diagnose WHY a stage performs differently on specific questions.

    CREATES PARENT DIRECTORY:
    mkdir(parents=True, exist_ok=True) ensures the "results/" directory exists
    before trying to write to it — avoids FileNotFoundError on fresh repos.

    FIELDNAMES FROM FIRST ROW:
    DictWriter infers column names from the first dict's keys. All rows must
    have the same keys — the ablation harness guarantees this.
    """

    if not rows:
        return  # Nothing to write — don't create an empty file

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
