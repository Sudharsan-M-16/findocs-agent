"""
dataset.py — QLoRA Training Dataset Builder
============================================
PURPOSE: Generate the labeled data that trains the small relevance grader.

THE PIPELINE THIS ENABLES:
SEC filings → chunks → retrieval candidates → THIS FILE LABELS THEM
→ qlora_train.py trains on the labels → grader_eval.py measures quality
→ a small model replaces expensive LLM grader calls

WHY YOU NEED A LABELED DATASET:
Fine-tuning is supervised learning. The model learns "given this (query, chunk) pair,
predict relevant or irrelevant" — but to learn that, it needs (query, chunk, label)
examples where a human has determined the ground truth.

THERE IS NO SHORTCUT HERE:
You cannot generate fake labels and train on them. The fine-tuned model would
learn to predict fake relevance — useless for real evaluation. The label sheet
must be filled by inspecting actual retrieved chunks from the Apple 10-K.

THE WORKFLOW (see README for the full steps):
1. Run: py -m findocs.cli label-candidates --companies AAPL
   → Generates data/grader_label_sheet.csv with 100 candidate rows.
2. Open the CSV. For each row, read chunk_text and fill label:
   - label=1 if this chunk actually answers the question.
   - label=0 if it doesn't.
3. Run: py -m findocs.cli sync-labels
   → Copies relevant chunk IDs into data/eval_questions.json for retrieval eval.
4. Run: py -m findocs.finetune.qlora_train --labels data/grader_label_sheet.csv
   → Trains the QLoRA grader on your labeled pairs.
"""

import csv
from pathlib import Path

from findocs.types import Chunk, RetrievedChunk


def candidate_rows(
    question: dict, results: list[RetrievedChunk]
) -> list[dict[str, str | int]]:
    """
    Turn retrieved candidates into rows a human can label relevant or not.

    EACH ROW CONTAINS:
    - question_id : links this row to the question in eval_questions.json
    - question    : the actual question text (saves you from looking it up)
    - chunk_id    : the chunk identifier (for copying into eval_questions.json)
    - company     : which filing this chunk came from
    - section     : which SEC Item this chunk is from (helps with labeling)
    - rank        : the retrieval rank (rank 1 = model's best guess)
    - label       : EMPTY — this column is for YOU to fill in (1 or 0)
    - chunk_text  : first 1800 characters of the chunk text for reading

    WHY 1800 CHARACTERS?
    Enough to read the chunk without scrolling forever in a spreadsheet. The full
    chunk may be up to 1800 characters (our default chunk size), so this captures
    all of it for small chunks and the beginning of large ones.

    WHY INCLUDE SECTION?
    When labeling, knowing "this chunk is from ITEM 7 (MD&A)" helps you quickly
    judge whether it's likely to contain revenue figures. You don't need to read
    the full text if the section is obviously wrong for the question type.
    """

    rows: list[dict[str, str | int]] = []
    for result in results:
        rows.append({
            "question_id": question["id"],
            "question": question["question"],
            "chunk_id": result.chunk.chunk_id,
            "company": result.chunk.company,
            "section": result.chunk.section,
            "rank": result.rank,
            "label": "",  # You fill this in: 1 = relevant, 0 = irrelevant
            "chunk_text": result.chunk.text[:1800],
        })
    return rows


def write_label_sheet(rows: list[dict[str, str | int]], path: str) -> None:
    """
    Write a CSV where the only manual column is label: 1 relevant, 0 irrelevant.

    CREATES PARENT DIRECTORIES if they don't exist (data/ might not exist on
    fresh clone before first run).

    WHY CSV AND NOT JSON OR SQLITE?
    A CSV opens in Excel/Sheets directly — no conversion needed.
    Labeling is a manual process; people label in spreadsheets, not code editors.
    DictWriter writes the exact column order from the first row's keys.
    """

    if not rows:
        raise ValueError("No rows to write.")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_labeled_pairs(path: str) -> list[dict[str, str | int]]:
    """
    Load only completed rows so unfinished labels do not enter training.

    WHAT "COMPLETED" MEANS:
    A row is complete if its "label" column is exactly "0" or "1" (as strings,
    since CSV reads everything as strings). Rows with empty label ("") are
    skipped — they haven't been labeled yet.

    WHY STRICT STRING COMPARISON ("0" and "1")?
    CSV files store everything as strings. If we checked `row["label"] == 0`
    (integer comparison), it would never match because CSV values are strings.
    The set {"0", "1"} is the correct filter.

    USAGE:
    This function is called by:
    1. qlora_train.py: load completed labels to create training examples.
    2. relevant_ids_by_question(): extract relevant chunk IDs for eval JSON.
    3. grader_eval.py: load labels to measure grader accuracy.
    """

    with Path(path).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    # Keep only rows where label was explicitly set to "0" or "1"
    return [row for row in rows if str(row.get("label", "")).strip() in {"0", "1"}]


def relevant_ids_by_question(path: str) -> dict[str, list[str]]:
    """
    Collect chunk IDs marked relevant in the human label sheet.

    USED BY: sync_labels() in cli.py to update eval_questions.json.

    LOGIC:
    Load all completed labels. For each row where label == "1" (relevant),
    group the chunk_id under the question_id. Deduplicate within each group
    (a chunk that appeared for the same question multiple times should only
    be listed once as a relevant answer).

    RETURNS:
    {
        "q001": ["section-42", "section-7"],  # chunks that are relevant for q001
        "q002": ["section-18"],               # relevant chunk for q002
        ...
    }
    This dict is then merged into eval_questions.json by sync_labels().
    """

    labels = load_labeled_pairs(path)
    grouped: dict[str, list[str]] = {}

    for row in labels:
        if str(row["label"]).strip() == "1":  # Only relevant labels
            grouped.setdefault(row["question_id"], [])
            # Deduplicate: same chunk ID might appear in multiple rows
            if row["chunk_id"] not in grouped[row["question_id"]]:
                grouped[row["question_id"]].append(row["chunk_id"])

    return grouped


def prompt_for_pair(question: str, chunk: Chunk) -> str:
    """
    Format the exact supervised task the small grader will learn.

    THIS IS THE PROMPT TEMPLATE FOR INFERENCE:
    When the trained QLoRA model is used as a grader during retrieval, you'd
    call it with this exact prompt format. The model learned to respond with
    "relevant" or "irrelevant" during training (see qlora_train.py).

    WHY "relevant or irrelevant" AND NOT A SCORE?
    Binary classification is simpler to train and measure. A score would
    require calibration. "relevant" / "irrelevant" maps directly to "pass the
    grade" / "rewrite the query" in graph.py's grade node.

    NOTE: The training prompt in qlora_train.py is formatted slightly differently
    (includes ### headers for Qwen2.5's instruction format). This function shows
    the conceptual template; qlora_train.py has the exact training format.
    """

    return (
        "Decide whether the SEC filing chunk contains enough evidence "
        "to help answer the question.\n"
        f"Question: {question}\n"
        f"Chunk: {chunk.text}\n"
        "Answer with only relevant or irrelevant."
    )
