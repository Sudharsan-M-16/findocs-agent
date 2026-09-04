"""
grader_eval.py — Relevance Grader Evaluation Harness
=====================================================
PURPOSE: Compare any two relevance graders (LLM vs QLoRA model) on the same
labeled dataset using the same timing harness.

THE COMPARISON THIS HARNESS PRODUCES:
---------------------------------------------------------------------------
Metric          LLM Grader         QLoRA Grader
---------------------------------------------------------------------------
Accuracy        measure            measure
Precision       measure            measure
Recall          measure            measure
F1              measure            measure
Latency (ms)    measure            measure
Cost/call       configure           $0.00 (local inference)
GPU memory      measure            measure
---------------------------------------------------------------------------

Populate this table only after both prediction functions have run on the same
held-out human-labeled rows. Do not use target values as results.

WHAT "GRADER" MEANS IN THIS CONTEXT:
A grader takes a (question, chunk) pair and returns 1 (relevant) or 0 (irrelevant).
In graph.py, the grade() node currently uses word overlap as a heuristic grader.
The QLoRA model would replace that heuristic.

HOW TO USE THIS MODULE:
1. Define an LLM-based grader function:
       def llm_grader(row: dict) -> int:
           response = call_gpt4(row["question"], row["chunk_text"])
           return 1 if "relevant" in response else 0
2. Define a QLoRA grader function (after training):
       def qlora_grader(row: dict) -> int:
           output = model.generate(prompt_for_pair(row["question"], chunk))
           return 1 if "relevant" in output else 0
3. Load the labeled pairs: rows = load_labeled_pairs("data/grader_label_sheet.csv")
4. Evaluate both: llm_results = evaluate_grader("LLM", rows, llm_grader)
5. Compare: print(llm_results); print(qlora_results)
"""

import time
from collections.abc import Callable
from pathlib import Path


# Type for any grader function: takes one labeled row dict, returns 0 or 1
PredictionFn = Callable[[dict], int]


def classification_metrics(
    labels: list[int], predictions: list[int]
) -> dict[str, float]:
    """
    Return accuracy, precision, recall, and F1 for binary relevance labels.

    CONFUSION MATRIX COMPONENTS:
    TP (True Positive):  labeled relevant (1), predicted relevant (1). Correct.
    FP (False Positive): labeled irrelevant (0), predicted relevant (1). False alarm.
    FN (False Negative): labeled relevant (1), predicted irrelevant (0). Missed.
    TN (True Negative):  labeled irrelevant (0), predicted irrelevant (0). Correct.

    THE FOUR METRICS:
    -----------------
    Accuracy  = (TP + TN) / total
                Overall correctness. Can be misleading with imbalanced classes
                (if 90% are irrelevant, always predicting 0 gives 90% accuracy).

    Precision = TP / (TP + FP)
                "Of all the chunks I said were relevant, how many actually were?"
                Low precision → noisy grader (lets through irrelevant chunks).

    Recall    = TP / (TP + FN)
                "Of all truly relevant chunks, how many did I find?"
                Low recall → conservative grader (misses good chunks).

    F1        = 2 × Precision × Recall / (Precision + Recall)
                Harmonic mean. Balances precision and recall.
                F1 = 1.0 → perfect. F1 = 0.0 → completely wrong.

    WHY HARMONIC MEAN AND NOT ARITHMETIC MEAN?
    Arithmetic mean (P+R)/2 can be gamed: predict all-1 → recall=1.0,
    precision=base_rate, arithmetic mean looks decent.
    Harmonic mean heavily penalises when one of P/R is near zero.

    GUARDS:
    max(tp + fp, 1): prevents division by zero when no positives predicted.
    max(tp + fn, 1): prevents division by zero when no true positives in labels.
    1e-12 in F1:     prevents division by zero when P=R=0.
    """

    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have the same length.")

    total = max(len(labels), 1)
    tp = sum(label == 1 and pred == 1 for label, pred in zip(labels, predictions))
    fp = sum(label == 0 and pred == 1 for label, pred in zip(labels, predictions))
    fn = sum(label == 1 and pred == 0 for label, pred in zip(labels, predictions))

    accuracy = sum(label == pred for label, pred in zip(labels, predictions)) / total
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_grader(
    name: str,
    rows: list[dict],
    predict: PredictionFn,
    cost_per_call: float = 0.0,
    gpu_memory_gb: float = 0.0,
) -> dict[str, float | str]:
    """
    Measure quality, latency, estimated cost, and GPU memory for one grader.

    ARGUMENTS:
    name          : Human-readable grader name for the output table ("LLM", "QLoRA-1.5B").
    rows          : Completed label rows from load_labeled_pairs(). Each row has
                    "label", "question", "chunk_text" etc.
    predict       : A function(row) → 0 or 1. Called once per row.
    cost_per_call : If using an LLM API, the approximate cost in USD per API call.
                    Set to 0.0 for local models.
    gpu_memory_gb : Peak GPU memory in GB during inference. Measure with
                    torch.cuda.max_memory_allocated() / 1e9 after inference.

    HOW LATENCY IS MEASURED:
    time.perf_counter() is a high-resolution wall-clock timer. We time the
    ENTIRE predict loop (not just one call) and then divide by count. This
    gives average ms/call including Python overhead. On an LLM API this will
    be 500-2000ms. On a local QLoRA model it should be 20-100ms.

    WHY INCLUDE COST AND MEMORY?
    The engineering decision isn't just "which is more accurate?" It's:
    "Is the accuracy improvement of the LLM worth 4x cost and 10x latency?"
    Presenting all four dimensions (accuracy + latency + cost + memory) is what
    makes this comparison look like engineering, not just benchmark-running.

    RETURNS:
    One dict with all metrics + grader name, ready to be row in a comparison CSV:
    {
        "grader": "LLM",
        "accuracy": 0.92,
        "precision": 0.89,
        "recall": 0.94,
        "f1": 0.91,
        "latency_ms": 823.4,
        "cost_per_call": 0.002,
        "gpu_memory_gb": 0.0
    }
    """

    labels = [int(row["label"]) for row in rows]

    # Time the entire prediction loop
    start = time.perf_counter()
    predictions = [int(predict(row)) for row in rows]
    elapsed = time.perf_counter() - start

    # Compute classification metrics
    metrics = classification_metrics(labels, predictions)

    # Add operational metrics
    metrics.update({
        "grader": name,
        # Total elapsed time divided by number of predictions = ms per call
        "latency_ms": elapsed * 1000 / max(len(rows), 1),
        "cost_per_call": cost_per_call,
        "gpu_memory_gb": gpu_memory_gb,
    })

    return metrics


def main() -> None:
    """
    CLI runner for relevance grader evaluation.

    USAGE:
        py -m findocs.finetune.grader_eval --labels data/grader_label_sheet.csv --model models/qlora-grader
    """

    import argparse
    import json
    from findocs.finetune.dataset import load_labeled_pairs

    parser = argparse.ArgumentParser(description="Evaluate relevance grader model performance")
    parser.add_argument(
        "--labels",
        default="data/grader_label_sheet.csv",
        help="Path to labeled CSV",
    )
    parser.add_argument(
        "--model",
        default="models/qlora-grader",
        help="Path to fine-tuned LoRA adapter directory",
    )
    parser.add_argument(
        "--output",
        default="results/grader_eval.json",
        help="JSON file where measured grader results are saved",
    )
    args = parser.parse_args()

    rows = load_labeled_pairs(args.labels)
    if not rows:
        print(f"No labeled pairs found in {args.labels}.")
        return

    # Baseline Heuristic Grader (word overlap)
    def word_overlap_grader(row: dict) -> int:
        q_words = set(row["question"].lower().split())
        c_words = set(row["chunk_text"].lower().split())
        overlap = len(q_words & c_words)
        return 1 if overlap >= 2 else 0

    heuristic_res = evaluate_grader("Heuristic (Word Overlap)", rows, word_overlap_grader)
    results = [heuristic_res]
    print("\n=== Heuristic Grader Baseline ===")
    print(json.dumps(heuristic_res, indent=2))

    # Evaluate Fine-Tuned QLoRA Adapter if available
    try:
        import torch
        import unsloth
        from unsloth import FastLanguageModel

        print(f"\nLoading fine-tuned QLoRA grader from {args.model}...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=2048,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)

        def qlora_grader(row: dict) -> int:
            prompt = (
                "### Instruction\n"
                "Decide whether the SEC filing chunk contains enough evidence "
                "to help answer the question.\n\n"
                f"### Question\n{row['question']}\n\n"
                f"### Chunk\n{row['chunk_text']}\n\n"
                "### Answer\n"
            )
            inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
            outputs = model.generate(**inputs, max_new_tokens=10, use_cache=True)
            text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            # Answer section parsing
            answer_part = text.split("### Answer\n")[-1] if "### Answer\n" in text else text
            return 1 if "relevant" in answer_part.lower() and "irrelevant" not in answer_part.lower() else 0

        peak_vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        qlora_res = evaluate_grader("Fine-Tuned QLoRA (Qwen2.5-1.5B)", rows, qlora_grader, gpu_memory_gb=peak_vram)
        print("\n=== Fine-Tuned QLoRA Grader ===")
        print(json.dumps(qlora_res, indent=2))
        results.append(qlora_res)

    except Exception as exc:
        print(f"\n[Note] QLoRA evaluation skipped or failed: {exc}")

    # Persist the exact measurements so a resume claim can be audited later.
    # The output is written even when QLoRA evaluation fails, preserving the
    # heuristic baseline and the failure context in the terminal log.
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved grader results to {output}")


if __name__ == "__main__":
    main()
