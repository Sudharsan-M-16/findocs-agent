"""Evaluate relevance graders with the same labels and timing harness."""

import time
from collections.abc import Callable


PredictionFn = Callable[[dict], int]


def classification_metrics(labels: list[int], predictions: list[int]) -> dict[str, float]:
    """Return accuracy, precision, recall, and F1 for binary relevance labels."""

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
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def evaluate_grader(name: str, rows: list[dict], predict: PredictionFn, cost_per_call: float = 0.0, gpu_memory_gb: float = 0.0) -> dict[str, float | str]:
    """Measure quality, latency, estimated cost, and GPU memory for one grader."""

    labels = [int(row["label"]) for row in rows]
    start = time.perf_counter()
    predictions = [int(predict(row)) for row in rows]
    elapsed = time.perf_counter() - start
    metrics = classification_metrics(labels, predictions)
    metrics.update({
        "grader": name,
        "latency_ms": elapsed * 1000 / max(len(rows), 1),
        "cost_per_call": cost_per_call,
        "gpu_memory_gb": gpu_memory_gb,
    })
    return metrics
