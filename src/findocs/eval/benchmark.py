"""Measurement helpers for the no-retry/retry and grader comparisons."""

from dataclasses import dataclass
from time import perf_counter
from typing import Callable


@dataclass
class TimedResult:
    """A result plus wall-clock latency; cost and memory stay explicit inputs."""

    name: str
    accuracy: float
    latency_ms: float
    cost_per_call: float | None = None
    gpu_memory_gb: float | None = None


def measure_callable(name: str, fn: Callable[[], float], repeats: int = 3, **resources) -> TimedResult:
    """Run a grader/agent callable repeatedly and report mean latency."""

    timings = []
    answers = []
    for _ in range(repeats):
        start = perf_counter()
        answers.append(float(fn()))
        timings.append((perf_counter() - start) * 1000)
    return TimedResult(name, sum(answers) / len(answers), sum(timings) / len(timings), **resources)

