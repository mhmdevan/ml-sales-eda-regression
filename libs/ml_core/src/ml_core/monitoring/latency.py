from __future__ import annotations

from time import perf_counter
from typing import Any, Callable

import numpy as np


def benchmark_latency(
    predict_fn: Callable[[Any], Any],
    payload: Any,
    iterations: int = 100,
) -> dict[str, float]:
    if iterations < 1:
        raise ValueError("iterations must be at least 1.")

    durations: list[float] = []
    for _ in range(iterations):
        started = perf_counter()
        predict_fn(payload)
        finished = perf_counter()
        durations.append((finished - started) * 1000)

    values = np.asarray(durations, dtype=float)
    return {
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "max_ms": float(np.max(values)),
    }
