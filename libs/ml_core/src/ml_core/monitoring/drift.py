from __future__ import annotations

import numpy as np


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
) -> float:
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if reference.size == 0 or current.size == 0:
        raise ValueError("Both reference and current arrays must be non-empty.")
    if bins < 2:
        raise ValueError("bins must be at least 2.")

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(reference, quantiles)
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_ratio = ref_counts / np.maximum(ref_counts.sum(), 1)
    cur_ratio = cur_counts / np.maximum(cur_counts.sum(), 1)

    epsilon = 1e-8
    value = np.sum((cur_ratio - ref_ratio) * np.log((cur_ratio + epsilon) / (ref_ratio + epsilon)))
    return float(value)
