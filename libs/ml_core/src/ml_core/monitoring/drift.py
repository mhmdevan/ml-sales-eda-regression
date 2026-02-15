from __future__ import annotations

import numpy as np


def _coerce_non_empty(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError(f"{name} array must contain at least one finite numeric value.")
    return array


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
) -> float:
    reference = _coerce_non_empty(reference, name="reference")
    current = _coerce_non_empty(current, name="current")
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


def kolmogorov_smirnov_statistic(
    reference: np.ndarray,
    current: np.ndarray,
) -> float:
    reference = np.sort(_coerce_non_empty(reference, name="reference"))
    current = np.sort(_coerce_non_empty(current, name="current"))

    support = np.sort(np.unique(np.concatenate([reference, current])))
    ref_cdf = np.searchsorted(reference, support, side="right") / float(reference.size)
    cur_cdf = np.searchsorted(current, support, side="right") / float(current.size)
    statistic = np.max(np.abs(ref_cdf - cur_cdf))
    return float(statistic)


def _kolmogorov_p_value(statistic: float, effective_n: float, max_terms: int = 100) -> float:
    if statistic <= 0:
        return 1.0
    if effective_n <= 0:
        return 1.0
    adjusted = (effective_n + 0.12 + (0.11 / effective_n)) * statistic
    terms = [
        ((-1) ** (index - 1)) * np.exp(-2.0 * (index**2) * (adjusted**2))
        for index in range(1, max_terms + 1)
    ]
    p_value = 2.0 * float(np.sum(terms))
    return float(np.clip(p_value, 0.0, 1.0))


def kolmogorov_smirnov_test(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    alpha: float = 0.05,
) -> dict[str, float | int | bool]:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

    reference = _coerce_non_empty(reference, name="reference")
    current = _coerce_non_empty(current, name="current")

    statistic = kolmogorov_smirnov_statistic(reference=reference, current=current)
    effective_n = np.sqrt((reference.size * current.size) / float(reference.size + current.size))
    p_value = _kolmogorov_p_value(statistic=statistic, effective_n=float(effective_n))
    critical_value = np.sqrt(-0.5 * np.log(alpha / 2.0)) / effective_n

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "alpha": float(alpha),
        "critical_value": float(critical_value),
        "drift_detected": bool(statistic > critical_value),
        "n_reference": int(reference.size),
        "n_current": int(current.size),
    }
