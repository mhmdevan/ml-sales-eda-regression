from __future__ import annotations

import numpy as np


def regression_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if y_true.size == 0:
        raise ValueError("y_true and y_pred must be non-empty.")

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(y_pred, quantiles)
    edges[0] = -np.inf
    edges[-1] = np.inf

    error = 0.0
    total = float(y_true.size)

    for start, end in zip(edges[:-1], edges[1:]):
        mask = (y_pred > start) & (y_pred <= end)
        if not np.any(mask):
            continue
        mean_true = float(np.mean(y_true[mask]))
        mean_pred = float(np.mean(y_pred[mask]))
        weight = float(np.sum(mask)) / total
        error += abs(mean_true - mean_pred) * weight

    return float(error)
