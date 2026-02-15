from __future__ import annotations

import math

import numpy as np


def fit_split_conformal_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    alpha: float = 0.1,
) -> dict[str, float | int | str]:
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

    errors = np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))
    if errors.size == 0:
        raise ValueError("Conformal calibration requires at least one sample.")
    if not np.all(np.isfinite(errors)):
        raise ValueError("Conformal calibration requires finite residuals.")

    n = int(errors.size)
    quantile_level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    q_hat = float(np.quantile(errors, quantile_level, method="higher"))
    empirical_coverage = float(np.mean(errors <= q_hat))

    return {
        "method": "split_conformal_abs_residual",
        "alpha": float(alpha),
        "nominal_coverage": float(1 - alpha),
        "q_hat": q_hat,
        "calibration_size": n,
        "empirical_coverage": empirical_coverage,
    }


def conformal_interval(
    prediction: float,
    q_hat: float,
) -> tuple[float, float]:
    lower = float(prediction - q_hat)
    upper = float(prediction + q_hat)
    return (min(lower, upper), max(lower, upper))
