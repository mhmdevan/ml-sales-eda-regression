from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd


def _numeric_distribution(values: pd.Series) -> dict[str, float | int | str]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {
            "type": "numeric",
            "count": 0,
            "missing_ratio": 1.0,
        }

    quantiles = numeric.quantile([0.1, 0.5, 0.9])
    return {
        "type": "numeric",
        "count": int(numeric.shape[0]),
        "missing_ratio": float(1.0 - (numeric.shape[0] / max(values.shape[0], 1))),
        "mean": float(numeric.mean()),
        "std": float(numeric.std(ddof=0)),
        "min": float(numeric.min()),
        "p10": float(quantiles.loc[0.1]),
        "p50": float(quantiles.loc[0.5]),
        "p90": float(quantiles.loc[0.9]),
        "max": float(numeric.max()),
    }


def _categorical_distribution(values: pd.Series, *, top_k_categories: int) -> dict[str, object]:
    cleaned = values.dropna().astype(str)
    if cleaned.empty:
        return {
            "type": "categorical",
            "count": 0,
            "unique": 0,
            "missing_ratio": 1.0,
            "top_values": [],
        }

    frequencies = cleaned.value_counts(normalize=True).head(top_k_categories)
    top_values = [
        {"value": str(category), "ratio": float(ratio)}
        for category, ratio in frequencies.to_dict().items()
    ]
    return {
        "type": "categorical",
        "count": int(cleaned.shape[0]),
        "unique": int(cleaned.nunique()),
        "missing_ratio": float(1.0 - (cleaned.shape[0] / max(values.shape[0], 1))),
        "top_values": top_values,
    }


def summarize_feature_distributions(
    *,
    frame: pd.DataFrame,
    feature_names: list[str],
    top_k_categories: int = 5,
) -> dict[str, dict[str, object]]:
    if top_k_categories < 1:
        raise ValueError("top_k_categories must be at least 1.")

    distributions: dict[str, dict[str, object]] = {}
    for feature in feature_names:
        if feature not in frame.columns:
            distributions[feature] = {
                "present": False,
                "type": "missing",
                "count": 0,
                "missing_ratio": 1.0,
            }
            continue

        values = frame[feature]
        if pd.api.types.is_numeric_dtype(values):
            summary = _numeric_distribution(values)
        else:
            summary = _categorical_distribution(values, top_k_categories=top_k_categories)
        summary["present"] = True
        distributions[feature] = summary

    return distributions


def build_feature_distribution_log(
    *,
    reference_frame: pd.DataFrame,
    current_frame: pd.DataFrame,
    feature_names: list[str],
    top_k_categories: int = 5,
) -> dict[str, object]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feature_count": len(feature_names),
        "reference": summarize_feature_distributions(
            frame=reference_frame,
            feature_names=feature_names,
            top_k_categories=top_k_categories,
        ),
        "current": summarize_feature_distributions(
            frame=current_frame,
            feature_names=feature_names,
            top_k_categories=top_k_categories,
        ),
    }
