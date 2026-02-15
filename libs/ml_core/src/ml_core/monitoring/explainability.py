from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import csv
import json
from typing import Any

import numpy as np
import pandas as pd

from ml_core.features.schema import FeatureSchema


def _is_tree_based_regressor(regressor: Any) -> bool:
    return hasattr(regressor, "feature_importances_") or hasattr(regressor, "estimators_")


def _pipeline_parts(pipeline: Any) -> tuple[Any | None, Any]:
    named_steps = getattr(pipeline, "named_steps", {})
    preprocessor = named_steps.get("preprocessor")
    regressor = named_steps.get("regressor", pipeline)
    return preprocessor, regressor


def _to_dense(values: Any) -> np.ndarray:
    if hasattr(values, "toarray"):
        return np.asarray(values.toarray(), dtype=float)
    return np.asarray(values, dtype=float)


def _transformed_feature_names(*, preprocessor: Any | None, transformed: np.ndarray) -> list[str]:
    if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
        return [str(name) for name in preprocessor.get_feature_names_out()]
    return [f"f{i}" for i in range(transformed.shape[1])]


def _base_feature_name(feature_name: str, schema: FeatureSchema) -> str:
    if feature_name.startswith("numeric__"):
        return feature_name.replace("numeric__", "", 1)
    if feature_name.startswith("categorical__"):
        suffix = feature_name.replace("categorical__", "", 1)
        for categorical in schema.categorical_features:
            token = f"{categorical}_"
            if suffix.startswith(token):
                return categorical
        return suffix.split("_", 1)[0]
    return feature_name


def _aggregate_importance(
    *,
    transformed_names: list[str],
    values: np.ndarray,
    schema: FeatureSchema,
) -> dict[str, float]:
    aggregated: dict[str, float] = {}
    for feature_name, value in zip(transformed_names, values, strict=False):
        base_name = _base_feature_name(feature_name, schema)
        aggregated[base_name] = aggregated.get(base_name, 0.0) + float(value)
    return aggregated


def _import_shap() -> Any | None:
    try:
        import shap
    except Exception:
        return None
    return shap


def explain_single_prediction(
    *,
    pipeline: Any,
    frame: pd.DataFrame,
    schema: FeatureSchema,
    top_k: int = 10,
) -> dict[str, Any]:
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if frame.shape[0] != 1:
        raise ValueError("frame must include exactly one row for local explanation.")

    preprocessor, regressor = _pipeline_parts(pipeline)
    if not _is_tree_based_regressor(regressor):
        return {
            "available": False,
            "reason": "best model is not tree-based",
        }

    shap = _import_shap()
    if shap is None:
        return {
            "available": False,
            "reason": "shap is not installed",
        }

    transformed_raw = preprocessor.transform(frame) if preprocessor is not None else frame
    transformed = _to_dense(transformed_raw)
    feature_names = _transformed_feature_names(preprocessor=preprocessor, transformed=transformed)

    try:
        explainer = shap.TreeExplainer(regressor)
        shap_values = explainer.shap_values(transformed)
    except Exception as exc:
        return {
            "available": False,
            "reason": f"shap computation failed: {exc}",
        }

    shap_array = np.asarray(shap_values, dtype=float)
    if shap_array.ndim == 3:
        shap_array = shap_array[0]
    if shap_array.ndim == 1:
        row_values = shap_array
    else:
        row_values = shap_array[0]

    aggregated = _aggregate_importance(
        transformed_names=feature_names,
        values=row_values,
        schema=schema,
    )
    ranked = sorted(aggregated.items(), key=lambda item: abs(item[1]), reverse=True)
    top = ranked[:top_k]

    expected = getattr(explainer, "expected_value", None)
    if isinstance(expected, np.ndarray):
        base_value = float(np.asarray(expected).reshape(-1)[0])
    elif expected is None:
        base_value = None
    else:
        base_value = float(expected)

    return {
        "available": True,
        "reason": None,
        "method": "shap_tree",
        "base_value": base_value,
        "contributions": {name: float(value) for name, value in top},
    }


def generate_shap_artifacts(
    *,
    pipeline: Any,
    frame: pd.DataFrame,
    schema: FeatureSchema,
    output_dir: Path,
    sample_size: int = 512,
    top_k: int = 20,
) -> dict[str, Any]:
    if sample_size < 1:
        raise ValueError("sample_size must be at least 1.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "shap_summary.json"
    csv_path = output_dir / "shap_feature_importance.csv"

    preprocessor, regressor = _pipeline_parts(pipeline)
    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "available": False,
        "method": None,
        "reason": None,
        "top_features": [],
        "sample_size": 0,
    }

    if frame.empty:
        summary["reason"] = "empty training frame"
    elif not _is_tree_based_regressor(regressor):
        summary["reason"] = "best model is not tree-based"
    else:
        shap = _import_shap()
        if shap is None:
            summary["reason"] = "shap is not installed"
        else:
            sample_frame = frame.sample(n=min(sample_size, frame.shape[0]), random_state=42).copy()
            transformed_raw = preprocessor.transform(sample_frame) if preprocessor is not None else sample_frame
            transformed = _to_dense(transformed_raw)
            feature_names = _transformed_feature_names(preprocessor=preprocessor, transformed=transformed)
            try:
                explainer = shap.TreeExplainer(regressor)
                shap_values = explainer.shap_values(transformed)
                shap_array = np.asarray(shap_values, dtype=float)
                if shap_array.ndim == 3:
                    shap_array = shap_array[0]
                if shap_array.ndim == 1:
                    shap_array = np.expand_dims(shap_array, axis=0)
                absolute_mean = np.mean(np.abs(shap_array), axis=0)
                aggregated = _aggregate_importance(
                    transformed_names=feature_names,
                    values=absolute_mean,
                    schema=schema,
                )
                ranked = sorted(aggregated.items(), key=lambda item: abs(item[1]), reverse=True)
                top_features = [
                    {"feature": feature, "mean_abs_shap": float(value)}
                    for feature, value in ranked[:top_k]
                ]
                summary.update(
                    {
                        "available": True,
                        "method": "shap_tree",
                        "reason": None,
                        "top_features": top_features,
                        "sample_size": int(sample_frame.shape[0]),
                    }
                )
                with csv_path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["feature", "mean_abs_shap"])
                    writer.writeheader()
                    for feature, value in ranked:
                        writer.writerow({"feature": feature, "mean_abs_shap": float(value)})
            except Exception as exc:
                summary["reason"] = f"shap computation failed: {exc}"

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "summary_path": str(summary_path),
        "importance_csv_path": str(csv_path if csv_path.exists() else ""),
        "summary": summary,
    }
