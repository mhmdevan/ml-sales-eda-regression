from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except Exception:
        return None
    if np.isfinite(parsed):
        return parsed
    return None


def _column_profile(values: pd.Series) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "dtype": str(values.dtype),
        "missing_count": int(values.isna().sum()),
        "missing_ratio": float(values.isna().mean()) if values.shape[0] else 0.0,
        "unique_count": int(values.nunique(dropna=True)),
    }
    if pd.api.types.is_numeric_dtype(values):
        numeric = pd.to_numeric(values, errors="coerce")
        quantiles = numeric.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_dict()
        profile.update(
            {
                "type": "numeric",
                "mean": _safe_float(numeric.mean()),
                "std": _safe_float(numeric.std(ddof=0)),
                "min": _safe_float(numeric.min()),
                "max": _safe_float(numeric.max()),
                "quantiles": {str(key): _safe_float(value) for key, value in quantiles.items()},
            }
        )
    else:
        clean = values.dropna().astype(str)
        top_values = clean.value_counts(normalize=True).head(10).to_dict()
        profile.update(
            {
                "type": "categorical",
                "top_values": {str(key): float(value) for key, value in top_values.items()},
            }
        )
    return profile


def _dataset_fingerprint(frame: pd.DataFrame, columns: list[str]) -> str:
    if not columns:
        return ""
    selected = frame.loc[:, columns]
    row_hash = pd.util.hash_pandas_object(selected, index=False).to_numpy()
    digest = hashlib.sha256()
    digest.update(",".join(columns).encode("utf-8"))
    digest.update(str(selected.shape[0]).encode("utf-8"))
    digest.update(row_hash.tobytes())
    return digest.hexdigest()


def build_tabular_eda_payload(
    *,
    frame: pd.DataFrame,
    project_name: str,
    report_version: str,
    numeric_features: list[str],
    categorical_features: list[str],
    target_name: str,
) -> dict[str, Any]:
    available_numeric = [feature for feature in numeric_features if feature in frame.columns]
    available_categorical = [feature for feature in categorical_features if feature in frame.columns]
    available_columns = [column for column in frame.columns if column in set(available_numeric + available_categorical + [target_name])]

    profiles = {column: _column_profile(frame[column]) for column in available_columns}
    correlation: dict[str, float] = {}
    if target_name in frame.columns and available_numeric:
        target = pd.to_numeric(frame[target_name], errors="coerce")
        for feature in available_numeric:
            feature_values = pd.to_numeric(frame[feature], errors="coerce")
            value = feature_values.corr(target)
            if value is not None and np.isfinite(value):
                correlation[feature] = float(value)

    numeric_frame = frame.loc[:, available_numeric] if available_numeric else pd.DataFrame()
    if not numeric_frame.empty:
        corr_matrix = numeric_frame.corr().fillna(0.0)
        correlation_matrix = {
            row: {column: float(corr_matrix.loc[row, column]) for column in corr_matrix.columns}
            for row in corr_matrix.index
        }
    else:
        correlation_matrix = {}

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_name": project_name,
        "report_version": report_version,
        "row_count": int(frame.shape[0]),
        "column_count": int(frame.shape[1]),
        "feature_groups": {
            "numeric_features": available_numeric,
            "categorical_features": available_categorical,
            "target_name": target_name,
        },
        "dataset_fingerprint": {
            "algorithm": "sha256",
            "value": _dataset_fingerprint(frame=frame, columns=available_columns),
        },
        "profiles": profiles,
        "target_correlations": correlation,
        "numeric_correlation_matrix": correlation_matrix,
    }
    return payload


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# {payload['project_name']} EDA Report",
        "",
        f"- Report version: {payload['report_version']}",
        f"- Generated at (UTC): {payload['generated_at']}",
        f"- Rows: {payload['row_count']}",
        f"- Columns: {payload['column_count']}",
        f"- Dataset fingerprint: `{payload['dataset_fingerprint']['value']}`",
        "",
        "## Target Correlations",
    ]
    correlations = dict(payload.get("target_correlations", {}))
    if correlations:
        for feature, value in sorted(correlations.items(), key=lambda item: abs(item[1]), reverse=True):
            lines.append(f"- {feature}: {value:.6f}")
    else:
        lines.append("- No numeric target correlation was computed.")

    lines.append("")
    lines.append("## Feature Profiles")
    profiles = dict(payload.get("profiles", {}))
    for feature in sorted(profiles):
        profile = profiles[feature]
        feature_type = profile.get("type")
        missing_ratio = float(profile.get("missing_ratio", 0.0))
        lines.append(f"- {feature} ({feature_type}) missing={missing_ratio:.4f}")

    return "\n".join(lines) + "\n"


def _write_optional_plots(
    *,
    frame: pd.DataFrame,
    numeric_features: list[str],
    target_name: str,
    output_dir: Path,
) -> dict[str, str | None]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return {"target_histogram": None, "correlation_heatmap": None}

    paths: dict[str, str | None] = {"target_histogram": None, "correlation_heatmap": None}
    if target_name in frame.columns:
        target_path = output_dir / "target_histogram.png"
        values = pd.to_numeric(frame[target_name], errors="coerce").dropna()
        if not values.empty:
            fig, ax = plt.subplots(figsize=(8, 4.8))
            ax.hist(values.to_numpy(dtype=float), bins=30)
            ax.set_title(f"Target Distribution: {target_name}")
            ax.set_xlabel(target_name)
            ax.set_ylabel("Count")
            fig.tight_layout()
            fig.savefig(target_path)
            plt.close(fig)
            paths["target_histogram"] = str(target_path)

    numeric = [feature for feature in numeric_features if feature in frame.columns]
    if numeric:
        corr = frame.loc[:, numeric].corr()
        if not corr.empty:
            heatmap_path = output_dir / "numeric_correlation_heatmap.png"
            fig, ax = plt.subplots(figsize=(7, 6))
            image = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1.0, vmax=1.0)
            ax.set_xticks(range(len(corr.columns)))
            ax.set_xticklabels(list(corr.columns), rotation=90)
            ax.set_yticks(range(len(corr.index)))
            ax.set_yticklabels(list(corr.index))
            ax.set_title("Numeric Correlation Heatmap")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(heatmap_path)
            plt.close(fig)
            paths["correlation_heatmap"] = str(heatmap_path)
    return paths


def write_eda_artifacts(
    *,
    frame: pd.DataFrame,
    numeric_features: list[str],
    target_name: str,
    output_dir: Path,
    payload: dict[str, Any],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "eda_report.json"
    markdown_path = output_dir / "EDA_REPORT.md"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown_path.write_text(_render_markdown(payload), encoding="utf-8")

    plot_paths = _write_optional_plots(
        frame=frame,
        numeric_features=numeric_features,
        target_name=target_name,
        output_dir=output_dir,
    )
    plot_manifest = output_dir / "plots.json"
    plot_manifest.write_text(json.dumps(plot_paths, indent=2), encoding="utf-8")

    return {
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "plots_path": str(plot_manifest),
    }
