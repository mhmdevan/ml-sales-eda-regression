from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json

import pandas as pd

from ml_core.monitoring.drift import kolmogorov_smirnov_test, population_stability_index


def compute_numeric_drift_summary(
    *,
    reference_frame: pd.DataFrame,
    current_frame: pd.DataFrame,
    numeric_features: list[str],
    psi_threshold: float = 0.2,
    ks_alpha: float = 0.05,
) -> dict[str, object]:
    feature_psi: dict[str, float] = {}
    feature_ks: dict[str, dict[str, float | int | bool]] = {}
    drifted_features: list[str] = []

    for feature in numeric_features:
        ref_values = pd.to_numeric(reference_frame[feature], errors="coerce").dropna().to_numpy(dtype=float)
        cur_values = pd.to_numeric(current_frame[feature], errors="coerce").dropna().to_numpy(dtype=float)
        if ref_values.size == 0 or cur_values.size == 0:
            continue
        psi = float(population_stability_index(ref_values, cur_values))
        ks_result = kolmogorov_smirnov_test(ref_values, cur_values, alpha=ks_alpha)
        feature_psi[feature] = psi
        feature_ks[feature] = ks_result
        if psi >= psi_threshold or bool(ks_result["drift_detected"]):
            drifted_features.append(feature)

    total = max(len(feature_psi), 1)
    drift_ratio = len(drifted_features) / total

    return {
        "feature_psi": feature_psi,
        "feature_ks": feature_ks,
        "drifted_features": drifted_features,
        "drift_ratio": drift_ratio,
        "psi_threshold": psi_threshold,
        "ks_alpha": ks_alpha,
    }


def build_drift_alert_payload(
    *,
    project_name: str,
    drift_summary: dict[str, object],
    alert_ratio_threshold: float,
) -> dict[str, object]:
    drift_ratio = float(drift_summary["drift_ratio"])
    alert = drift_ratio >= alert_ratio_threshold

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_name": project_name,
        "alert": alert,
        "alert_ratio_threshold": alert_ratio_threshold,
        "drift": drift_summary,
    }


def write_alert_artifacts(
    *,
    output_dir: Path,
    payload: dict[str, object],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "drift_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    alert_path = output_dir / "ALERT.txt"
    if bool(payload.get("alert")):
        alert_path.write_text(
            f"Drift alert for {payload['project_name']} at {payload['generated_at']}",
            encoding="utf-8",
        )
    elif alert_path.exists():
        alert_path.unlink()

    return {
        "summary_path": str(summary_path),
        "alert_path": str(alert_path),
    }
