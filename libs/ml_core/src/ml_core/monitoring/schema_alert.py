from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
from typing import Mapping

import pandas as pd


def detect_schema_changes(
    *,
    expected_features: list[str] | tuple[str, ...],
    current_frame: pd.DataFrame,
    expected_dtypes: Mapping[str, str] | None = None,
) -> dict[str, object]:
    if not expected_features:
        raise ValueError("expected_features must include at least one feature.")

    expected = list(expected_features)
    expected_set = set(expected)
    observed = list(current_frame.columns)
    observed_set = set(observed)

    missing_features = sorted(expected_set - observed_set)
    unexpected_features = sorted(observed_set - expected_set)
    dtype_mismatches: dict[str, dict[str, str]] = {}

    if expected_dtypes:
        for feature in expected:
            if feature not in current_frame.columns:
                continue
            expected_dtype = str(expected_dtypes.get(feature, ""))
            if not expected_dtype:
                continue
            observed_dtype = str(current_frame[feature].dtype)
            if observed_dtype != expected_dtype:
                dtype_mismatches[feature] = {
                    "expected": expected_dtype,
                    "observed": observed_dtype,
                }

    expected_order_in_current = [column for column in observed if column in expected_set]
    canonical_order = [column for column in expected if column in observed_set]
    order_changed = expected_order_in_current != canonical_order
    alert = bool(missing_features or unexpected_features or dtype_mismatches or order_changed)

    return {
        "expected_feature_count": len(expected),
        "observed_feature_count": len(observed),
        "missing_features": missing_features,
        "unexpected_features": unexpected_features,
        "dtype_mismatches": dtype_mismatches,
        "order_changed": order_changed,
        "alert": alert,
    }


def build_schema_alert_payload(
    *,
    project_name: str,
    schema_check: dict[str, object],
) -> dict[str, object]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_name": project_name,
        "alert": bool(schema_check.get("alert", False)),
        "schema": schema_check,
    }


def write_schema_alert_artifacts(
    *,
    output_dir: Path,
    payload: dict[str, object],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "schema_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    alert_path = output_dir / "SCHEMA_ALERT.txt"
    if bool(payload.get("alert")):
        alert_path.write_text(
            f"Schema alert for {payload['project_name']} at {payload['generated_at']}",
            encoding="utf-8",
        )
    elif alert_path.exists():
        alert_path.unlink()

    return {
        "summary_path": str(summary_path),
        "alert_path": str(alert_path),
    }
