from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_CORE_SRC = REPO_ROOT / "libs" / "ml_core" / "src"
SALES_SRC = REPO_ROOT / "projects" / "sales_forecasting_regression" / "src"
CALIFORNIA_SRC = REPO_ROOT / "projects" / "california_housing_template" / "src"

for source in [ML_CORE_SRC, SALES_SRC, CALIFORNIA_SRC]:
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from california_housing_template.config import CALIFORNIA_FEATURES, CALIFORNIA_TARGET
from california_housing_template.data import CaliforniaHousingLoader
from ml_core.monitoring.distributions import build_feature_distribution_log
from ml_core.monitoring.drift_report import (
    build_drift_alert_payload,
    compute_numeric_drift_summary,
    write_alert_artifacts,
)
from ml_core.monitoring.schema_alert import (
    build_schema_alert_payload,
    detect_schema_changes,
    write_schema_alert_artifacts,
)
from sales_forecasting_regression.config import SALES_CATEGORICAL_FEATURES, SALES_NUMERIC_FEATURES, SALES_TARGET
from sales_forecasting_regression.data import SalesDatasetLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate daily drift report with Evidently and alert artifacts.")
    parser.add_argument("--project", choices=["sales", "california"], required=True)
    parser.add_argument("--reference-csv", type=str, default=None)
    parser.add_argument("--current-csv", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="reports/monitoring")
    parser.add_argument("--psi-threshold", type=float, default=0.2)
    parser.add_argument("--alert-ratio-threshold", type=float, default=0.3)
    parser.add_argument("--run-label", type=str, default=None)
    parser.add_argument("--fail-on-alert", action="store_true")
    return parser.parse_args()


def _load_project_frames(
    *,
    project: str,
    reference_csv: Path | None,
    current_csv: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], str]:
    if project == "sales":
        loader = SalesDatasetLoader(random_state=42)
        reference = (
            pd.read_csv(reference_csv)
            if reference_csv is not None and reference_csv.exists()
            else loader.synthetic_frame(2500)
        )
        current = (
            pd.read_csv(current_csv)
            if current_csv is not None and current_csv.exists()
            else loader.synthetic_frame(2500)
        )
        return (
            reference,
            current,
            list(SALES_NUMERIC_FEATURES),
            list(SALES_NUMERIC_FEATURES) + list(SALES_CATEGORICAL_FEATURES),
            SALES_TARGET,
        )

    loader = CaliforniaHousingLoader(random_state=42)
    reference = (
        pd.read_csv(reference_csv)
        if reference_csv is not None and reference_csv.exists()
        else loader.load(force_synthetic=True)
    )
    current = (
        pd.read_csv(current_csv)
        if current_csv is not None and current_csv.exists()
        else loader.load(force_synthetic=True)
    )
    return (
        reference,
        current,
        list(CALIFORNIA_FEATURES),
        list(CALIFORNIA_FEATURES),
        CALIFORNIA_TARGET,
    )


def _save_evidently_report(
    *,
    reference_frame: pd.DataFrame,
    current_frame: pd.DataFrame,
    features: list[str],
    output_dir: Path,
    project: str,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report
    except Exception:
        from evidently import Report
        from evidently.presets import DataDriftPreset

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(
        reference_data=reference_frame.loc[:, features],
        current_data=current_frame.loc[:, features],
    )
    serializable = snapshot if snapshot is not None else report

    html_path = output_dir / f"{project}_evidently_report.html"
    json_path = output_dir / f"{project}_evidently_report.json"

    if hasattr(serializable, "save_html"):
        serializable.save_html(str(html_path))
    elif hasattr(serializable, "save"):
        serializable.save(str(html_path))
    else:
        raise RuntimeError("Unsupported Evidently report API: no HTML save method found.")

    if hasattr(serializable, "save_json"):
        serializable.save_json(str(json_path))
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    elif hasattr(serializable, "as_dict"):
        payload = serializable.as_dict()
    elif hasattr(serializable, "dict"):
        payload = serializable.dict()
    elif hasattr(serializable, "json"):
        payload = json.loads(serializable.json())
    else:
        payload = {"warning": "Could not serialize Evidently report to JSON payload."}

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return {
        "html_path": str(html_path),
        "json_path": str(json_path),
    }


def main() -> None:
    args = parse_args()

    reference, current, drift_features, schema_features, target_name = _load_project_frames(
        project=args.project,
        reference_csv=Path(args.reference_csv) if args.reference_csv else None,
        current_csv=Path(args.current_csv) if args.current_csv else None,
    )

    drift_summary = compute_numeric_drift_summary(
        reference_frame=reference,
        current_frame=current,
        numeric_features=drift_features,
        psi_threshold=args.psi_threshold,
    )

    payload = build_drift_alert_payload(
        project_name=args.project,
        drift_summary=drift_summary,
        alert_ratio_threshold=args.alert_ratio_threshold,
    )

    distribution_log = build_feature_distribution_log(
        reference_frame=reference,
        current_frame=current,
        feature_names=schema_features,
    )
    expected_dtypes = {feature: str(reference[feature].dtype) for feature in schema_features if feature in reference.columns}
    current_features_frame = current.drop(columns=[target_name], errors="ignore")
    schema_check = detect_schema_changes(
        expected_features=schema_features,
        expected_dtypes=expected_dtypes,
        current_frame=current_features_frame,
    )
    schema_payload = build_schema_alert_payload(project_name=args.project, schema_check=schema_check)

    run_label = args.run_label or str(date.today())
    run_dir = Path(args.output_dir) / args.project / run_label
    artifact_paths = write_alert_artifacts(output_dir=run_dir, payload=payload)
    schema_artifacts = write_schema_alert_artifacts(output_dir=run_dir, payload=schema_payload)
    evidently_paths = _save_evidently_report(
        reference_frame=reference,
        current_frame=current,
        features=drift_features,
        output_dir=run_dir,
        project=args.project,
    )

    output = {
        "payload": payload,
        "schema": schema_payload,
        "feature_distributions": distribution_log,
        "artifacts": artifact_paths,
        "schema_artifacts": schema_artifacts,
        "evidently": evidently_paths,
    }

    print(json.dumps(output, indent=2))

    if (payload["alert"] or schema_payload["alert"]) and args.fail_on_alert:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
