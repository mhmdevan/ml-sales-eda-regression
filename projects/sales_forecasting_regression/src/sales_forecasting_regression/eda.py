from __future__ import annotations

from pathlib import Path
from typing import Any

from ml_core.monitoring.eda import build_tabular_eda_payload, write_eda_artifacts
from sales_forecasting_regression.config import SALES_CATEGORICAL_FEATURES, SALES_NUMERIC_FEATURES, SALES_TARGET
from sales_forecasting_regression.data import SalesDatasetLoader


def generate_sales_eda_report(
    *,
    output_dir: Path,
    report_version: str,
    csv_path: Path | None = None,
    use_synthetic_if_missing: bool = True,
) -> dict[str, Any]:
    frame = SalesDatasetLoader(random_state=42).load(
        csv_path=csv_path,
        use_synthetic_if_missing=use_synthetic_if_missing,
    )
    payload = build_tabular_eda_payload(
        frame=frame,
        project_name="sales_forecasting_regression",
        report_version=report_version,
        numeric_features=list(SALES_NUMERIC_FEATURES),
        categorical_features=list(SALES_CATEGORICAL_FEATURES),
        target_name=SALES_TARGET,
    )
    paths = write_eda_artifacts(
        frame=frame,
        numeric_features=list(SALES_NUMERIC_FEATURES),
        target_name=SALES_TARGET,
        output_dir=output_dir,
        payload=payload,
    )
    return {
        "report": payload,
        "artifacts": paths,
    }
