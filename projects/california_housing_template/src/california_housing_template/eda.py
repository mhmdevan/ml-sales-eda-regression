from __future__ import annotations

from pathlib import Path
from typing import Any

from california_housing_template.config import CALIFORNIA_FEATURES, CALIFORNIA_TARGET
from california_housing_template.data import CaliforniaHousingLoader
from ml_core.monitoring.eda import build_tabular_eda_payload, write_eda_artifacts


def generate_california_eda_report(
    *,
    output_dir: Path,
    report_version: str,
    force_synthetic: bool = False,
    use_synthetic_if_fetch_fails: bool = True,
) -> dict[str, Any]:
    frame = CaliforniaHousingLoader(random_state=42).load(
        force_synthetic=force_synthetic,
        use_synthetic_if_fetch_fails=use_synthetic_if_fetch_fails,
    )
    payload = build_tabular_eda_payload(
        frame=frame,
        project_name="california_housing_template",
        report_version=report_version,
        numeric_features=list(CALIFORNIA_FEATURES),
        categorical_features=[],
        target_name=CALIFORNIA_TARGET,
    )
    paths = write_eda_artifacts(
        frame=frame,
        numeric_features=list(CALIFORNIA_FEATURES),
        target_name=CALIFORNIA_TARGET,
        output_dir=output_dir,
        payload=payload,
    )
    return {
        "report": payload,
        "artifacts": paths,
    }
