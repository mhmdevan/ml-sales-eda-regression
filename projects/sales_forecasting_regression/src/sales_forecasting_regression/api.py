from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI

from ml_core.serving.app_factory import create_regression_app
from sales_forecasting_regression.config import sales_schema


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def create_app(artifact_dir: Path | None = None) -> FastAPI:
    registry_dir_env = os.getenv("MODEL_REGISTRY_DIR")
    shadow_log_env = os.getenv("SHADOW_LOG_PATH")

    return create_regression_app(
        title="Sales Forecasting Regression API",
        description="Production contract for sales regression inference.",
        schema=sales_schema(),
        artifact_dir=artifact_dir or (PROJECT_ROOT / "models"),
        model_filename="sales_regressor.joblib",
        metadata_filename="sales_regressor_metadata.json",
        shadow_model_filename="sales_regressor_candidate.joblib",
        shadow_metadata_filename="sales_regressor_candidate_metadata.json",
        shadow_log_path=Path(shadow_log_env) if shadow_log_env else (PROJECT_ROOT / "reports" / "shadow_predictions.jsonl"),
        registry_dir=Path(registry_dir_env) if registry_dir_env else None,
        registry_model_name="sales_forecasting_regression" if registry_dir_env else None,
        primary_alias=os.getenv("MODEL_PRIMARY_ALIAS", "prod"),
        shadow_alias=os.getenv("MODEL_SHADOW_ALIAS", "staging"),
        version="1.0.0",
    )


app = create_app()
