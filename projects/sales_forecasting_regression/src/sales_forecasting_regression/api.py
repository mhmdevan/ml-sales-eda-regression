from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from ml_core.serving.app_factory import create_regression_app
from sales_forecasting_regression.config import sales_schema


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def create_app(artifact_dir: Path | None = None) -> FastAPI:
    return create_regression_app(
        title="Sales Forecasting Regression API",
        description="Production contract for sales regression inference.",
        schema=sales_schema(),
        artifact_dir=artifact_dir or (PROJECT_ROOT / "models"),
        model_filename="sales_regressor.joblib",
        metadata_filename="sales_regressor_metadata.json",
        version="1.0.0",
    )


app = create_app()
