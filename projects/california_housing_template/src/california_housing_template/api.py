from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from california_housing_template.config import california_schema
from ml_core.serving.app_factory import create_regression_app


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def create_app(artifact_dir: Path | None = None) -> FastAPI:
    return create_regression_app(
        title="California Housing Template API",
        description="Reference implementation of the shared production inference contract.",
        schema=california_schema(),
        artifact_dir=artifact_dir or (PROJECT_ROOT / "models"),
        model_filename="california_model.joblib",
        metadata_filename="california_metadata.json",
        version="1.0.0",
    )


app = create_app()
