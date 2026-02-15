from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from california_housing_template.api import create_app as create_california_app
from california_housing_template.config import california_schema
from california_housing_template.predict import (
    load_assets as load_california_assets,
)
from california_housing_template.predict import (
    predict_one as predict_california,
)
from california_housing_template.train import train_california_housing_model
from ml_core.features.schema import FeatureSchema
from sales_forecasting_regression.api import create_app as create_sales_app
from sales_forecasting_regression.config import sales_schema
from sales_forecasting_regression.predict import load_assets as load_sales_assets
from sales_forecasting_regression.predict import predict_one as predict_sales
from sales_forecasting_regression.train import train_sales_model


@dataclass(frozen=True, slots=True)
class ProjectCase:
    name: str
    train_fn: Callable[..., dict[str, Any]]
    load_assets_fn: Callable[..., tuple[Any, dict[str, Any]]]
    predict_fn: Callable[..., float]
    schema_fn: Callable[[], FeatureSchema]
    create_app_fn: Callable[..., FastAPI]
    model_filename: str
    metadata_filename: str
    onnx_filename: str
    train_kwargs_fn: Callable[[Path], dict[str, Any]]
    sample_overrides: dict[str, Any]


def _project_cases() -> tuple[ProjectCase, ...]:
    return (
        ProjectCase(
            name="sales",
            train_fn=train_sales_model,
            load_assets_fn=load_sales_assets,
            predict_fn=predict_sales,
            schema_fn=sales_schema,
            create_app_fn=create_sales_app,
            model_filename="sales_regressor.joblib",
            metadata_filename="sales_regressor_metadata.json",
            onnx_filename="sales_regressor.onnx",
            train_kwargs_fn=lambda tmp_path: {
                "data_path": tmp_path / "missing_sales.csv",
                "selection_mode": "baseline",
            },
            sample_overrides={
                "PRODUCTLINE": "Classic Cars",
                "COUNTRY": "USA",
                "DEALSIZE": "Medium",
            },
        ),
        ProjectCase(
            name="california",
            train_fn=train_california_housing_model,
            load_assets_fn=load_california_assets,
            predict_fn=predict_california,
            schema_fn=california_schema,
            create_app_fn=create_california_app,
            model_filename="california_model.joblib",
            metadata_filename="california_metadata.json",
            onnx_filename="california_model.onnx",
            train_kwargs_fn=lambda _: {"force_synthetic": True, "selection_mode": "baseline"},
            sample_overrides={},
        ),
    )


def _has_skl2onnx() -> bool:
    try:
        import skl2onnx  # noqa: F401
    except Exception:
        return False
    return True


def _sample_features(schema: FeatureSchema, overrides: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {name: 1.0 for name in schema.numeric_features}
    payload.update({name: "sample" for name in schema.categorical_features})
    payload.update(overrides)
    return payload


@pytest.mark.parametrize("case", _project_cases(), ids=lambda item: item.name)
def test_artifact_contract(case: ProjectCase, tmp_path: Path) -> None:
    artifact_dir = tmp_path / case.name / "models"
    case.train_fn(artifact_dir=artifact_dir, **case.train_kwargs_fn(tmp_path))

    model_path = artifact_dir / case.model_filename
    metadata_path = artifact_dir / case.metadata_filename
    onnx_path = artifact_dir / case.onnx_filename
    schema_path = artifact_dir / "schema.json"

    assert model_path.exists()
    assert metadata_path.exists()
    assert schema_path.exists()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert {"best_model_name", "feature_names", "target_name", "metrics"}.issubset(set(metadata))
    assert metadata["metrics"]["rmse"] > 0

    schema_payload = json.loads(schema_path.read_text(encoding="utf-8"))
    assert {"numeric_features", "categorical_features", "feature_names", "target_name"}.issubset(set(schema_payload))

    if _has_skl2onnx():
        assert onnx_path.exists()
        assert onnx_path.stat().st_size > 0


@pytest.mark.parametrize("case", _project_cases(), ids=lambda item: item.name)
def test_inference_is_stable_and_reproducible(case: ProjectCase, tmp_path: Path) -> None:
    artifact_dir = tmp_path / case.name / "models"
    case.train_fn(artifact_dir=artifact_dir, **case.train_kwargs_fn(tmp_path))

    schema = case.schema_fn()
    features = _sample_features(schema, case.sample_overrides)

    p1 = case.predict_fn(features, artifact_dir=artifact_dir)
    p2 = case.predict_fn(features, artifact_dir=artifact_dir)

    assert np.isfinite(p1)
    assert np.isfinite(p2)
    assert abs(p1) < 1_000_000_000
    assert abs(p2) < 1_000_000_000
    assert p1 == pytest.approx(p2, abs=1e-12)

    model, metadata = case.load_assets_fn(artifact_dir=artifact_dir)
    ordered = {name: features[name] for name in metadata["feature_names"]}
    p3 = float(model.predict(pd.DataFrame([ordered]))[0])
    assert p1 == pytest.approx(p3, abs=1e-12)


@pytest.mark.parametrize("case", _project_cases(), ids=lambda item: item.name)
def test_training_and_inference_schema_are_compatible(case: ProjectCase, tmp_path: Path) -> None:
    artifact_dir = tmp_path / case.name / "models"
    payload = case.train_fn(artifact_dir=artifact_dir, **case.train_kwargs_fn(tmp_path))
    schema = case.schema_fn()

    metadata = json.loads((artifact_dir / case.metadata_filename).read_text(encoding="utf-8"))
    saved_schema = json.loads((artifact_dir / "schema.json").read_text(encoding="utf-8"))

    assert tuple(payload["feature_names"]) == schema.all_features
    assert tuple(metadata["feature_names"]) == schema.all_features
    assert tuple(saved_schema["feature_names"]) == schema.all_features
    assert tuple(saved_schema["numeric_features"]) == schema.numeric_features
    assert tuple(saved_schema["categorical_features"]) == schema.categorical_features
    assert payload["target"] == schema.target_name
    assert metadata["target_name"] == schema.target_name
    assert saved_schema["target_name"] == schema.target_name

    app = case.create_app_fn(artifact_dir=artifact_dir)
    features = _sample_features(schema, case.sample_overrides)

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["feature_count"] == len(schema.all_features)

        prediction = client.post("/predict", json=features)
        assert prediction.status_code == 200
        body = prediction.json()
        assert tuple(body["input_features"].keys()) == schema.all_features
        assert body["target_name"] == schema.target_name
