from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from ml_core.data.splitters import split_regression_frame
from ml_core.features.schema import FeatureSchema
from ml_core.modeling.contracts import CandidateModel
from ml_core.modeling.trainer import RegressionTrainer
from ml_core.registry.artifacts import ArtifactRegistry
from ml_core.serving.app_factory import create_regression_app


def _train_and_save(
    *,
    tmp_path: Path,
    candidate: CandidateModel,
    uncertainty: dict[str, float] | None,
    conformal: dict[str, float] | None = None,
) -> tuple[Path, FeatureSchema]:
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({"x": rng.uniform(-4.0, 4.0, size=320)})
    frame["target"] = 3.2 * frame["x"] + rng.normal(0.0, 0.8, size=320)

    schema = FeatureSchema.create(["x"], [], "target")
    split = split_regression_frame(frame, schema=schema, random_state=42)
    trainer = RegressionTrainer(schema=schema, candidates=[candidate], primary_metric="rmse")
    result = trainer.fit(split, selection_mode="baseline")

    artifact_dir = tmp_path / "models"
    registry = ArtifactRegistry(
        root_dir=artifact_dir,
        model_filename="model.joblib",
        metadata_filename="metadata.json",
        onnx_filename="model.onnx",
    )
    extra: dict[str, object] = {}
    if uncertainty is not None:
        extra["uncertainty"] = uncertainty
    if conformal is not None:
        extra["conformal_intervals"] = conformal
    registry.save_training_result(
        result=result,
        project_name="uncertainty_test",
        extra_metadata=extra,
        export_onnx=False,
        sample_frame=split.x_train,
    )
    return artifact_dir, schema


def test_predict_returns_ensemble_quantiles_for_random_forest(tmp_path: Path) -> None:
    candidate = CandidateModel(
        name="RandomForestRegressor",
        estimator_factory=lambda params: RandomForestRegressor(random_state=42, n_jobs=-1, **params),
        parameter_grid=({"n_estimators": 120, "max_depth": 8},),
    )
    artifact_dir, schema = _train_and_save(
        tmp_path=tmp_path,
        candidate=candidate,
        uncertainty={"residual_q10": -1.0, "residual_q90": 1.0},
    )

    app = create_regression_app(
        title="Uncertainty Test",
        description="test",
        schema=schema,
        artifact_dir=artifact_dir,
        model_filename="model.joblib",
        metadata_filename="metadata.json",
    )
    with TestClient(app) as client:
        response = client.post("/predict", json={"x": 1.8})
    assert response.status_code == 200
    body = response.json()
    assert body["interval_method"] == "ensemble_quantiles"
    assert body["p10"] <= body["y_hat"] <= body["p90"]


def test_predict_falls_back_to_residual_quantiles_when_not_ensemble(tmp_path: Path) -> None:
    candidate = CandidateModel(
        name="LinearRegression",
        estimator_factory=lambda _: LinearRegression(),
        parameter_grid=({},),
    )
    artifact_dir, schema = _train_and_save(
        tmp_path=tmp_path,
        candidate=candidate,
        uncertainty={"residual_q10": -2.0, "residual_q90": 3.0},
    )

    app = create_regression_app(
        title="Uncertainty Test",
        description="test",
        schema=schema,
        artifact_dir=artifact_dir,
        model_filename="model.joblib",
        metadata_filename="metadata.json",
    )
    with TestClient(app) as client:
        response = client.post("/predict", json={"x": 2.4})
    assert response.status_code == 200
    body = response.json()
    assert body["interval_method"] == "residual_quantiles"
    assert body["p10"] == pytest.approx(body["y_hat"] - 2.0)
    assert body["p90"] == pytest.approx(body["y_hat"] + 3.0)


def test_predict_uses_conformal_intervals_when_available(tmp_path: Path) -> None:
    candidate = CandidateModel(
        name="LinearRegression",
        estimator_factory=lambda _: LinearRegression(),
        parameter_grid=({},),
    )
    artifact_dir, schema = _train_and_save(
        tmp_path=tmp_path,
        candidate=candidate,
        uncertainty={"residual_q10": -2.0, "residual_q90": 3.0},
        conformal={"method": "split_conformal_abs_residual", "alpha": 0.1, "q_hat": 1.25},
    )

    app = create_regression_app(
        title="Uncertainty Test",
        description="test",
        schema=schema,
        artifact_dir=artifact_dir,
        model_filename="model.joblib",
        metadata_filename="metadata.json",
    )
    with TestClient(app) as client:
        response = client.post("/predict", json={"x": 2.4})
    assert response.status_code == 200
    body = response.json()
    assert body["interval_method"] == "conformal_intervals"
    assert body["p10"] == pytest.approx(body["y_hat"] - 1.25)
    assert body["p90"] == pytest.approx(body["y_hat"] + 1.25)


def test_predict_returns_null_intervals_without_uncertainty_metadata(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "models"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    target = pd.Series([2.0, 4.0, 6.0])
    model = LinearRegression().fit(frame, target)
    metadata = {
        "best_model_name": "LinearRegression",
        "target_name": "target",
        "feature_names": ["x"],
        "metrics": {"rmse": 0.1, "mae": 0.1, "mse": 0.01, "r2": 0.9},
        "extra": {},
    }
    import joblib

    joblib.dump(model, artifact_dir / "model.joblib")
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    app = create_regression_app(
        title="Uncertainty Test",
        description="test",
        schema=FeatureSchema.create(["x"], [], "target"),
        artifact_dir=artifact_dir,
        model_filename="model.joblib",
        metadata_filename="metadata.json",
    )
    with TestClient(app) as client:
        response = client.post("/predict", json={"x": 4.0})
    assert response.status_code == 200
    body = response.json()
    assert body["interval_method"] is None
    assert body["p10"] is None
    assert body["p90"] is None
