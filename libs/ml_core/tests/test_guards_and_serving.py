from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from ml_core.data.guards import ensure_no_leakage, ensure_required_columns
from ml_core.features.schema import FeatureSchema
from ml_core.serving.app_factory import create_regression_app


def test_ensure_required_columns_raises_for_missing() -> None:
    frame = pd.DataFrame({"x": [1]})
    with pytest.raises(ValueError):
        ensure_required_columns(frame, ["x", "y"])


def test_ensure_no_leakage_raises_when_target_in_features() -> None:
    with pytest.raises(ValueError):
        ensure_no_leakage(["x1", "target"], "target")


def test_create_regression_app_returns_503_without_artifact(tmp_path: Path) -> None:
    schema = FeatureSchema.create(["x1", "x2"], ["cat"], "target")
    app = create_regression_app(
        title="Test App",
        description="test",
        schema=schema,
        artifact_dir=tmp_path / "missing_models",
    )

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "model_not_loaded"

        predict = client.post(
            "/predict",
            json={"x1": 1.0, "x2": 2.0, "cat": "a"},
        )
        assert predict.status_code == 503
