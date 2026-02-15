import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from ml_core.features.schema import FeatureSchema
from ml_core.serving.app_factory import create_regression_app


def _train_linear_model(multiplier: float) -> Pipeline:
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    target = frame["x"] * multiplier
    model = Pipeline(steps=[("regressor", LinearRegression())])
    model.fit(frame, target)
    return model


def test_shadow_model_logs_predictions(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "models"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    primary_model = _train_linear_model(2.0)
    shadow_model = _train_linear_model(2.2)

    joblib.dump(primary_model, artifact_dir / "model.joblib")
    joblib.dump(shadow_model, artifact_dir / "shadow.joblib")

    metadata = {
        "best_model_name": "primary",
        "target_name": "target",
        "feature_names": ["x"],
    }
    shadow_metadata = {
        "best_model_name": "shadow",
        "target_name": "target",
        "feature_names": ["x"],
    }

    (artifact_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (artifact_dir / "shadow_metadata.json").write_text(json.dumps(shadow_metadata), encoding="utf-8")

    shadow_log_path = tmp_path / "shadow_logs" / "predictions.jsonl"

    app = create_regression_app(
        title="Shadow Test",
        description="shadow",
        schema=FeatureSchema.create(["x"], [], "target"),
        artifact_dir=artifact_dir,
        model_filename="model.joblib",
        metadata_filename="metadata.json",
        shadow_model_filename="shadow.joblib",
        shadow_metadata_filename="shadow_metadata.json",
        shadow_log_path=shadow_log_path,
    )

    with TestClient(app) as client:
        response = client.post("/predict", json={"x": 5.0})
        assert response.status_code == 200

    assert shadow_log_path.exists()
    row = json.loads(shadow_log_path.read_text(encoding="utf-8").strip())
    assert "primary_prediction" in row
    assert "shadow_prediction" in row
    assert row["primary_model_name"] == "primary"
    assert row["shadow_model_name"] == "shadow"
