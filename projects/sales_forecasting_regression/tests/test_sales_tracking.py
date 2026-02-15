import builtins
import sys
import types
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from ml_core.features.schema import FeatureSchema
from ml_core.modeling.contracts import ModelScore, TrainingResult
from sales_forecasting_regression.tracking import maybe_log_to_mlflow


class FakeRunContext:
    def __init__(self, run_id: str):
        self.info = types.SimpleNamespace(run_id=run_id)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeMlflow:
    def __init__(self) -> None:
        self.tracking_uri = None
        self.experiment = None
        self.logged_params = {}
        self.logged_metrics = {}
        self.run_name = None
        self.logged_artifacts = []
        self.logged_models = []
        self.sklearn = types.SimpleNamespace(log_model=self._log_model)

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uri = uri

    def set_experiment(self, name: str) -> None:
        self.experiment = name

    def start_run(self, run_name: str):
        self.run_name = run_name
        return FakeRunContext("fake-run-id")

    def log_params(self, params):
        self.logged_params.update(params)

    def log_metrics(self, metrics):
        self.logged_metrics.update(metrics)

    def log_param(self, key: str, value):
        self.logged_params[key] = value

    def log_artifact(self, local_path: str, artifact_path: str | None = None):
        self.logged_artifacts.append((local_path, artifact_path))

    def log_artifacts(self, local_dir: str, artifact_path: str | None = None):
        self.logged_artifacts.append((local_dir, artifact_path))

    def _log_model(self, sk_model, artifact_path: str, signature=None, input_example=None):
        self.logged_models.append(
            {
                "artifact_path": artifact_path,
                "signature": signature,
                "has_input_example": input_example is not None,
            }
        )


def build_training_result() -> TrainingResult:
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]})
    model = Pipeline(steps=[("regressor", LinearRegression())])
    model.fit(frame[["x"]], frame["y"])
    schema = FeatureSchema.create(["x"], [], "y")
    score = ModelScore(
        model_name="LinearRegression",
        parameters={},
        metrics={"mse": 0.0, "rmse": 0.0, "mae": 0.0, "r2": 1.0},
    )
    return TrainingResult(
        pipeline=model,
        best_score=score,
        leaderboard=(score,),
        schema=schema,
    )


def test_maybe_log_to_mlflow_returns_none_if_unavailable(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "mlflow", raising=False)
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "mlflow":
            raise ImportError("mlflow")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    run_id = maybe_log_to_mlflow(
        result=build_training_result(),
        project_name="sales",
        tracking_uri="file:mlruns",
        experiment_name="exp",
        run_params={"rows": 3},
    )

    assert run_id is None


def test_maybe_log_to_mlflow_logs_with_fake_module(monkeypatch, tmp_path: Path) -> None:
    fake_mlflow = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    artifact_file = tmp_path / "tmp_mlflow_artifact.txt"
    artifact_file.write_text("ok", encoding="utf-8")

    run_id = maybe_log_to_mlflow(
        result=build_training_result(),
        project_name="sales",
        tracking_uri="file:mlruns",
        experiment_name="exp",
        run_params={"rows": 3},
        artifact_paths=[artifact_file],
        signature_frame=pd.DataFrame({"x": [1.0, 2.0]}),
    )

    assert run_id == "fake-run-id"
    assert fake_mlflow.tracking_uri == "file:mlruns"
    assert fake_mlflow.experiment == "exp"
    assert fake_mlflow.logged_params["rows"] == 3
    assert fake_mlflow.logged_params["best_model_name"] == "LinearRegression"
    assert "rmse" in fake_mlflow.logged_metrics
    assert len(fake_mlflow.logged_artifacts) >= 1
    assert len(fake_mlflow.logged_models) == 1
