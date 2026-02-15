from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ml_core.modeling.contracts import TrainingResult


def maybe_log_to_mlflow(
    *,
    result: TrainingResult,
    project_name: str,
    tracking_uri: str,
    experiment_name: str,
    run_params: dict[str, Any],
    artifact_paths: list[Path] | None = None,
    signature_frame: pd.DataFrame | None = None,
) -> str | None:
    try:
        import mlflow
    except Exception:
        return None

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"{project_name}-{result.best_score.model_name}") as run:
        mlflow.log_params(run_params)
        mlflow.log_metrics(result.best_score.metrics)
        mlflow.log_param("best_model_name", result.best_score.model_name)
        mlflow.log_param("target_name", result.schema.target_name)
        mlflow.log_param("feature_count", len(result.schema.all_features))

        signature = None
        input_example = None
        if signature_frame is not None and not signature_frame.empty:
            try:
                from mlflow.models.signature import infer_signature

                predictions = result.pipeline.predict(signature_frame)
                signature = infer_signature(signature_frame, predictions)
                input_example = signature_frame.head(1)
            except Exception:
                signature = None
                input_example = None

        try:
            if hasattr(mlflow, "sklearn") and hasattr(mlflow.sklearn, "log_model"):
                mlflow.sklearn.log_model(
                    sk_model=result.pipeline,
                    artifact_path="model",
                    signature=signature,
                    input_example=input_example,
                )
        except Exception:
            pass

        for path in artifact_paths or []:
            if not path.exists():
                continue
            try:
                if path.is_dir():
                    mlflow.log_artifacts(str(path), artifact_path=f"artifacts/{path.name}")
                else:
                    mlflow.log_artifact(str(path), artifact_path="artifacts")
            except Exception:
                continue
        return run.info.run_id
