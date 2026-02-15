from __future__ import annotations

from typing import Any

from ml_core.modeling.contracts import TrainingResult


def maybe_log_to_mlflow(
    *,
    result: TrainingResult,
    project_name: str,
    tracking_uri: str,
    experiment_name: str,
    run_params: dict[str, Any],
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
        return run.info.run_id
