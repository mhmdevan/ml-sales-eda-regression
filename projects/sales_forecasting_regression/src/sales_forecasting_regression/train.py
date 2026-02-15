from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from ml_core.data.splitters import split_regression_frame
from ml_core.modeling.contracts import CandidateModel
from ml_core.modeling.trainer import RegressionTrainer
from ml_core.registry.artifacts import ArtifactRegistry, leaderboard_as_records
from sales_forecasting_regression.config import SalesPaths, sales_schema
from sales_forecasting_regression.data import SalesDatasetLoader
from sales_forecasting_regression.tracking import maybe_log_to_mlflow


def sales_candidates() -> list[CandidateModel]:
    return [
        CandidateModel(
            name="LinearRegression",
            estimator_factory=lambda _: LinearRegression(),
            parameter_grid=({},),
        ),
        CandidateModel(
            name="RandomForestRegressor",
            estimator_factory=lambda params: RandomForestRegressor(random_state=42, n_jobs=-1, **params),
            parameter_grid=(
                {"n_estimators": 180, "max_depth": None},
                {"n_estimators": 260, "max_depth": 12},
            ),
        ),
        CandidateModel(
            name="GradientBoostingRegressor",
            estimator_factory=lambda params: GradientBoostingRegressor(random_state=42, **params),
            parameter_grid=(
                {"n_estimators": 120, "learning_rate": 0.08, "max_depth": 3},
                {"n_estimators": 180, "learning_rate": 0.05, "max_depth": 3},
            ),
        ),
    ]


def train_sales_model(
    *,
    data_path: Path | None = None,
    artifact_dir: Path | None = None,
    report_path: Path | None = None,
    enable_mlflow: bool = False,
    mlflow_tracking_uri: str = "file:mlruns",
    mlflow_experiment_name: str = "sales_forecasting_regression",
) -> dict[str, Any]:
    paths = SalesPaths.default()
    schema = sales_schema()

    loader = SalesDatasetLoader(random_state=42)
    frame = loader.load(csv_path=data_path, use_synthetic_if_missing=True)

    split_data = split_regression_frame(frame, schema=schema, test_size=0.2, random_state=42)

    trainer = RegressionTrainer(schema=schema, candidates=sales_candidates(), primary_metric="rmse")
    result = trainer.fit(split_data)

    registry = ArtifactRegistry(
        root_dir=artifact_dir or paths.models_dir,
        model_filename="sales_regressor.joblib",
        metadata_filename="sales_regressor_metadata.json",
        onnx_filename="sales_regressor.onnx",
    )

    metadata = registry.save_training_result(
        result=result,
        project_name="sales_forecasting_regression",
        extra_metadata={"dataset_rows": int(frame.shape[0])},
        export_onnx=True,
        sample_frame=split_data.x_train,
    )

    run_id = None
    if enable_mlflow:
        run_id = maybe_log_to_mlflow(
            result=result,
            project_name="sales_forecasting_regression",
            tracking_uri=mlflow_tracking_uri,
            experiment_name=mlflow_experiment_name,
            run_params={
                "dataset_rows": int(frame.shape[0]),
                "test_size": 0.2,
                "candidate_count": len(sales_candidates()),
            },
        )

    payload = {
        "best_model": result.best_score.model_name,
        "best_metrics": result.best_score.metrics,
        "target": schema.target_name,
        "feature_names": list(schema.all_features),
        "leaderboard": leaderboard_as_records(result.leaderboard),
        "artifacts": {
            "model": str(registry.model_path),
            "metadata": str(registry.metadata_path),
            "onnx": str(registry.onnx_path) if registry.onnx_path.exists() else None,
        },
        "mlflow_run_id": run_id,
        "metadata": metadata,
    }

    output_report = report_path or (paths.reports_dir / "sales_metrics.json")
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train sales forecasting regression models.")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--artifact-dir", type=str, default=None)
    parser.add_argument("--report-path", type=str, default=None)
    parser.add_argument("--enable-mlflow", action="store_true")
    parser.add_argument("--mlflow-tracking-uri", type=str, default="file:mlruns")
    parser.add_argument("--mlflow-experiment-name", type=str, default="sales_forecasting_regression")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = train_sales_model(
        data_path=Path(args.data_path) if args.data_path else None,
        artifact_dir=Path(args.artifact_dir) if args.artifact_dir else None,
        report_path=Path(args.report_path) if args.report_path else None,
        enable_mlflow=bool(args.enable_mlflow),
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_experiment_name=args.mlflow_experiment_name,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
