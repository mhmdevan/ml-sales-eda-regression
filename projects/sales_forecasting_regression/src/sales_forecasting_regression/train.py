from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from ml_core.data.splitters import split_regression_frame
from ml_core.modeling.contracts import CandidateModel
from ml_core.modeling.trainer import RegressionTrainer
from ml_core.monitoring.conformal import fit_split_conformal_interval
from ml_core.monitoring.explainability import generate_shap_artifacts
from ml_core.registry.artifacts import ArtifactRegistry, leaderboard_as_records
from ml_core.registry.versioning import (
    promote_to_prod,
    read_aliases,
    register_model_version,
    set_alias,
)
from ml_core.validation.pandera_gate import validate_frame
from sales_forecasting_regression.config import SalesPaths, sales_schema
from sales_forecasting_regression.data import SalesDatasetLoader
from sales_forecasting_regression.time_series import generate_segment_time_series_report
from sales_forecasting_regression.tracking import maybe_log_to_mlflow


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_repository_root(),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _to_json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        return _to_json_safe(value.item())
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, list):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if pd.isna(value):
        return None
    return str(value)


def _dataset_fingerprint(frame: pd.DataFrame, columns: tuple[str, ...], target: str) -> str:
    selected = frame.loc[:, list(columns) + [target]]
    row_hash = pd.util.hash_pandas_object(selected, index=False).to_numpy()
    digest = hashlib.sha256()
    digest.update(",".join(list(columns) + [target]).encode("utf-8"))
    digest.update(str(selected.shape[0]).encode("utf-8"))
    digest.update(row_hash.tobytes())
    return digest.hexdigest()


def _inference_input_example(frame: pd.DataFrame, feature_names: tuple[str, ...]) -> dict[str, Any]:
    row = frame.loc[:, list(feature_names)].iloc[0]
    return {name: _to_json_safe(row[name]) for name in feature_names}


def _residual_uncertainty(split_data: Any, pipeline: Any) -> dict[str, Any]:
    residuals = split_data.y_train.to_numpy() - pipeline.predict(split_data.x_train)
    q10, q90 = np.quantile(residuals, [0.1, 0.9])
    return {
        "method": "residual_quantiles",
        "residual_q10": float(q10),
        "residual_q90": float(q90),
        "residual_count": int(residuals.shape[0]),
    }


def _conformal_intervals(split_data: Any, pipeline: Any, alpha: float = 0.1) -> dict[str, Any]:
    calibration_pred = pipeline.predict(split_data.x_test)
    return fit_split_conformal_interval(
        y_true=split_data.y_test.to_numpy(),
        y_pred=np.asarray(calibration_pred, dtype=float),
        alpha=alpha,
    )


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
            parameter_grid=({"n_estimators": 180, "max_depth": None, "min_samples_split": 2},),
            tuning_parameter_grid={
                "n_estimators": [180, 260],
                "max_depth": [None, 12],
                "min_samples_split": [2, 5],
            },
            tuning_cv=3,
            tuning_n_jobs=-1,
        ),
        CandidateModel(
            name="GradientBoostingRegressor",
            estimator_factory=lambda params: GradientBoostingRegressor(random_state=42, **params),
            parameter_grid=({"n_estimators": 120, "learning_rate": 0.08, "max_depth": 3},),
            tuning_parameter_grid={
                "n_estimators": [120, 180],
                "learning_rate": [0.05, 0.08],
                "max_depth": [3],
            },
            tuning_cv=3,
            tuning_n_jobs=-1,
        ),
    ]


def train_sales_model(
    *,
    data_path: Path | None = None,
    artifact_dir: Path | None = None,
    report_path: Path | None = None,
    metrics_path: Path | None = None,
    enable_mlflow: bool = False,
    mlflow_tracking_uri: str = "file:mlruns",
    mlflow_experiment_name: str = "sales_forecasting_regression",
    selection_mode: str = "both",
    registry_dir: Path | None = None,
    promote_staging: bool = False,
    promote_prod: bool = False,
    time_series_report_path: Path | None = None,
    time_series_product_line: str = "Classic Cars",
    time_series_country: str = "USA",
    disable_time_series: bool = False,
) -> dict[str, Any]:
    training_started_at = datetime.now(timezone.utc)
    paths = SalesPaths.default()
    schema = sales_schema()
    models_root = artifact_dir or paths.models_dir
    default_report = (
        (models_root.parent / "reports" / "sales_metrics.json")
        if artifact_dir is not None
        else (paths.reports_dir / "sales_metrics.json")
    )
    output_report = report_path or default_report
    output_metrics = metrics_path or (output_report.parent / "sales_metrics_summary.json")
    time_series_output = time_series_report_path or (output_report.parent / "time_series_risk.json")

    loader = SalesDatasetLoader(random_state=42)
    frame = loader.load(csv_path=data_path, use_synthetic_if_missing=True)
    frame = validate_frame(
        frame=frame,
        schema=schema,
        require_target=True,
        context="sales_train_pre_split",
        allow_extra_columns=True,
    )

    split_data = split_regression_frame(frame, schema=schema, test_size=0.2, random_state=42)

    trainer = RegressionTrainer(schema=schema, candidates=sales_candidates(), primary_metric="rmse")
    result = trainer.fit(split_data, selection_mode=selection_mode)
    training_finished_at = datetime.now(timezone.utc)
    shap_artifacts = generate_shap_artifacts(
        pipeline=result.pipeline,
        frame=split_data.x_train,
        schema=schema,
        output_dir=models_root / "explainability",
    )

    time_series_report: dict[str, Any] | None = None
    if not disable_time_series:
        time_series_report = generate_segment_time_series_report(
            frame=frame,
            output_path=time_series_output,
            product_line=time_series_product_line,
            country=time_series_country,
        )

    feature_dtypes = {feature: str(frame[feature].dtype) for feature in schema.all_features}
    metric_summary = {
        "primary_metric": "rmse",
        "best_model": result.best_score.model_name,
        "best_value": float(result.best_score.metrics["rmse"]),
        "leaderboard": [
            {
                "model_name": score.model_name,
                "rmse": float(score.metrics["rmse"]),
                "mae": float(score.metrics["mae"]),
                "r2": float(score.metrics["r2"]),
            }
            for score in result.leaderboard
        ],
    }
    rich_metadata = {
        "dataset_rows": int(frame.shape[0]),
        "dataset_fingerprint": {
            "algorithm": "sha256",
            "value": _dataset_fingerprint(frame=frame, columns=schema.all_features, target=schema.target_name),
        },
        "training_context": {
            "selection_mode": selection_mode,
            "started_at": training_started_at.isoformat(),
            "finished_at": training_finished_at.isoformat(),
            "duration_seconds": round((training_finished_at - training_started_at).total_seconds(), 6),
            "git_commit_hash": _git_commit_hash(),
        },
        "feature_contract": {
            "feature_names": list(schema.all_features),
            "target_name": schema.target_name,
            "dtypes": feature_dtypes,
            "target_dtype": str(frame[schema.target_name].dtype),
        },
        "metric_summary": metric_summary,
        "model_hyperparameters": _to_json_safe(result.best_score.parameters),
        "inference_input_example": _inference_input_example(frame=split_data.x_train, feature_names=schema.all_features),
        "uncertainty": _residual_uncertainty(split_data=split_data, pipeline=result.pipeline),
        "conformal_intervals": _conformal_intervals(split_data=split_data, pipeline=result.pipeline, alpha=0.1),
        "shap_explainability": shap_artifacts["summary"],
        "time_series": (
            {
                "segment": time_series_report["segment"],
                "best_model": time_series_report["selection"]["best_model"],
                "risk_method": time_series_report["risk"]["method"],
                "report_path": str(time_series_output),
            }
            if time_series_report is not None
            else None
        ),
    }

    registry = ArtifactRegistry(
        root_dir=models_root,
        model_filename="sales_regressor.joblib",
        metadata_filename="sales_regressor_metadata.json",
        onnx_filename="sales_regressor.onnx",
    )

    metadata = registry.save_training_result(
        result=result,
        project_name="sales_forecasting_regression",
        extra_metadata=rich_metadata,
        export_onnx=True,
        sample_frame=split_data.x_train,
    )

    registry_info: dict[str, Any] | None = None
    if registry_dir is not None:
        registration = register_model_version(
            source_dir=registry.root_dir,
            registry_dir=registry_dir,
            model_name="sales_forecasting_regression",
            artifact_filenames=[
                "sales_regressor.joblib",
                "sales_regressor_metadata.json",
                "sales_regressor.onnx",
                "schema.json",
            ],
            metadata={"best_model": result.best_score.model_name, "metrics": result.best_score.metrics},
        )

        if promote_staging:
            set_alias(
                registry_dir=registry_dir,
                model_name="sales_forecasting_regression",
                alias="staging",
                version=registration["version"],
            )

        if promote_prod:
            promote_to_prod(
                registry_dir=registry_dir,
                model_name="sales_forecasting_regression",
                version=registration["version"],
            )

        registry_info = {
            "registration": registration,
            "aliases": read_aliases(registry_dir=registry_dir, model_name="sales_forecasting_regression"),
            "registry_dir": str(registry_dir),
        }

    payload = {
        "best_model": result.best_score.model_name,
        "best_metrics": result.best_score.metrics,
        "target": schema.target_name,
        "feature_names": list(schema.all_features),
        "leaderboard": leaderboard_as_records(result.leaderboard),
        "selection_mode": selection_mode,
        "artifacts": {
            "model": str(registry.model_path),
            "metadata": str(registry.metadata_path),
            "onnx": str(registry.onnx_path) if registry.onnx_path.exists() else None,
            "schema": str(registry.schema_path),
            "shap_summary": str(models_root / "explainability" / "shap_summary.json"),
            "shap_feature_importance": str(models_root / "explainability" / "shap_feature_importance.csv"),
            "time_series_risk_report": str(time_series_output) if time_series_report is not None else None,
        },
        "mlflow_run_id": None,
        "metadata": metadata,
        "registry": registry_info,
    }

    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    metrics_payload = {
        "best_model": result.best_score.model_name,
        "selection_mode": selection_mode,
        "rmse": float(result.best_score.metrics["rmse"]),
        "mae": float(result.best_score.metrics["mae"]),
        "r2": float(result.best_score.metrics["r2"]),
    }
    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    output_metrics.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    run_id = None
    if enable_mlflow:
        artifact_paths: list[Path] = [
            registry.model_path,
            registry.metadata_path,
            registry.schema_path,
            output_report,
            output_metrics,
            models_root / "explainability" / "shap_summary.json",
        ]
        if registry.onnx_path.exists():
            artifact_paths.append(registry.onnx_path)
        shap_csv = models_root / "explainability" / "shap_feature_importance.csv"
        if shap_csv.exists():
            artifact_paths.append(shap_csv)
        if time_series_report is not None and time_series_output.exists():
            artifact_paths.append(time_series_output)

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
            artifact_paths=artifact_paths,
            signature_frame=split_data.x_test.head(50),
        )
        payload["mlflow_run_id"] = run_id
        output_report.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train sales forecasting regression models.")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--artifact-dir", type=str, default=None)
    parser.add_argument("--report-path", type=str, default=None)
    parser.add_argument("--metrics-path", type=str, default=None)
    parser.add_argument("--enable-mlflow", action="store_true")
    parser.add_argument("--mlflow-tracking-uri", type=str, default="file:mlruns")
    parser.add_argument("--mlflow-experiment-name", type=str, default="sales_forecasting_regression")
    parser.add_argument("--selection-mode", type=str, choices=["baseline", "tuned", "both"], default="both")
    parser.add_argument("--registry-dir", type=str, default=None)
    parser.add_argument("--promote-staging", action="store_true")
    parser.add_argument("--promote-prod", action="store_true")
    parser.add_argument("--time-series-report-path", type=str, default=None)
    parser.add_argument("--time-series-product-line", type=str, default="Classic Cars")
    parser.add_argument("--time-series-country", type=str, default="USA")
    parser.add_argument("--disable-time-series", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = train_sales_model(
        data_path=Path(args.data_path) if args.data_path else None,
        artifact_dir=Path(args.artifact_dir) if args.artifact_dir else None,
        report_path=Path(args.report_path) if args.report_path else None,
        metrics_path=Path(args.metrics_path) if args.metrics_path else None,
        enable_mlflow=bool(args.enable_mlflow),
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_experiment_name=args.mlflow_experiment_name,
        selection_mode=args.selection_mode,
        registry_dir=Path(args.registry_dir) if args.registry_dir else None,
        promote_staging=bool(args.promote_staging),
        promote_prod=bool(args.promote_prod),
        time_series_report_path=Path(args.time_series_report_path) if args.time_series_report_path else None,
        time_series_product_line=args.time_series_product_line,
        time_series_country=args.time_series_country,
        disable_time_series=bool(args.disable_time_series),
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
