from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from california_housing_template.config import CaliforniaPaths, california_schema
from california_housing_template.data import CaliforniaHousingLoader
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


def california_candidates() -> list[CandidateModel]:
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
                "max_depth": [None, 20],
                "min_samples_split": [2, 5],
            },
            tuning_cv=3,
            tuning_n_jobs=-1,
        ),
    ]


def train_california_housing_model(
    *,
    artifact_dir: Path | None = None,
    report_path: Path | None = None,
    metrics_path: Path | None = None,
    use_synthetic_if_fetch_fails: bool = True,
    force_synthetic: bool = False,
    selection_mode: str = "both",
    registry_dir: Path | None = None,
    promote_staging: bool = False,
    promote_prod: bool = False,
) -> dict[str, Any]:
    schema = california_schema()
    loader = CaliforniaHousingLoader(random_state=42)
    paths = CaliforniaPaths.default()
    models_root = artifact_dir or paths.models_dir
    default_report = (
        (models_root.parent / "reports" / "california_metrics.json")
        if artifact_dir is not None
        else (paths.project_root / "reports" / "california_metrics.json")
    )
    output_report = report_path or default_report
    output_metrics = metrics_path or (output_report.parent / "california_metrics_summary.json")
    frame = loader.load(
        use_synthetic_if_fetch_fails=use_synthetic_if_fetch_fails,
        force_synthetic=force_synthetic,
    )
    frame = validate_frame(
        frame=frame,
        schema=schema,
        require_target=True,
        context="california_train_pre_split",
        allow_extra_columns=True,
    )

    split_data = split_regression_frame(frame=frame, schema=schema, test_size=0.2, random_state=42)
    trainer = RegressionTrainer(schema=schema, candidates=california_candidates(), primary_metric="rmse")
    result = trainer.fit(split_data, selection_mode=selection_mode)
    shap_artifacts = generate_shap_artifacts(
        pipeline=result.pipeline,
        frame=split_data.x_train,
        schema=schema,
        output_dir=models_root / "explainability",
    )
    registry = ArtifactRegistry(
        root_dir=models_root,
        model_filename="california_model.joblib",
        metadata_filename="california_metadata.json",
        onnx_filename="california_model.onnx",
    )

    metadata = registry.save_training_result(
        result=result,
        project_name="california_housing_template",
        extra_metadata={
            "dataset_rows": int(frame.shape[0]),
            "reference_template": True,
            "uncertainty": _residual_uncertainty(split_data=split_data, pipeline=result.pipeline),
            "conformal_intervals": _conformal_intervals(split_data=split_data, pipeline=result.pipeline, alpha=0.1),
            "shap_explainability": shap_artifacts["summary"],
        },
        export_onnx=True,
        sample_frame=split_data.x_train,
    )

    registry_info: dict[str, Any] | None = None
    if registry_dir is not None:
        registration = register_model_version(
            source_dir=registry.root_dir,
            registry_dir=registry_dir,
            model_name="california_housing_template",
            artifact_filenames=[
                "california_model.joblib",
                "california_metadata.json",
                "california_model.onnx",
                "schema.json",
            ],
            metadata={"best_model": result.best_score.model_name, "metrics": result.best_score.metrics},
        )

        if promote_staging:
            set_alias(
                registry_dir=registry_dir,
                model_name="california_housing_template",
                alias="staging",
                version=registration["version"],
            )

        if promote_prod:
            promote_to_prod(
                registry_dir=registry_dir,
                model_name="california_housing_template",
                version=registration["version"],
            )

        registry_info = {
            "registration": registration,
            "aliases": read_aliases(registry_dir=registry_dir, model_name="california_housing_template"),
            "registry_dir": str(registry_dir),
        }

    payload = {
        "best_model": result.best_score.model_name,
        "best_metrics": result.best_score.metrics,
        "feature_names": list(schema.all_features),
        "target": schema.target_name,
        "leaderboard": leaderboard_as_records(result.leaderboard),
        "selection_mode": selection_mode,
        "artifacts": {
            "model": str(registry.model_path),
            "metadata": str(registry.metadata_path),
            "onnx": str(registry.onnx_path) if registry.onnx_path.exists() else None,
            "schema": str(registry.schema_path),
            "shap_summary": str(models_root / "explainability" / "shap_summary.json"),
            "shap_feature_importance": str(models_root / "explainability" / "shap_feature_importance.csv"),
        },
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
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the California Housing template model.")
    parser.add_argument("--artifact-dir", type=str, default=None)
    parser.add_argument("--report-path", type=str, default=None)
    parser.add_argument("--metrics-path", type=str, default=None)
    parser.add_argument("--strict-fetch", action="store_true")
    parser.add_argument("--force-synthetic", action="store_true")
    parser.add_argument("--selection-mode", type=str, choices=["baseline", "tuned", "both"], default="both")
    parser.add_argument("--registry-dir", type=str, default=None)
    parser.add_argument("--promote-staging", action="store_true")
    parser.add_argument("--promote-prod", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = train_california_housing_model(
        artifact_dir=Path(args.artifact_dir) if args.artifact_dir else None,
        report_path=Path(args.report_path) if args.report_path else None,
        metrics_path=Path(args.metrics_path) if args.metrics_path else None,
        use_synthetic_if_fetch_fails=not args.strict_fetch,
        force_synthetic=bool(args.force_synthetic),
        selection_mode=args.selection_mode,
        registry_dir=Path(args.registry_dir) if args.registry_dir else None,
        promote_staging=bool(args.promote_staging),
        promote_prod=bool(args.promote_prod),
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
