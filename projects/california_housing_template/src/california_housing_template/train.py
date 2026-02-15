from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from california_housing_template.config import CaliforniaPaths, california_schema
from california_housing_template.data import CaliforniaHousingLoader
from ml_core.data.splitters import split_regression_frame
from ml_core.modeling.contracts import CandidateModel
from ml_core.modeling.trainer import RegressionTrainer
from ml_core.registry.artifacts import ArtifactRegistry, leaderboard_as_records


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
            parameter_grid=(
                {"n_estimators": 180, "max_depth": None, "min_samples_split": 2},
                {"n_estimators": 260, "max_depth": 20, "min_samples_split": 5},
            ),
        ),
    ]


def train_california_housing_model(
    *,
    artifact_dir: Path | None = None,
    use_synthetic_if_fetch_fails: bool = True,
    force_synthetic: bool = False,
) -> dict[str, Any]:
    schema = california_schema()
    loader = CaliforniaHousingLoader(random_state=42)
    frame = loader.load(
        use_synthetic_if_fetch_fails=use_synthetic_if_fetch_fails,
        force_synthetic=force_synthetic,
    )

    split_data = split_regression_frame(frame=frame, schema=schema, test_size=0.2, random_state=42)
    trainer = RegressionTrainer(schema=schema, candidates=california_candidates(), primary_metric="rmse")
    result = trainer.fit(split_data)

    paths = CaliforniaPaths.default()
    registry = ArtifactRegistry(
        root_dir=artifact_dir or paths.models_dir,
        model_filename="california_model.joblib",
        metadata_filename="california_metadata.json",
        onnx_filename="california_model.onnx",
    )

    metadata = registry.save_training_result(
        result=result,
        project_name="california_housing_template",
        extra_metadata={"dataset_rows": int(frame.shape[0]), "reference_template": True},
        export_onnx=True,
        sample_frame=split_data.x_train,
    )

    return {
        "best_model": result.best_score.model_name,
        "best_metrics": result.best_score.metrics,
        "feature_names": list(schema.all_features),
        "target": schema.target_name,
        "leaderboard": leaderboard_as_records(result.leaderboard),
        "metadata": metadata,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the California Housing template model.")
    parser.add_argument("--artifact-dir", type=str, default=None)
    parser.add_argument("--strict-fetch", action="store_true")
    parser.add_argument("--force-synthetic", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = train_california_housing_model(
        artifact_dir=Path(args.artifact_dir) if args.artifact_dir else None,
        use_synthetic_if_fetch_fails=not args.strict_fetch,
        force_synthetic=bool(args.force_synthetic),
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
