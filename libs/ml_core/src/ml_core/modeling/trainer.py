from __future__ import annotations

from sklearn.pipeline import Pipeline

from ml_core.data.contracts import SplitDataset
from ml_core.features.preprocessor import build_preprocessor
from ml_core.features.schema import FeatureSchema
from ml_core.modeling.contracts import CandidateModel, ModelScore, TrainingResult
from ml_core.modeling.metrics import compute_regression_metrics


class RegressionTrainer:
    def __init__(
        self,
        schema: FeatureSchema,
        candidates: list[CandidateModel],
        primary_metric: str = "rmse",
    ) -> None:
        if not candidates:
            raise ValueError("At least one candidate model is required.")
        self.schema = schema
        self.candidates = candidates
        self.primary_metric = primary_metric

    def fit(self, split_data: SplitDataset) -> TrainingResult:
        self.schema.ensure_valid()
        scores: list[ModelScore] = []
        best_pipeline: Pipeline | None = None
        best_score: ModelScore | None = None

        for candidate in self.candidates:
            parameter_grid = candidate.parameter_grid or ({},)
            for parameters in parameter_grid:
                estimator = candidate.estimator_factory(parameters)
                pipeline = Pipeline(
                    steps=[
                        ("preprocessor", build_preprocessor(self.schema)),
                        ("regressor", estimator),
                    ]
                )
                pipeline.fit(split_data.x_train, split_data.y_train)
                y_pred = pipeline.predict(split_data.x_test)
                metrics = compute_regression_metrics(
                    split_data.y_test.to_numpy(),
                    y_pred,
                )
                score = ModelScore(
                    model_name=candidate.name,
                    parameters=dict(parameters),
                    metrics=metrics,
                )
                scores.append(score)

                if best_score is None:
                    best_score = score
                    best_pipeline = pipeline
                    continue

                if score.metrics[self.primary_metric] < best_score.metrics[self.primary_metric]:
                    best_score = score
                    best_pipeline = pipeline

        if best_pipeline is None or best_score is None:
            raise RuntimeError("Training did not produce a valid model.")

        ordered = tuple(sorted(scores, key=lambda item: item.metrics[self.primary_metric]))
        return TrainingResult(
            pipeline=best_pipeline,
            best_score=best_score,
            leaderboard=ordered,
            schema=self.schema,
        )
