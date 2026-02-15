from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

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

    def fit(self, split_data: SplitDataset, selection_mode: str = "both") -> TrainingResult:
        if selection_mode not in {"baseline", "tuned", "both"}:
            raise ValueError("selection_mode must be one of: baseline, tuned, both.")
        self.schema.ensure_valid()
        scores: list[ModelScore] = []
        best_pipeline: Pipeline | None = None
        best_score: ModelScore | None = None

        for candidate in self.candidates:
            if selection_mode in {"baseline", "both"}:
                parameter_grid = candidate.parameter_grid or ({},)
                for parameters in parameter_grid:
                    pipeline = self._build_pipeline(candidate=candidate, parameters=parameters)
                    score = self._fit_and_score_pipeline(
                        candidate_name=candidate.name,
                        parameters=parameters,
                        pipeline=pipeline,
                        split_data=split_data,
                    )
                    scores.append(score)
                    best_pipeline, best_score = self._pick_best(
                        current_pipeline=pipeline,
                        current_score=score,
                        best_pipeline=best_pipeline,
                        best_score=best_score,
                    )

            if selection_mode in {"tuned", "both"} and candidate.tuning_parameter_grid:
                tuned_pipeline, tuned_parameters = self._fit_tuned_pipeline(candidate=candidate, split_data=split_data)
                tuned_score = self._score_pipeline(
                    candidate_name=candidate.name,
                    parameters=tuned_parameters,
                    pipeline=tuned_pipeline,
                    split_data=split_data,
                )
                scores.append(tuned_score)
                best_pipeline, best_score = self._pick_best(
                    current_pipeline=tuned_pipeline,
                    current_score=tuned_score,
                    best_pipeline=best_pipeline,
                    best_score=best_score,
                )

        if best_pipeline is None or best_score is None:
            raise RuntimeError("Training did not produce a valid model for the selected mode.")

        ordered = tuple(sorted(scores, key=lambda item: item.metrics[self.primary_metric]))
        return TrainingResult(
            pipeline=best_pipeline,
            best_score=best_score,
            leaderboard=ordered,
            schema=self.schema,
        )

    def _build_pipeline(self, candidate: CandidateModel, parameters: dict[str, object]) -> Pipeline:
        estimator = candidate.estimator_factory(parameters)
        return Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(self.schema)),
                ("regressor", estimator),
            ]
        )

    def _score_pipeline(
        self,
        *,
        candidate_name: str,
        parameters: dict[str, object],
        pipeline: Pipeline,
        split_data: SplitDataset,
    ) -> ModelScore:
        y_pred = pipeline.predict(split_data.x_test)
        metrics = compute_regression_metrics(
            split_data.y_test.to_numpy(),
            y_pred,
        )
        return ModelScore(
            model_name=candidate_name,
            parameters=dict(parameters),
            metrics=metrics,
        )

    def _fit_and_score_pipeline(
        self,
        *,
        candidate_name: str,
        parameters: dict[str, object],
        pipeline: Pipeline,
        split_data: SplitDataset,
    ) -> ModelScore:
        pipeline.fit(split_data.x_train, split_data.y_train)
        return self._score_pipeline(
            candidate_name=candidate_name,
            parameters=parameters,
            pipeline=pipeline,
            split_data=split_data,
        )

    def _fit_tuned_pipeline(self, *, candidate: CandidateModel, split_data: SplitDataset) -> tuple[Pipeline, dict[str, object]]:
        grid = candidate.tuning_parameter_grid
        if not grid:
            raise ValueError("tuning_parameter_grid must be provided for tuned selection mode.")

        base_pipeline = self._build_pipeline(candidate=candidate, parameters={})
        prefixed_grid = {f"regressor__{key}": values for key, values in grid.items()}
        search = GridSearchCV(
            estimator=base_pipeline,
            param_grid=prefixed_grid,
            cv=candidate.tuning_cv,
            scoring=self._scoring_name(),
            n_jobs=candidate.tuning_n_jobs,
            refit=True,
        )
        search.fit(split_data.x_train, split_data.y_train)
        tuned_parameters = {
            key.replace("regressor__", "", 1): value
            for key, value in search.best_params_.items()
            if key.startswith("regressor__")
        }
        return search.best_estimator_, tuned_parameters

    def _scoring_name(self) -> str:
        scoring_by_metric = {
            "rmse": "neg_root_mean_squared_error",
            "mse": "neg_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
        }
        try:
            return scoring_by_metric[self.primary_metric]
        except KeyError as exc:
            raise ValueError(f"Unsupported primary metric for tuning: {self.primary_metric}") from exc

    def _pick_best(
        self,
        *,
        current_pipeline: Pipeline,
        current_score: ModelScore,
        best_pipeline: Pipeline | None,
        best_score: ModelScore | None,
    ) -> tuple[Pipeline, ModelScore]:
        if best_score is None or best_pipeline is None:
            return current_pipeline, current_score

        if current_score.metrics[self.primary_metric] < best_score.metrics[self.primary_metric]:
            return current_pipeline, current_score

        return best_pipeline, best_score
