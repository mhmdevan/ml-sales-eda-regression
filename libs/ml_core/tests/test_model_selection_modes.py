from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from ml_core.data.contracts import SplitDataset
from ml_core.data.splitters import split_regression_frame
from ml_core.features.schema import FeatureSchema
from ml_core.modeling.contracts import CandidateModel
from ml_core.modeling.trainer import RegressionTrainer


def _nonlinear_frame(n_rows: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    x1 = rng.uniform(-3.0, 3.0, size=n_rows)
    x2 = rng.normal(0.0, 1.0, size=n_rows)
    target = 4.0 + (x1**2) + 0.2 * x2 + rng.normal(0.0, 0.2, size=n_rows)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": target})


def _trainer() -> tuple[RegressionTrainer, SplitDataset]:
    frame = _nonlinear_frame()
    schema = FeatureSchema.create(["x1", "x2"], [], "target")
    split = split_regression_frame(frame, schema, test_size=0.2, random_state=42)

    candidates = [
        CandidateModel(
            name="LinearRegression",
            estimator_factory=lambda _: LinearRegression(),
            parameter_grid=({},),
        ),
        CandidateModel(
            name="RandomForestRegressor",
            estimator_factory=lambda params: RandomForestRegressor(random_state=42, n_jobs=-1, **params),
            parameter_grid=({"n_estimators": 25, "max_depth": 2},),
            tuning_parameter_grid={
                "n_estimators": [25, 120],
                "max_depth": [2, None],
            },
            tuning_cv=3,
            tuning_n_jobs=-1,
        ),
    ]
    return RegressionTrainer(schema=schema, candidates=candidates, primary_metric="rmse"), split


def test_selection_mode_baseline_returns_baseline_scores_only() -> None:
    trainer, split = _trainer()
    result = trainer.fit(split, selection_mode="baseline")

    assert len(result.leaderboard) == 2
    assert result.best_score.metrics["rmse"] > 0


def test_selection_mode_tuned_runs_grid_search_and_returns_best() -> None:
    trainer, split = _trainer()
    result = trainer.fit(split, selection_mode="tuned")

    assert len(result.leaderboard) == 1
    assert result.best_score.model_name == "RandomForestRegressor"
    assert set(result.best_score.parameters).issubset({"n_estimators", "max_depth"})
    assert result.best_score.metrics["rmse"] > 0


def test_selection_mode_both_combines_baseline_and_tuned_scores() -> None:
    trainer, split = _trainer()
    result = trainer.fit(split, selection_mode="both")

    assert len(result.leaderboard) == 3
    assert result.best_score.metrics["rmse"] > 0


def test_selection_mode_raises_for_invalid_value() -> None:
    trainer, split = _trainer()

    with pytest.raises(ValueError):
        trainer.fit(split, selection_mode="invalid")
