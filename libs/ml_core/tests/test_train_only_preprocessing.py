from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from ml_core.data.contracts import SplitDataset
from ml_core.features.schema import FeatureSchema
from ml_core.modeling.contracts import CandidateModel
from ml_core.modeling.trainer import RegressionTrainer


def test_preprocessing_is_fit_on_train_only() -> None:
    x_train = pd.DataFrame(
        {
            "x_num": [1.0, 2.0, 3.0, 4.0],
            "x_cat": ["a", "a", "b", "b"],
        }
    )
    y_train = pd.Series([10.0, 12.0, 14.0, 16.0], name="target")

    x_test = pd.DataFrame(
        {
            "x_num": [5000.0],
            "x_cat": ["test_only_category"],
        }
    )
    y_test = pd.Series([0.0], name="target")

    split_data = SplitDataset(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
    )

    schema = FeatureSchema.create(
        numeric_features=["x_num"],
        categorical_features=["x_cat"],
        target_name="target",
    )
    trainer = RegressionTrainer(
        schema=schema,
        candidates=[
            CandidateModel(
                name="LinearRegression",
                estimator_factory=lambda _: LinearRegression(),
                parameter_grid=({},),
            )
        ],
    )

    result = trainer.fit(split_data)
    pipeline = result.pipeline
    preprocessor = pipeline.named_steps["preprocessor"]
    numeric_scaler = preprocessor.named_transformers_["numeric"]
    categorical_encoder = preprocessor.named_transformers_["categorical"]

    train_mean = float(x_train["x_num"].mean())
    full_mean = float(pd.concat([x_train["x_num"], x_test["x_num"]], axis=0).mean())

    assert numeric_scaler.mean_[0] == pytest.approx(train_mean)
    assert numeric_scaler.mean_[0] != pytest.approx(full_mean)
    assert "test_only_category" not in set(categorical_encoder.categories_[0])

    prediction = pipeline.predict(x_test)
    assert prediction.shape == (1,)
    assert np.isfinite(prediction[0])
