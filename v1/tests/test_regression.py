# tests/test_regression.py

from sales_regression import (
    build_regression_dataset,
    train_baseline_and_tuned_models,
)


def test_build_regression_dataset_shapes_and_features():
    """
    Basic sanity check:
    - X, y have same length
    - we have at least one row
    - numeric & categorical features
    """
    X, y, numeric_features, categorical_features = build_regression_dataset(
        limit_rows=300
    )

    assert len(X) == len(y)
    assert len(X) > 0

    for col in numeric_features:
        assert col in X.columns

    for col in categorical_features:
        assert col in X.columns


def test_train_baseline_and_tuned_models_returns_metrics():
    """
    End-to-end test for regression training
    """
    metrics_df = train_baseline_and_tuned_models(limit_rows=300)

    assert metrics_df is not None
    assert not metrics_df.empty

    for col in ["mse", "rmse", "mae", "r2"]:
        assert col in metrics_df.columns

    assert (metrics_df["rmse"] > 0).all()
