# tests/test_regression_pipeline.py

from sales_regression import (
    build_regression_dataset,
    train_baseline_and_tuned_models,
)

def test_build_regression_dataset_produces_features_and_target():
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
    metrics_df = train_baseline_and_tuned_models(limit_rows=300)

    assert not metrics_df.empty
    for col in ["mse", "rmse", "mae", "r2"]:
        assert col in metrics_df.columns

    # rmse should be positive
    assert (metrics_df["rmse"] > 0).all()
