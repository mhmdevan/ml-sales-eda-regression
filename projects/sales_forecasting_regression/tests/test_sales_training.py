from pathlib import Path

from sales_forecasting_regression.train import train_sales_model


def test_train_sales_model_writes_artifacts(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "models"
    report_path = tmp_path / "reports" / "metrics.json"

    payload = train_sales_model(
        data_path=tmp_path / "missing.csv",
        artifact_dir=artifact_dir,
        report_path=report_path,
        enable_mlflow=False,
    )

    assert payload["best_metrics"]["rmse"] > 0
    assert (artifact_dir / "sales_regressor.joblib").exists()
    assert (artifact_dir / "sales_regressor_metadata.json").exists()
    assert report_path.exists()
