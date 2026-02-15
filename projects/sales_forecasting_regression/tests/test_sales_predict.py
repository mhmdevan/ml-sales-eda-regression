from pathlib import Path

from sales_forecasting_regression.predict import predict_one
from sales_forecasting_regression.train import train_sales_model


def test_predict_one_returns_float(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "models"
    train_sales_model(
        data_path=tmp_path / "missing.csv",
        artifact_dir=artifact_dir,
        report_path=tmp_path / "reports" / "metrics.json",
        enable_mlflow=False,
    )

    features = {
        "QUANTITYORDERED": 36,
        "PRICEEACH": 91.0,
        "ORDERLINENUMBER": 3,
        "MSRP": 120.0,
        "QTR_ID": 2,
        "MONTH_ID": 5,
        "YEAR_ID": 2004,
        "PRODUCTLINE": "Classic Cars",
        "COUNTRY": "USA",
        "DEALSIZE": "Medium",
    }

    prediction = predict_one(features, artifact_dir=artifact_dir)
    assert isinstance(prediction, float)
