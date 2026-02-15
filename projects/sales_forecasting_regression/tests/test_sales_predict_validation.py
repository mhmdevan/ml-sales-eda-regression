from pathlib import Path

import pytest

from sales_forecasting_regression.predict import predict_one
from sales_forecasting_regression.train import train_sales_model


def test_predict_one_raises_for_missing_features(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "models"
    train_sales_model(
        data_path=tmp_path / "missing.csv",
        artifact_dir=artifact_dir,
        report_path=tmp_path / "reports" / "metrics.json",
        enable_mlflow=False,
    )

    with pytest.raises(ValueError):
        predict_one(
            {
                "QUANTITYORDERED": 36,
                "PRICEEACH": 91.0,
            },
            artifact_dir=artifact_dir,
        )
