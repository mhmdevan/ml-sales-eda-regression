from pathlib import Path

from fastapi.testclient import TestClient

from sales_forecasting_regression.api import create_app
from sales_forecasting_regression.train import train_sales_model


def test_sales_api_predict_endpoint(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "models"
    train_sales_model(
        data_path=tmp_path / "missing.csv",
        artifact_dir=artifact_dir,
        report_path=tmp_path / "reports" / "metrics.json",
        enable_mlflow=False,
    )

    app = create_app(artifact_dir=artifact_dir)
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        response = client.post(
            "/predict",
            json={
                "QUANTITYORDERED": 25,
                "PRICEEACH": 80.0,
                "ORDERLINENUMBER": 2,
                "MSRP": 104.0,
                "QTR_ID": 3,
                "MONTH_ID": 7,
                "YEAR_ID": 2004,
                "PRODUCTLINE": "Classic Cars",
                "COUNTRY": "USA",
                "DEALSIZE": "Small",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert "prediction" in body
        assert body["target_name"] == "SALES"
