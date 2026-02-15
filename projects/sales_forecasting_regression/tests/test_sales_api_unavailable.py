from pathlib import Path

from fastapi.testclient import TestClient

from sales_forecasting_regression.api import create_app


def test_sales_api_reports_unavailable_without_artifacts(tmp_path: Path) -> None:
    app = create_app(artifact_dir=tmp_path / "models")

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "model_not_loaded"

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
        assert response.status_code == 503
