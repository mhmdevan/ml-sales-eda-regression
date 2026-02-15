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
        selection_mode="baseline",
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
        assert "y_hat" in body
        assert "p10" in body
        assert "p90" in body
        assert body["target_name"] == "SALES"
        assert body["y_hat"] == body["prediction"]
        assert body["p10"] <= body["y_hat"] <= body["p90"]

        explain = client.post(
            "/explain",
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
        assert explain.status_code == 200
        explain_body = explain.json()
        assert "available" in explain_body
        assert "prediction" in explain_body
        assert "y_hat" in explain_body
        assert explain_body["target_name"] == "SALES"
        assert explain_body["y_hat"] == explain_body["prediction"]
        if explain_body["available"]:
            assert explain_body["method"] == "shap_tree"
            assert len(explain_body["contributions"]) > 0
        else:
            assert explain_body["reason"] is not None
