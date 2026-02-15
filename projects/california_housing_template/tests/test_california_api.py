from pathlib import Path

from fastapi.testclient import TestClient

from california_housing_template.api import create_app
from california_housing_template.train import train_california_housing_model


def test_california_api_predict_endpoint(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "models"
    train_california_housing_model(
        artifact_dir=artifact_dir,
        use_synthetic_if_fetch_fails=True,
        force_synthetic=True,
        selection_mode="baseline",
    )

    app = create_app(artifact_dir=artifact_dir)
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        response = client.post(
            "/predict",
            json={
                "MedInc": 8.0,
                "HouseAge": 22,
                "AveRooms": 5.5,
                "AveBedrms": 1.1,
                "Population": 700,
                "AveOccup": 2.1,
                "Latitude": 37.7,
                "Longitude": -122.4,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert "prediction" in body
        assert "y_hat" in body
        assert "p10" in body
        assert "p90" in body
        assert body["target_name"] == "MedHouseVal"
        assert body["y_hat"] == body["prediction"]
        assert body["p10"] <= body["y_hat"] <= body["p90"]

        explain = client.post(
            "/explain",
            json={
                "MedInc": 8.0,
                "HouseAge": 22,
                "AveRooms": 5.5,
                "AveBedrms": 1.1,
                "Population": 700,
                "AveOccup": 2.1,
                "Latitude": 37.7,
                "Longitude": -122.4,
            },
        )
        assert explain.status_code == 200
        explain_body = explain.json()
        assert "available" in explain_body
        assert explain_body["target_name"] == "MedHouseVal"
        if explain_body["available"]:
            assert explain_body["method"] == "shap_tree"
            assert len(explain_body["contributions"]) > 0
        else:
            assert explain_body["reason"] is not None
