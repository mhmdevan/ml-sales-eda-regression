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
        assert body["target_name"] == "MedHouseVal"
