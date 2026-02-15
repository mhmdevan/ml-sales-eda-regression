from pathlib import Path

from fastapi.testclient import TestClient

from california_housing_template.api import create_app


def test_california_api_reports_unavailable_without_artifacts(tmp_path: Path) -> None:
    app = create_app(artifact_dir=tmp_path / "models")

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "model_not_loaded"

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
        assert response.status_code == 503

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
        assert explain.status_code == 503
