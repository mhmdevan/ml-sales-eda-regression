from pathlib import Path

from california_housing_template.predict import predict_one
from california_housing_template.train import train_california_housing_model


def test_predict_returns_float(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "models"
    train_california_housing_model(
        artifact_dir=artifact_dir,
        use_synthetic_if_fetch_fails=True,
        force_synthetic=True,
    )

    prediction = predict_one(
        {
            "MedInc": 8.3,
            "HouseAge": 20,
            "AveRooms": 6.5,
            "AveBedrms": 1.0,
            "Population": 500,
            "AveOccup": 2.5,
            "Latitude": 37.86,
            "Longitude": -122.22,
        },
        artifact_dir=artifact_dir,
    )

    assert isinstance(prediction, float)
