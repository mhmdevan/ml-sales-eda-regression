from pathlib import Path

import pytest

from california_housing_template.predict import predict_one
from california_housing_template.train import train_california_housing_model


def test_predict_one_raises_for_missing_features(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "models"
    train_california_housing_model(
        artifact_dir=artifact_dir,
        use_synthetic_if_fetch_fails=True,
        force_synthetic=True,
        selection_mode="baseline",
    )

    with pytest.raises(ValueError):
        predict_one(
            {
                "MedInc": 8.3,
                "HouseAge": 20,
            },
            artifact_dir=artifact_dir,
        )
