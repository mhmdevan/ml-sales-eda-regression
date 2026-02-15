from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ml_core.features.schema import FeatureSchema


CALIFORNIA_FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

CALIFORNIA_TARGET = "MedHouseVal"


@dataclass(frozen=True, slots=True)
class CaliforniaPaths:
    project_root: Path
    models_dir: Path

    @classmethod
    def default(cls) -> "CaliforniaPaths":
        project_root = Path(__file__).resolve().parents[2]
        return cls(project_root=project_root, models_dir=project_root / "models")


def california_schema() -> FeatureSchema:
    return FeatureSchema.create(
        numeric_features=CALIFORNIA_FEATURES,
        categorical_features=[],
        target_name=CALIFORNIA_TARGET,
    )
