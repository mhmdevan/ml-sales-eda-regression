from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ml_core.features.schema import FeatureSchema


SALES_NUMERIC_FEATURES = [
    "QUANTITYORDERED",
    "PRICEEACH",
    "ORDERLINENUMBER",
    "MSRP",
    "QTR_ID",
    "MONTH_ID",
    "YEAR_ID",
]

SALES_CATEGORICAL_FEATURES = [
    "PRODUCTLINE",
    "COUNTRY",
    "DEALSIZE",
]

SALES_TARGET = "SALES"


@dataclass(frozen=True, slots=True)
class SalesPaths:
    project_root: Path
    models_dir: Path
    reports_dir: Path

    @classmethod
    def default(cls) -> "SalesPaths":
        project_root = Path(__file__).resolve().parents[2]
        return cls(
            project_root=project_root,
            models_dir=project_root / "models",
            reports_dir=project_root / "reports",
        )


def sales_schema() -> FeatureSchema:
    return FeatureSchema.create(
        numeric_features=SALES_NUMERIC_FEATURES,
        categorical_features=SALES_CATEGORICAL_FEATURES,
        target_name=SALES_TARGET,
    )
