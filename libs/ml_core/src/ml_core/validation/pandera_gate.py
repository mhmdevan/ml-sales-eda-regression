from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ml_core.features.schema import FeatureSchema


@dataclass(frozen=True, slots=True)
class ValidationContext:
    name: str


def build_pandera_schema(
    schema: FeatureSchema,
    *,
    require_target: bool,
    allow_extra_columns: bool = False,
):
    try:
        import pandera.pandas as pa
    except Exception as exc:
        raise RuntimeError("Pandera is required for validation gates.") from exc

    columns: dict[str, Any] = {}

    for feature in schema.numeric_features:
        columns[feature] = pa.Column(float, nullable=False, coerce=True)

    for feature in schema.categorical_features:
        columns[feature] = pa.Column(str, nullable=False, coerce=True)

    if require_target:
        columns[schema.target_name] = pa.Column(float, nullable=False, coerce=True)

    return pa.DataFrameSchema(
        columns=columns,
        strict=not allow_extra_columns,
        coerce=True,
    )


def validate_frame(
    frame: pd.DataFrame,
    schema: FeatureSchema,
    *,
    require_target: bool,
    context: str,
    allow_extra_columns: bool = False,
) -> pd.DataFrame:
    schema.ensure_valid()
    pandera_schema = build_pandera_schema(
        schema,
        require_target=require_target,
        allow_extra_columns=allow_extra_columns,
    )

    try:
        validated = pandera_schema.validate(frame, lazy=True)
    except Exception as exc:
        raise ValueError(f"Validation gate failed in {context}: {exc}") from exc

    return validated
