import pandas as pd
import pytest

from ml_core.features.schema import FeatureSchema
from ml_core.validation.pandera_gate import validate_frame


def test_validation_gate_accepts_valid_training_frame() -> None:
    schema = FeatureSchema.create(["x1", "x2"], ["cat"], "target")
    frame = pd.DataFrame(
        {
            "x1": [1.0, 2.0],
            "x2": [3.0, 4.0],
            "cat": ["a", "b"],
            "target": [5.0, 6.0],
        }
    )

    validated = validate_frame(
        frame=frame,
        schema=schema,
        require_target=True,
        context="unit_test",
        allow_extra_columns=False,
    )

    assert list(validated.columns) == ["x1", "x2", "cat", "target"]


def test_validation_gate_rejects_missing_required_column() -> None:
    schema = FeatureSchema.create(["x1", "x2"], ["cat"], "target")
    frame = pd.DataFrame(
        {
            "x1": [1.0],
            "cat": ["a"],
            "target": [3.0],
        }
    )

    with pytest.raises(ValueError):
        validate_frame(
            frame=frame,
            schema=schema,
            require_target=True,
            context="unit_test",
            allow_extra_columns=False,
        )


def test_validation_gate_rejects_invalid_numeric_value() -> None:
    schema = FeatureSchema.create(["x1"], [], "target")
    frame = pd.DataFrame(
        {
            "x1": ["invalid"],
            "target": [1.0],
        }
    )

    with pytest.raises(ValueError):
        validate_frame(
            frame=frame,
            schema=schema,
            require_target=True,
            context="unit_test",
            allow_extra_columns=False,
        )
