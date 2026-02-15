import pandas as pd
import pytest

from ml_core.features.schema import FeatureSchema


def test_schema_validates_frame_columns() -> None:
    schema = FeatureSchema.create(["x1", "x2"], ["cat"], "target")
    frame = pd.DataFrame({"x1": [1.0], "x2": [2.0], "cat": ["a"], "target": [3.0]})
    schema.validate_frame(frame)


def test_schema_raises_for_missing_columns() -> None:
    schema = FeatureSchema.create(["x1"], [], "target")
    frame = pd.DataFrame({"x1": [1.0]})
    with pytest.raises(ValueError):
        schema.validate_frame(frame)
