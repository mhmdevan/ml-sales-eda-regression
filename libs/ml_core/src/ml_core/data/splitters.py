from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from ml_core.data.contracts import SplitDataset
from ml_core.data.guards import ensure_no_leakage, ensure_required_columns
from ml_core.features.schema import FeatureSchema


def split_regression_frame(
    frame: pd.DataFrame,
    schema: FeatureSchema,
    test_size: float = 0.2,
    random_state: int = 42,
) -> SplitDataset:
    schema.ensure_valid()
    ensure_required_columns(frame, list(schema.all_features) + [schema.target_name])
    ensure_no_leakage(schema.all_features, schema.target_name)
    x = frame.loc[:, list(schema.all_features)].copy()
    y = frame.loc[:, schema.target_name].copy()
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    return SplitDataset(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
