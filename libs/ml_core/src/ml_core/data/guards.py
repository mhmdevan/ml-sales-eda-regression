from __future__ import annotations

from typing import Iterable

import pandas as pd


def ensure_required_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    required = list(columns)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def ensure_no_leakage(feature_names: Iterable[str], target_name: str) -> None:
    names = list(feature_names)
    if target_name in names:
        raise ValueError(f"Target column '{target_name}' must not appear in features.")
