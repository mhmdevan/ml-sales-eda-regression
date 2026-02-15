from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class FeatureSchema:
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    target_name: str

    @classmethod
    def create(
        cls,
        numeric_features: list[str] | tuple[str, ...],
        categorical_features: list[str] | tuple[str, ...],
        target_name: str,
    ) -> "FeatureSchema":
        return cls(
            numeric_features=tuple(numeric_features),
            categorical_features=tuple(categorical_features),
            target_name=target_name,
        )

    @property
    def all_features(self) -> tuple[str, ...]:
        return self.numeric_features + self.categorical_features

    def ensure_valid(self) -> None:
        if not self.numeric_features and not self.categorical_features:
            raise ValueError("Feature schema must include at least one feature.")
        duplicates = [name for name in self.all_features if self.all_features.count(name) > 1]
        if duplicates:
            raise ValueError(f"Duplicated features are not allowed: {sorted(set(duplicates))}")
        if self.target_name in self.all_features:
            raise ValueError("Target column must not be part of input features.")

    def validate_frame(self, frame: pd.DataFrame, require_target: bool = True) -> None:
        required = list(self.all_features)
        if require_target:
            required.append(self.target_name)
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
