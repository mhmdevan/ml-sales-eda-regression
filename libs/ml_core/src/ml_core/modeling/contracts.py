from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from sklearn.base import RegressorMixin
from sklearn.pipeline import Pipeline

from ml_core.features.schema import FeatureSchema


EstimatorFactory = Callable[[dict[str, Any]], RegressorMixin]


@dataclass(frozen=True, slots=True)
class CandidateModel:
    name: str
    estimator_factory: EstimatorFactory
    parameter_grid: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class ModelScore:
    model_name: str
    parameters: dict[str, Any]
    metrics: dict[str, float]


@dataclass(frozen=True, slots=True)
class TrainingResult:
    pipeline: Pipeline
    best_score: ModelScore
    leaderboard: tuple[ModelScore, ...]
    schema: FeatureSchema
