from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    project_name: str
    task_type: str
    feature_names: tuple[str, ...]
    target_name: str
    best_model_name: str
    metrics: dict[str, float]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    extra: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "project_name": self.project_name,
            "task_type": self.task_type,
            "feature_names": list(self.feature_names),
            "target_name": self.target_name,
            "best_model_name": self.best_model_name,
            "metrics": self.metrics,
            "created_at": self.created_at,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "ModelMetadata":
        return cls(
            project_name=str(raw["project_name"]),
            task_type=str(raw["task_type"]),
            feature_names=tuple(raw["feature_names"]),
            target_name=str(raw["target_name"]),
            best_model_name=str(raw["best_model_name"]),
            metrics={str(k): float(v) for k, v in dict(raw["metrics"]).items()},
            created_at=str(raw.get("created_at", datetime.now(timezone.utc).isoformat())),
            extra=dict(raw.get("extra", {})),
        )
