from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from ml_core.modeling.contracts import ModelScore, TrainingResult
from ml_core.registry.metadata import ModelMetadata
from ml_core.registry.onnx_export import export_pipeline_to_onnx


class ArtifactRegistry:
    def __init__(
        self,
        root_dir: Path,
        model_filename: str = "model.joblib",
        metadata_filename: str = "metadata.json",
        onnx_filename: str = "model.onnx",
        schema_filename: str = "schema.json",
    ) -> None:
        self.root_dir = root_dir
        self.model_path = root_dir / model_filename
        self.metadata_path = root_dir / metadata_filename
        self.onnx_path = root_dir / onnx_filename
        self.schema_path = root_dir / schema_filename

    def save_training_result(
        self,
        result: TrainingResult,
        project_name: str,
        extra_metadata: dict[str, Any] | None = None,
        export_onnx: bool = False,
        sample_frame: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(result.pipeline, self.model_path)

        leaderboard: list[dict[str, Any]] = [
            {
                "model_name": score.model_name,
                "parameters": score.parameters,
                "metrics": score.metrics,
            }
            for score in result.leaderboard
        ]

        merged_extra = dict(extra_metadata or {})
        merged_extra["leaderboard"] = leaderboard

        metadata = ModelMetadata(
            project_name=project_name,
            task_type="regression",
            feature_names=result.schema.all_features,
            target_name=result.schema.target_name,
            best_model_name=result.best_score.model_name,
            metrics=result.best_score.metrics,
            extra=merged_extra,
        )

        payload = metadata.as_dict()
        self.metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        schema_payload = {
            "numeric_features": list(result.schema.numeric_features),
            "categorical_features": list(result.schema.categorical_features),
            "feature_names": list(result.schema.all_features),
            "target_name": result.schema.target_name,
        }
        self.schema_path.write_text(json.dumps(schema_payload, indent=2), encoding="utf-8")

        exported_onnx: Path | None = None
        if export_onnx and sample_frame is not None:
            exported_onnx = export_pipeline_to_onnx(result.pipeline, sample_frame, self.onnx_path)

        if exported_onnx is not None:
            payload["onnx_path"] = str(exported_onnx)

        return payload

    def load_model(self) -> Any:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {self.model_path}")
        return joblib.load(self.model_path)

    def load_metadata(self) -> dict[str, Any]:
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata artifact not found: {self.metadata_path}")
        return json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def load_schema(self) -> dict[str, Any]:
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema artifact not found: {self.schema_path}")
        return json.loads(self.schema_path.read_text(encoding="utf-8"))

    def model_exists(self) -> bool:
        return self.model_path.exists() and self.metadata_path.exists() and self.schema_path.exists()


def leaderboard_as_records(leaderboard: tuple[ModelScore, ...]) -> list[dict[str, Any]]:
    return [
        {
            "model_name": score.model_name,
            "parameters": score.parameters,
            "metrics": score.metrics,
        }
        for score in leaderboard
    ]
