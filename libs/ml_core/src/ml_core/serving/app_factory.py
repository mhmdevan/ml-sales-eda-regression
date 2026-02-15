from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, create_model

from ml_core.features.schema import FeatureSchema
from ml_core.registry.artifacts import ArtifactRegistry


class PredictionResponse(BaseModel):
    prediction: float
    model_name: str
    target_name: str
    input_features: dict[str, Any]


@dataclass(slots=True)
class RuntimeState:
    model: Any | None
    metadata: dict[str, Any]


def build_request_model(schema: FeatureSchema, name: str) -> type[BaseModel]:
    fields: dict[str, tuple[type[Any], Any]] = {}
    for feature in schema.numeric_features:
        fields[feature] = (float, Field(...))
    for feature in schema.categorical_features:
        fields[feature] = (str, Field(...))
    return create_model(name, **fields)


def create_regression_app(
    *,
    title: str,
    description: str,
    schema: FeatureSchema,
    artifact_dir: Path,
    model_filename: str = "model.joblib",
    metadata_filename: str = "metadata.json",
    version: str = "1.0.0",
) -> FastAPI:
    schema.ensure_valid()
    request_model = build_request_model(schema, f"{title.replace(' ', '')}Request")
    registry = ArtifactRegistry(
        root_dir=artifact_dir,
        model_filename=model_filename,
        metadata_filename=metadata_filename,
    )
    state = RuntimeState(model=None, metadata={})

    app = FastAPI(title=title, description=description, version=version)

    @app.on_event("startup")
    def startup_event() -> None:
        if not registry.model_exists():
            state.model = None
            state.metadata = {}
            return
        state.model = registry.load_model()
        state.metadata = registry.load_metadata()

    @app.get("/health")
    def health() -> dict[str, Any]:
        status = "ok" if state.model is not None else "model_not_loaded"
        return {
            "status": status,
            "model_name": state.metadata.get("best_model_name"),
            "target_name": state.metadata.get("target_name", schema.target_name),
            "feature_count": len(schema.all_features),
        }

    @app.post("/predict", response_model=PredictionResponse)
    def predict(payload: request_model) -> PredictionResponse:
        if state.model is None:
            raise HTTPException(status_code=503, detail="Model artifact is not available.")

        payload_dict = payload.model_dump()
        ordered = {name: payload_dict[name] for name in schema.all_features}
        frame = pd.DataFrame([ordered])
        prediction = float(state.model.predict(frame)[0])

        return PredictionResponse(
            prediction=prediction,
            model_name=str(state.metadata.get("best_model_name", "unknown")),
            target_name=str(state.metadata.get("target_name", schema.target_name)),
            input_features=ordered,
        )

    return app
