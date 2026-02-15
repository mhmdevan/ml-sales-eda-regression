from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, create_model

from ml_core.features.schema import FeatureSchema
from ml_core.monitoring.conformal import conformal_interval
from ml_core.monitoring.explainability import explain_single_prediction
from ml_core.registry.versioning import resolve_alias_artifact_paths
from ml_core.validation.pandera_gate import validate_frame


class PredictionResponse(BaseModel):
    prediction: float
    y_hat: float
    p10: float | None = None
    p90: float | None = None
    interval_method: str | None = None
    model_name: str
    target_name: str
    input_features: dict[str, Any]


class ExplainResponse(BaseModel):
    available: bool
    reason: str | None = None
    method: str | None = None
    prediction: float
    y_hat: float
    base_value: float | None = None
    contributions: dict[str, float] = Field(default_factory=dict)
    model_name: str
    target_name: str
    input_features: dict[str, Any]


@dataclass(slots=True)
class RuntimeState:
    model: Any | None
    metadata: dict[str, Any]
    shadow_model: Any | None
    shadow_metadata: dict[str, Any]


def build_request_model(schema: FeatureSchema, name: str) -> type[BaseModel]:
    fields: dict[str, tuple[type[Any], Any]] = {}
    for feature in schema.numeric_features:
        fields[feature] = (float, Field(...))
    for feature in schema.categorical_features:
        fields[feature] = (str, Field(...))
    return create_model(name, **fields)


def _load_model_with_metadata(model_path: Path, metadata_path: Path) -> tuple[Any, dict[str, Any]]:
    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return model, metadata


def _append_shadow_log(
    *,
    shadow_log_path: Path,
    features: dict[str, Any],
    primary_prediction: float,
    shadow_prediction: float,
    primary_model_name: str,
    shadow_model_name: str,
) -> None:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": features,
        "primary_prediction": primary_prediction,
        "shadow_prediction": shadow_prediction,
        "prediction_delta": shadow_prediction - primary_prediction,
        "primary_model_name": primary_model_name,
        "shadow_model_name": shadow_model_name,
    }

    shadow_log_path.parent.mkdir(parents=True, exist_ok=True)
    with shadow_log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _ensemble_quantile_interval(
    *,
    model: Any,
    frame: pd.DataFrame,
    lower_q: float = 0.1,
    upper_q: float = 0.9,
) -> tuple[float, float] | None:
    try:
        preprocessor = getattr(model, "named_steps", {}).get("preprocessor")
        regressor = getattr(model, "named_steps", {}).get("regressor", model)
        estimators = getattr(regressor, "estimators_", None)
        if estimators is None:
            return None

        transformed = preprocessor.transform(frame) if preprocessor is not None else frame
        member_predictions = np.asarray(
            [float(estimator.predict(transformed)[0]) for estimator in estimators],
            dtype=float,
        )
        if member_predictions.size == 0 or not np.all(np.isfinite(member_predictions)):
            return None

        lower = float(np.quantile(member_predictions, lower_q))
        upper = float(np.quantile(member_predictions, upper_q))
        return (min(lower, upper), max(lower, upper))
    except Exception:
        return None


def _residual_interval_from_metadata(
    *,
    prediction: float,
    metadata: dict[str, Any],
) -> tuple[float, float] | None:
    uncertainty = metadata.get("extra", {}).get("uncertainty")
    if not isinstance(uncertainty, dict):
        return None
    if "residual_q10" not in uncertainty or "residual_q90" not in uncertainty:
        return None
    lower = prediction + float(uncertainty["residual_q10"])
    upper = prediction + float(uncertainty["residual_q90"])
    return (min(lower, upper), max(lower, upper))


def _conformal_interval_from_metadata(
    *,
    prediction: float,
    metadata: dict[str, Any],
) -> tuple[float, float] | None:
    conformal = metadata.get("extra", {}).get("conformal_intervals")
    if not isinstance(conformal, dict):
        return None
    if "q_hat" not in conformal:
        return None
    q_hat = float(conformal["q_hat"])
    if not np.isfinite(q_hat) or q_hat < 0:
        return None
    return conformal_interval(prediction=prediction, q_hat=q_hat)


def _compute_prediction_interval(
    *,
    model: Any,
    metadata: dict[str, Any],
    frame: pd.DataFrame,
    prediction: float,
) -> tuple[float, float, str] | None:
    conformal = _conformal_interval_from_metadata(prediction=prediction, metadata=metadata)
    if conformal is not None:
        return conformal[0], conformal[1], "conformal_intervals"

    ensemble_interval = _ensemble_quantile_interval(model=model, frame=frame)
    if ensemble_interval is not None:
        return ensemble_interval[0], ensemble_interval[1], "ensemble_quantiles"

    residual_interval = _residual_interval_from_metadata(prediction=prediction, metadata=metadata)
    if residual_interval is not None:
        return residual_interval[0], residual_interval[1], "residual_quantiles"

    return None


def create_regression_app(
    *,
    title: str,
    description: str,
    schema: FeatureSchema,
    artifact_dir: Path,
    model_filename: str = "model.joblib",
    metadata_filename: str = "metadata.json",
    shadow_model_filename: str | None = None,
    shadow_metadata_filename: str | None = None,
    shadow_log_path: Path | None = None,
    registry_dir: Path | None = None,
    registry_model_name: str | None = None,
    primary_alias: str = "prod",
    shadow_alias: str | None = "staging",
    version: str = "1.0.0",
) -> FastAPI:
    schema.ensure_valid()
    request_model = build_request_model(schema, f"{title.replace(' ', '')}Request")
    state = RuntimeState(model=None, metadata={}, shadow_model=None, shadow_metadata={})

    app = FastAPI(title=title, description=description, version=version)

    @app.on_event("startup")
    def startup_event() -> None:
        primary_model_path = artifact_dir / model_filename
        primary_metadata_path = artifact_dir / metadata_filename

        shadow_model_path = artifact_dir / shadow_model_filename if shadow_model_filename else None
        shadow_metadata_path = artifact_dir / shadow_metadata_filename if shadow_metadata_filename else None

        if registry_dir is not None and registry_model_name is not None:
            alias_paths = resolve_alias_artifact_paths(
                registry_dir=registry_dir,
                model_name=registry_model_name,
                model_filename=model_filename,
                metadata_filename=metadata_filename,
                primary_alias=primary_alias,
                shadow_alias=shadow_alias,
            )

            if alias_paths["primary_model_path"] is not None and alias_paths["primary_metadata_path"] is not None:
                primary_model_path = alias_paths["primary_model_path"]
                primary_metadata_path = alias_paths["primary_metadata_path"]

            if alias_paths["shadow_model_path"] is not None and alias_paths["shadow_metadata_path"] is not None:
                shadow_model_path = alias_paths["shadow_model_path"]
                shadow_metadata_path = alias_paths["shadow_metadata_path"]

        state.model = None
        state.metadata = {}
        if primary_model_path.exists() and primary_metadata_path.exists():
            state.model, state.metadata = _load_model_with_metadata(primary_model_path, primary_metadata_path)

        state.shadow_model = None
        state.shadow_metadata = {}
        if (
            shadow_model_path is not None
            and shadow_metadata_path is not None
            and shadow_model_path.exists()
            and shadow_metadata_path.exists()
        ):
            state.shadow_model, state.shadow_metadata = _load_model_with_metadata(
                shadow_model_path,
                shadow_metadata_path,
            )

    def _validated_request_frame(payload: Any) -> tuple[dict[str, Any], pd.DataFrame]:
        payload_dict = payload.model_dump()
        ordered = {name: payload_dict[name] for name in schema.all_features}
        frame = pd.DataFrame([ordered])
        try:
            frame = validate_frame(
                frame=frame,
                schema=schema,
                require_target=False,
                context="serve_pre_predict",
                allow_extra_columns=False,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return ordered, frame

    @app.get("/health")
    def health() -> dict[str, Any]:
        status = "ok" if state.model is not None else "model_not_loaded"
        return {
            "status": status,
            "model_name": state.metadata.get("best_model_name"),
            "target_name": state.metadata.get("target_name", schema.target_name),
            "feature_count": len(schema.all_features),
            "shadow_model_loaded": state.shadow_model is not None,
        }

    @app.post("/predict", response_model=PredictionResponse)
    def predict(payload: request_model) -> PredictionResponse:
        if state.model is None:
            raise HTTPException(status_code=503, detail="Model artifact is not available.")

        ordered, frame = _validated_request_frame(payload)

        primary_prediction = float(state.model.predict(frame)[0])
        interval = _compute_prediction_interval(
            model=state.model,
            metadata=state.metadata,
            frame=frame,
            prediction=primary_prediction,
        )
        p10 = interval[0] if interval is not None else None
        p90 = interval[1] if interval is not None else None
        interval_method = interval[2] if interval is not None else None

        if state.shadow_model is not None and shadow_log_path is not None:
            try:
                shadow_prediction = float(state.shadow_model.predict(frame)[0])
                _append_shadow_log(
                    shadow_log_path=shadow_log_path,
                    features=ordered,
                    primary_prediction=primary_prediction,
                    shadow_prediction=shadow_prediction,
                    primary_model_name=str(state.metadata.get("best_model_name", "unknown")),
                    shadow_model_name=str(state.shadow_metadata.get("best_model_name", "unknown")),
                )
            except Exception:
                pass

        return PredictionResponse(
            prediction=primary_prediction,
            y_hat=primary_prediction,
            p10=p10,
            p90=p90,
            interval_method=interval_method,
            model_name=str(state.metadata.get("best_model_name", "unknown")),
            target_name=str(state.metadata.get("target_name", schema.target_name)),
            input_features=ordered,
        )

    @app.post("/explain", response_model=ExplainResponse)
    def explain(payload: request_model) -> ExplainResponse:
        if state.model is None:
            raise HTTPException(status_code=503, detail="Model artifact is not available.")

        ordered, frame = _validated_request_frame(payload)
        prediction = float(state.model.predict(frame)[0])
        explanation = explain_single_prediction(
            pipeline=state.model,
            frame=frame,
            schema=schema,
        )

        contributions_raw = explanation.get("contributions")
        contributions: dict[str, float]
        if isinstance(contributions_raw, dict):
            contributions = {str(name): float(value) for name, value in contributions_raw.items()}
        else:
            contributions = {}

        method_raw = explanation.get("method")
        reason_raw = explanation.get("reason")
        base_value_raw = explanation.get("base_value")

        return ExplainResponse(
            available=bool(explanation.get("available", False)),
            reason=str(reason_raw) if reason_raw is not None else None,
            method=str(method_raw) if method_raw is not None else None,
            prediction=prediction,
            y_hat=prediction,
            base_value=float(base_value_raw) if base_value_raw is not None else None,
            contributions=contributions,
            model_name=str(state.metadata.get("best_model_name", "unknown")),
            target_name=str(state.metadata.get("target_name", schema.target_name)),
            input_features=ordered,
        )

    return app
