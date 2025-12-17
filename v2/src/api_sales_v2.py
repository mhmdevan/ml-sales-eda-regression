"""
api_sales_v2.py

FastAPI service for the v2 Sales Regression project.

It loads the best regression Pipeline trained by `train_sales_regression_mlflow.py`
and exposes a simple HTTP API:

    GET  /health   -> basic health check + model metadata
    POST /predict  -> predict SALES from input features

Run locally:

    uvicorn src.api_sales_v2:app --reload

Then visit: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .logging_utils import get_logger

logger = get_logger(__name__)

# -------------------------------------------------------------------
# Paths / global state
# -------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
MODELS_DIR: Path = PROJECT_ROOT / "models"
MODEL_PATH: Path = MODELS_DIR / "sales_regressor.joblib"
METADATA_PATH: Path = MODELS_DIR / "sales_regressor_metadata.json"

# Hard-coded fallback feature list (same as in predict_sales.py)
DEFAULT_FEATURES: List[str] = [
    "QUANTITYORDERED",
    "PRICEEACH",
    "ORDERLINENUMBER",
    "MSRP",
    "QTR_ID",
    "MONTH_ID",
    "YEAR_ID",
    "PRODUCTLINE",
    "COUNTRY",
    "DEALSIZE",
]


# Global variables for the loaded model + metadata
model_pipeline: Optional[Any] = None
model_metadata: Dict[str, Any] = {}
feature_order: List[str] = []


# -------------------------------------------------------------------
# Pydantic schema for request/response
# -------------------------------------------------------------------


class SalesFeatures(BaseModel):
    """
    Input schema for a single prediction request.

    Fields intentionally match the original column names from the dataset so that:
      - mapping to pandas.DataFrame is straightforward
      - they align with the `features` list in metadata
    """

    QUANTITYORDERED: int = Field(..., description="Number of units ordered in this line item.")
    PRICEEACH: float = Field(..., description="Unit price per item (USD).")
    ORDERLINENUMBER: int = Field(..., description="Line number of the item within the order.")
    MSRP: float = Field(..., description="Manufacturer's suggested retail price.")
    QTR_ID: int = Field(..., description="Quarter ID, e.g., 1-4.")
    MONTH_ID: int = Field(..., description="Month ID, e.g., 1-12.")
    YEAR_ID: int = Field(..., description="Year ID, e.g., 2003, 2004.")
    PRODUCTLINE: str = Field(..., description="Product line category, e.g., 'Classic Cars'.")
    COUNTRY: str = Field(..., description="Customer country, e.g., 'USA'.")
    DEALSIZE: str = Field(..., description="Deal size bucket, e.g., 'Small', 'Medium', 'Large'.")


class PredictionResponse(BaseModel):
    """
    Output schema for /predict endpoint.
    """

    prediction: float = Field(..., description="Predicted SALES value (continuous).")
    currency: str = Field("USD", description="Currency of the SALES value.")
    model_name: Optional[str] = Field(None, description="Name of the underlying model class.")
    mlflow_run_id: Optional[str] = Field(
        None, description="MLflow run id associated with this trained model."
    )
    features_used: Dict[str, Any] = Field(
        ..., description="Echo of features used for this prediction (after re-ordering)."
    )


# -------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------


def _load_model_and_metadata() -> None:
    """
    Load model pipeline and metadata into global variables.

    This is called on FastAPI startup. If anything fails, we log the error and
    leave `model_pipeline` as None, so /health and /predict can reflect the issue.
    """
    global model_pipeline, model_metadata, feature_order

    if not MODEL_PATH.exists():
        logger.error(
            f"[STARTUP] Model file not found at {MODEL_PATH}. "
            "Run `python -m src.train_sales_regression_mlflow` first."
        )
        model_pipeline = None
        model_metadata = {}
        feature_order = list(DEFAULT_FEATURES)
        return

    logger.info(f"[STARTUP] Loading model pipeline from {MODEL_PATH}")
    try:
        model_pipeline = joblib.load(MODEL_PATH)
    except Exception as e:  # noqa: BLE001
        logger.exception(f"[STARTUP] Failed to load model pipeline: {e}")
        model_pipeline = None
        model_metadata = {}
        feature_order = list(DEFAULT_FEATURES)
        return

    # Load metadata if available
    if METADATA_PATH.exists():
        logger.info(f"[STARTUP] Loading model metadata from {METADATA_PATH}")
        try:
            import json

            with METADATA_PATH.open("r", encoding="utf-8") as f:
                model_metadata = json.load(f)
        except Exception as e:  # noqa: BLE001
            logger.exception(f"[STARTUP] Failed to load metadata JSON: {e}")
            model_metadata = {}
    else:
        logger.warning(
            f"[STARTUP] Metadata file not found at {METADATA_PATH}; "
            "falling back to DEFAULT_FEATURES only."
        )
        model_metadata = {}

    # Resolve feature list
    if "features" in model_metadata and isinstance(model_metadata["features"], list):
        feature_order = list(model_metadata["features"])
        logger.info(f"[STARTUP] Using feature list from metadata: {feature_order}")
    else:
        feature_order = list(DEFAULT_FEATURES)
        logger.warning(
            "[STARTUP] 'features' missing in metadata; using DEFAULT_FEATURES. "
            f"Features: {feature_order}"
        )


def _build_features_dataframe(payload: SalesFeatures) -> pd.DataFrame:
    """
    Convert the Pydantic model into a pandas DataFrame with the exact
    feature order expected by the sklearn Pipeline.
    """
    if not feature_order:
        raise RuntimeError("feature_order is empty; model not properly initialized.")

    data_dict = payload.dict()
    # Fail fast if any expected feature is missing (should not happen with Pydantic)
    missing = set(feature_order) - set(data_dict.keys())
    if missing:
        raise KeyError(
            f"Missing required features in payload: {sorted(missing)}. "
            f"Payload keys: {sorted(data_dict.keys())}"
        )

    # Reorder according to feature_order
    ordered = {name: data_dict[name] for name in feature_order}
    df = pd.DataFrame([ordered])
    return df


# -------------------------------------------------------------------
# FastAPI app definition
# -------------------------------------------------------------------

app = FastAPI(
    title="Sales Regression API v2",
    description=(
        "FastAPI serving layer for the Sales EDA & Regression v2 project.\n\n"
        "This service wraps the best regression model (RandomForest / GBM / LinearRegression)\n"
        "trained with `train_sales_regression_mlflow.py` on Parquet/DuckDB-based sales data."
    ),
    version="2.0.0",
)


@app.on_event("startup")
def on_startup() -> None:
    """
    Lifecycle event: called once when the FastAPI app starts.

    We use it to load the model + metadata into memory so each request
    can be served quickly without re-loading from disk.
    """
    logger.info("[FASTAPI] Startup: loading model and metadata...")
    _load_model_and_metadata()
    if model_pipeline is not None:
        logger.info("[FASTAPI] Model loaded successfully.")
    else:
        logger.warning("[FASTAPI] Model is NOT loaded; /predict will return 503.")


# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------


@app.get("/health")
def health() -> Dict[str, Any]:
    """
    Basic health check endpoint.

    Returns:
      - status: "ok" if model is loaded, otherwise "model_not_loaded"
      - model_name, mlflow_run_id if available
      - feature_count and feature_list (first N features only for readability)
    """
    if model_pipeline is None:
        status = "model_not_loaded"
    else:
        status = "ok"

    best_model_name = model_metadata.get("best_model_name")
    best_run_id = model_metadata.get("best_run_id")

    return {
        "status": status,
        "model_name": best_model_name,
        "mlflow_run_id": best_run_id,
        "feature_count": len(feature_order),
        "feature_list_preview": feature_order[:10],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: SalesFeatures) -> PredictionResponse:
    """
    Predict SALES (continuous) from a single record of sales features.

    Raises:
      - 503 if the model is not loaded
      - 500 if something unexpected goes wrong in the pipeline
    """
    if model_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. "
            "Make sure training has been run and the model file exists.",
        )

    try:
        X = _build_features_dataframe(features)
        y_pred = model_pipeline.predict(X)
        pred_value = float(y_pred[0])
    except HTTPException:
        # re-raise FastAPI HTTP exceptions directly
        raise
    except Exception as e:  # noqa: BLE001
        logger.exception(f"[PREDICT] Unexpected error during prediction: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal prediction error: {e}",
        ) from e

    best_model_name = model_metadata.get("best_model_name")
    best_run_id = model_metadata.get("best_run_id")

    return PredictionResponse(
        prediction=pred_value,
        currency="USD",
        model_name=best_model_name,
        mlflow_run_id=best_run_id,
        features_used=features.dict(),
    )
