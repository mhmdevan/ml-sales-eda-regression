"""
sales_api.py

FastAPI prediction service for the trained sales regression model.

Uses the best model saved under:
    output/regression/sales_regression_best_*.joblib

Request expects raw business features (QUANTITYORDERED, PRICEEACH, MSRP, ORDERDATE,
PRODUCTLINE, DEALSIZE, COUNTRY). The API reuses the same feature engineering as the
training pipeline (add_date_features + create_regression_features).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sales_eda import PROJECT_ROOT, add_date_features
from sales_regression import create_regression_features

MODELS_DIR = PROJECT_ROOT / "output" / "regression"

app = FastAPI(
    title="Sales Regression API",
    version="0.1.0",
    description="Prediction API on top of the trained sales regression pipeline.",
)

model = None
model_path: Optional[Path] = None


def load_best_model() -> None:
    """
    Load the best regression model from the output/regression directory.
    """
    global model, model_path

    candidates = sorted(MODELS_DIR.glob("sales_regression_best_*.joblib"))
    if not candidates:
        raise FileNotFoundError(
            f"No model file matching 'sales_regression_best_*.joblib' found under {MODELS_DIR}"
        )

    # Pick the last candidate (lexicographically) as "best"
    model_path = candidates[-1]
    model = joblib.load(model_path)
    print(f"[API] Loaded model from {model_path}")


class SalesPredictionInput(BaseModel):
    QUANTITYORDERED: int
    PRICEEACH: float
    MSRP: float
    ORDERDATE: date
    PRODUCTLINE: str
    DEALSIZE: str
    COUNTRY: str


class SalesPredictionOutput(BaseModel):
    prediction: float
    currency: str = "USD"
    model_name: str
    model_path: str


@app.on_event("startup")
def startup_event() -> None:
    try:
        load_best_model()
    except Exception as exc:
        print(f"[API ERROR] Could not load model on startup: {exc}")


@app.post("/predict", response_model=SalesPredictionOutput)
def predict(payload: SalesPredictionInput) -> SalesPredictionOutput:
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    # Build a one-row DataFrame with raw features
    df = pd.DataFrame(
        [
            {
                "QUANTITYORDERED": payload.QUANTITYORDERED,
                "PRICEEACH": payload.PRICEEACH,
                "MSRP": payload.MSRP,
                "ORDERDATE": payload.ORDERDATE,
                "PRODUCTLINE": payload.PRODUCTLINE,
                "DEALSIZE": payload.DEALSIZE,
                "COUNTRY": payload.COUNTRY,
            }
        ]
    )

    # Reuse the same feature engineering as training:
    #  1) add YEAR, MONTH, etc.
    df_dates = add_date_features(df)
    #  2) add QUARTER, SEASON, ratio features, etc.
    df_features, _, _ = create_regression_features(df_dates)

    # The trained pipeline expects these feature columns and includes
    # the ColumnTransformer + model inside.
    y_pred = model.predict(df_features)[0]

    if hasattr(model, "named_steps") and "model" in model.named_steps:
        model_name = model.named_steps["model"].__class__.__name__
    else:
        model_name = model.__class__.__name__

    return SalesPredictionOutput(
        prediction=float(y_pred),
        model_name=model_name,
        model_path=str(model_path) if model_path is not None else "",
    )
