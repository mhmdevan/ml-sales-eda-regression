"""
predict_sales.py

Simple CLI script to run inference with the best regression model trained
in `train_sales_regression_mlflow.py`.

Usage examples:

    # Show required schema and an example payload
    python -m src.predict_sales --show-schema

    # Predict from inline JSON (double-quoted keys/values!)
    python -m src.predict_sales --json '{"QUANTITYORDERED": 30, "PRICEEACH": 95.7, "ORDERLINENUMBER": 3, "MSRP": 120.0, "QTR_ID": 3, "MONTH_ID": 7, "YEAR_ID": 2004, "PRODUCTLINE": "Classic Cars", "COUNTRY": "USA", "DEALSIZE": "Medium"}'

    # Predict from JSON file
    python -m src.predict_sales --json-file path/to/features.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd

# Try to import the same logger utilities you already use
from .logging_utils import get_logger

logger = get_logger(__name__)

# -------------------------------------------------------------------
# Project paths & defaults
# -------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
MODELS_DIR: Path = PROJECT_ROOT / "models"
MODEL_PATH: Path = MODELS_DIR / "sales_regressor.joblib"
METADATA_PATH: Path = MODELS_DIR / "sales_regressor_metadata.json"

# Hard-coded fallback feature list (matches your training log)
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


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def load_model_and_metadata() -> tuple[Any, Dict[str, Any]]:
    """
    Load the best model pipeline (sklearn Pipeline) and metadata.

    The model is expected to be a full sklearn Pipeline that knows how to:
      - preprocess numeric & categorical features
      - run the regression model

    Metadata is expected to be a JSON dict created by the training script, with
    at least:
      - "features": list of feature names (column order)
      - "target": target column name ("SALES")
      - "best_model_name": model class name, e.g. "RandomForestRegressor"
      - "best_run_id": MLflow run id (optional but nice to have)
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Make sure you have run `python -m src.train_sales_regression_mlflow` first."
        )

    logger.info(f"[LOAD] Loading model pipeline from {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    metadata: Dict[str, Any] = {}
    if METADATA_PATH.exists():
        logger.info(f"[LOAD] Loading model metadata from {METADATA_PATH}")
        with METADATA_PATH.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        logger.warning(
            f"[LOAD] Model metadata not found at {METADATA_PATH}. "
            "Falling back to hard-coded feature list."
        )

    return model, metadata


def resolve_feature_list(metadata: Dict[str, Any]) -> List[str]:
    """
    Resolve the list of input features in the correct order.

    Priority:
      1. metadata["features"] if present
      2. DEFAULT_FEATURES fallback
    """
    if "features" in metadata and isinstance(metadata["features"], list):
        features = list(metadata["features"])
        logger.info(f"[META] Using feature list from metadata: {features}")
        return features

    logger.warning("[META] 'features' missing in metadata; using DEFAULT_FEATURES.")
    return list(DEFAULT_FEATURES)


def parse_features_from_json_str(json_str: str) -> Dict[str, Any]:
    """
    Parse inline JSON string into a Python dict.

    Raises ValueError on invalid JSON.
    """
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON string: {e}") from e

    if not isinstance(data, dict):
        raise ValueError("Top-level JSON must be an object with feature: value pairs.")

    return data


def parse_features_from_file(path: Path) -> Dict[str, Any]:
    """
    Load a JSON file containing one feature dict.
    """
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("JSON file must contain a single object (feature: value pairs).")

    return data


def build_features_dataframe(
    features_payload: Dict[str, Any],
    feature_order: List[str],
) -> pd.DataFrame:
    """
    Build a one-row pandas DataFrame in the exact feature order expected by the model.

    - Missing features → error (explicit, fail-fast).
    - Extra keys → ignored with a warning.
    """
    payload_keys = set(features_payload.keys())
    expected_keys = set(feature_order)

    missing = expected_keys - payload_keys
    extra = payload_keys - expected_keys

    if missing:
        raise KeyError(
            f"Missing required features: {sorted(missing)}. "
            f"Payload keys: {sorted(payload_keys)}"
        )

    if extra:
        logger.warning(
            f"[PREDICT] Ignoring extra keys not used by the model: {sorted(extra)}"
        )

    # Reorder according to feature_order
    ordered_data = {feat: features_payload[feat] for feat in feature_order}
    df = pd.DataFrame([ordered_data])
    return df


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single prediction with the trained sales regression model.",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help=(
            "Inline JSON string with all required features. "
            'Example: \'{"QUANTITYORDERED": 30, "PRICEEACH": 95.7, ...}\''
        ),
    )
    parser.add_argument(
        "--json-file",
        type=str,
        default=None,
        help="Path to a JSON file containing one object with the input features.",
    )
    parser.add_argument(
        "--show-schema",
        action="store_true",
        help="Print the required input schema and an example payload, then exit.",
    )
    return parser.parse_args()


def print_schema_example(feature_order: List[str]) -> None:
    """
    Print the expected feature list + a concrete JSON example the user can copy.
    """
    print("\n[SCHEMA] Required features (column order):")
    for name in feature_order:
        print(f"  - {name}")

    example_payload = {
        "QUANTITYORDERED": 30,
        "PRICEEACH": 95.7,
        "ORDERLINENUMBER": 3,
        "MSRP": 120.0,
        "QTR_ID": 3,
        "MONTH_ID": 7,
        "YEAR_ID": 2004,
        "PRODUCTLINE": "Classic Cars",
        "COUNTRY": "USA",
        "DEALSIZE": "Medium",
    }

    print("\n[SCHEMA] Example JSON payload:")
    print(json.dumps(example_payload, indent=2))


def main() -> None:
    args = parse_args()

    # Load model + metadata
    model, metadata = load_model_and_metadata()
    feature_order = resolve_feature_list(metadata)

    if args.show_schema:
        print_schema_example(feature_order)
        return

    if args.json is None and args.json_file is None:
        print(
            "[ERROR] You must provide either --json or --json-file.\n"
            "Use --show-schema to see required fields and an example payload."
        )
        return

    # Build feature dict from selected source
    if args.json is not None:
        features_payload = parse_features_from_json_str(args.json)
        source = "inline JSON"
    else:
        json_path = Path(args.json_file)
        features_payload = parse_features_from_file(json_path)
        source = f"file: {json_path}"

    logger.info(f"[PREDICT] Received features from {source}")

    # Build DataFrame in correct order
    X = build_features_dataframe(features_payload, feature_order)

    # Run model prediction
    import numpy as np

    y_pred = model.predict(X)
    # For regressors, y_pred is a 1D array
    pred_value = float(y_pred[0])

    # Optional: if model exposes prediction intervals / variance, you could add it here.

    print()
    print("[RESULT]")
    print(f"  Predicted SALES: {pred_value:,.2f}")
    print()
    print("[DEBUG] Input features used (ordered):")
    print(X.to_string(index=False))

    # Print a bit of metadata for traceability
    best_model_name = metadata.get("best_model_name")
    best_run_id = metadata.get("best_run_id")
    print()
    print("[MODEL INFO]")
    if best_model_name:
        print(f"  best_model_name: {best_model_name}")
    if best_run_id:
        print(f"  mlflow_run_id:   {best_run_id}")
    print(f"  model_path:      {MODEL_PATH}")


if __name__ == "__main__":
    main()
