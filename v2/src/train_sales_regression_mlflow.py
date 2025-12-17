"""
train_sales_regression_mlflow.py

v2 training script for the Sales EDA project:

- Reads cleaned sales data from Parquet (produced by data_parquet.py)
- Builds features (numeric + categorical) for regression on SALES
- Trains several models (LinearRegression, RandomForest, GradientBoosting)
- Logs params + metrics + artifacts to MLflow (WITH input_example to avoid warnings)
- Saves the best model locally under models/sales_regressor.joblib
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple

import json
import math

import mlflow
import mlflow.sklearn
import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
import yaml

from .config_v2 import load_project_config
from .logging_utils import get_logger

logger = get_logger(__name__)


# -------------------------------------------------------------------
# 1. Small dataclasses for ML/MLflow config (read from YAML directly)
# -------------------------------------------------------------------


@dataclass
class MLConfig:
    target_column: str
    numeric_features: List[str]
    categorical_features: List[str]


@dataclass
class MLflowConfig:
    tracking_uri: str
    experiment_name: str


def _load_ml_configs(config_path: Path) -> Tuple[MLConfig, MLflowConfig]:
    """
    Read ml + mlflow sections directly from YAML.

    We don't force config_v2.ProjectConfig to know about ML;
    this keeps v2 paths/cleaning decoupled from training details.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config YAML not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    if "ml" not in raw_cfg:
        raise KeyError("Missing 'ml' section in config_v2.yaml")
    if "mlflow" not in raw_cfg:
        raise KeyError("Missing 'mlflow' section in config_v2.yaml")

    ml_section = raw_cfg["ml"]
    mlflow_section = raw_cfg["mlflow"]

    ml_cfg = MLConfig(
        target_column=str(ml_section["target_column"]),
        numeric_features=list(ml_section.get("numeric_features", [])),
        categorical_features=list(ml_section.get("categorical_features", [])),
    )

    mlflow_cfg = MLflowConfig(
        tracking_uri=str(mlflow_section.get("tracking_uri", "file:mlruns")),
        experiment_name=str(mlflow_section.get("experiment_name", "sales_regression_v2")),
    )

    if not ml_cfg.numeric_features and not ml_cfg.categorical_features:
        raise ValueError("MLConfig has no features defined (both numeric and categorical are empty).")

    return ml_cfg, mlflow_cfg


# -------------------------------------------------------------------
# 2. Data loading from Parquet (v2 layer)
# -------------------------------------------------------------------


def load_sales_from_parquet(parquet_dir: Path) -> pl.DataFrame:
    """
    Load cleaned sales data from Parquet (v2 data layer).

    We read all partition files, e.g. sales_YEAR=2003.parquet etc.
    """
    if not parquet_dir.exists():
        raise FileNotFoundError(f"Processed parquet dir does not exist: {parquet_dir}")

    glob_pattern = parquet_dir / "sales_*.parquet"
    logger.info(f"[DATA] Reading cleaned sales from parquet glob: {glob_pattern}")

    # Use lazy scan then collect to avoid memory spikes (on larger data).
    df = pl.scan_parquet(str(glob_pattern)).collect()
    logger.info(f"[DATA] Loaded cleaned sales from parquet: {df.shape[0]} rows × {df.shape[1]} columns")

    return df


def polars_to_sklearn_matrices(
    df: pl.DataFrame,
    ml_cfg: MLConfig,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Convert Polars DataFrame into (X, y) matrices for scikit-learn.

    - Checks all requested feature columns exist
    - Handles basic type coercions for numeric/categorical
    """
    missing = []
    required_cols = [ml_cfg.target_column] + ml_cfg.numeric_features + ml_cfg.categorical_features
    for col in required_cols:
        if col not in df.columns:
            missing.append(col)

    if missing:
        raise KeyError(f"The following required columns are missing from the cleaned data: {missing}")

    # Convert to pandas for scikit-learn
    df_pd = df.to_pandas()

    # Ensure numeric columns are numeric
    for col in ml_cfg.numeric_features:
        df_pd[col] = df_pd[col].astype(float)

    # Categorical columns: cast to string
    for col in ml_cfg.categorical_features:
        df_pd[col] = df_pd[col].astype(str)

    y = df_pd[ml_cfg.target_column].astype(float).values

    feature_cols = ml_cfg.numeric_features + ml_cfg.categorical_features
    X = df_pd[feature_cols].values  # matrix for ColumnTransformer

    logger.info(
        f"[DATA] Built X, y matrices for sklearn: X.shape={X.shape}, y.shape={y.shape}, "
        f"features={feature_cols}, target={ml_cfg.target_column}"
    )
    return X, y, ml_cfg.numeric_features, ml_cfg.categorical_features


# -------------------------------------------------------------------
# 3. Model definitions
# -------------------------------------------------------------------


def get_model_search_space() -> List[Dict[str, Any]]:
    """
    Return a list of model configs; each item describes:
      - model_name
      - estimator class
      - param_grid: list of dicts with hyperparameters

    We keep this intentionally small to avoid overengineering,
    but enough to show "I tried several models and hyperparams".
    """
    search_space: List[Dict[str, Any]] = []

    # 1) LinearRegression as a simple baseline
    search_space.append(
        {
            "model_name": "LinearRegression",
            "estimator": LinearRegression,
            "param_grid": [
                # No hyperparams to tune here; we still keep a "dummy" dict
                {},
            ],
        }
    )

    # 2) RandomForestRegressor with a tiny grid
    search_space.append(
        {
            "model_name": "RandomForestRegressor",
            "estimator": RandomForestRegressor,
            "param_grid": [
                {
                    "n_estimators": 100,
                    "max_depth": None,
                },
                {
                    "n_estimators": 300,
                    "max_depth": 10,
                },
            ],
        }
    )

    # 3) GradientBoostingRegressor as a boosting baseline
    search_space.append(
        {
            "model_name": "GradientBoostingRegressor",
            "estimator": GradientBoostingRegressor,
            "param_grid": [
                {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 3,
                },
                {
                    "n_estimators": 200,
                    "learning_rate": 0.05,
                    "max_depth": 3,
                },
            ],
        }
    )

    return search_space


# -------------------------------------------------------------------
# 4. MLflow helpers
# -------------------------------------------------------------------


def setup_mlflow(mlflow_cfg: MLflowConfig) -> None:
    """
    Configure MLflow tracking URI and experiment name.
    """
    mlflow.set_tracking_uri(mlflow_cfg.tracking_uri)
    mlflow.set_experiment(mlflow_cfg.experiment_name)
    logger.info(
        f"[MLFLOW] Tracking URI set to {mlflow_cfg.tracking_uri}, "
        f"experiment='{mlflow_cfg.experiment_name}'"
    )


def log_run_to_mlflow(
    model_name: str,
    base_params: Dict[str, Any],
    regressor_params: Dict[str, Any],
    metrics: Dict[str, float],
    pipeline: Pipeline,
    X_example: np.ndarray,
) -> str:
    """
    Start an MLflow run, log params/metrics, and log the sklearn model.

    We explicitly pass `input_example` so MLflow can infer the model
    signature and does NOT spam:

      "Model logged without a signature and input example..."

    Returns the run_id.
    """
    # Use a small slice (e.g. first 5 samples) as input_example
    # to keep artifacts light.
    if X_example.shape[0] > 5:
        input_example = X_example[:5]
    else:
        input_example = X_example

    with mlflow.start_run(run_name=f"{model_name}") as run:
        # Log parameters
        mlflow.log_param("model_name", model_name)
        for key, value in base_params.items():
            mlflow.log_param(key, value)
        for key, value in regressor_params.items():
            mlflow.log_param(key, value)

        # Log metrics
        for key, value in metrics.items():
            mlflow.log_metric(key, float(value))

        # Log the entire sklearn pipeline as an artifact WITH input_example
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            input_example=input_example,
        )

        run_id = run.info.run_id
        logger.info(f"[MLFLOW] Logged run_id={run_id} for model={model_name}")
        return run_id


# -------------------------------------------------------------------
# 5. Training core
# -------------------------------------------------------------------


def train_models_with_mlflow(
    X: np.ndarray,
    y: np.ndarray,
    numeric_features: List[str],
    categorical_features: List[str],
    models_def: List[Dict[str, Any]],
    random_state: int = 42,
    test_size: float = 0.2,
) -> Dict[str, Any]:
    """
    Train several models, log them to MLflow, and return info about the best one.

    Returns a dict:
      {
        "best_model_name": ...,
        "best_rmse": ...,
        "best_pipeline": sklearn Pipeline,
        "best_run_id": ...,
        "best_metrics": {...},
        "features": [...],           # original feature names (num + cat)
      }
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    logger.info(
        f"[SPLIT] Train size={X_train.shape[0]}, Test size={X_test.shape[0]}, "
        f"test_size={test_size}"
    )

    # Preprocessing pipeline: StandardScaler for numeric, OneHotEncoder for categorical
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), list(range(len(numeric_features)))),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                list(
                    range(
                        len(numeric_features),
                        len(numeric_features) + len(categorical_features),
                    )
                ),
            ),
        ]
    )

    best_rmse = math.inf
    best_info: Dict[str, Any] = {
        "best_model_name": None,
        "best_rmse": None,
        "best_pipeline": None,
        "best_run_id": None,
        "best_metrics": None,
        "features": numeric_features + categorical_features,
    }

    # Base params to log for each run
    base_params = {
        "random_state": random_state,
        "test_size": test_size,
        "n_train_samples": int(X_train.shape[0]),
        "n_test_samples": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "n_numeric_features": len(numeric_features),
        "n_categorical_features": len(categorical_features),
    }

    # We'll reuse the same X_train as source for input_example
    X_example = X_train

    for model_def in models_def:
        model_name = model_def["model_name"]
        estimator_cls = model_def["estimator"]
        param_grid = model_def["param_grid"]

        for reg_params in param_grid:
            # Build estimator with given hyperparameters
            # Some estimators support random_state, some don't.
            base_estimator = estimator_cls()
            est_params = base_estimator.get_params()

            if "random_state" in est_params:
                estimator = estimator_cls(random_state=random_state, **reg_params)
            else:
                estimator = estimator_cls(**reg_params)

            full_pipeline = Pipeline(
                steps=[
                    ("preprocess", preprocessor),
                    ("regressor", estimator),
                ]
            )

            logger.info(
                f"[TRAIN] Training model={model_name} with params={reg_params}"
            )
            full_pipeline.fit(X_train, y_train)

            # Evaluate on test set
            y_pred = full_pipeline.predict(X_test)

            rmse = math.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            metrics = {
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
            }

            logger.info(
                f"[EVAL] model={model_name}, params={reg_params} | "
                f"RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}"
            )

            # Log to MLflow (with input_example to avoid warning)
            run_id = log_run_to_mlflow(
                model_name=model_name,
                base_params=base_params,
                regressor_params=reg_params,
                metrics=metrics,
                pipeline=full_pipeline,
                X_example=X_example,
            )

            # Track best model by RMSE (lower is better)
            if rmse < best_rmse:
                best_rmse = rmse
                best_info = {
                    "best_model_name": model_name,
                    "best_rmse": rmse,
                    "best_pipeline": full_pipeline,
                    "best_run_id": run_id,
                    "best_metrics": metrics,
                    "features": numeric_features + categorical_features,
                }

    logger.info(
        f"[SELECT] Best model={best_info['best_model_name']} "
        f"with RMSE={best_info['best_rmse']:.4f}, run_id={best_info['best_run_id']}"
    )

    return best_info


# -------------------------------------------------------------------
# 6. Saving best model locally
# -------------------------------------------------------------------


def save_best_model(
    project_root: Path,
    best_info: Dict[str, Any],
) -> None:
    """
    Save the best sklearn pipeline + metadata in models/ directory.

    Metadata now also includes:
      - "features": original feature names used for training (num + cat)
    so that inference code (CLI / FastAPI) can read it instead of
    falling back to hard-coded lists.
    """
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "sales_regressor.joblib"
    metadata_path = models_dir / "sales_regressor_metadata.json"

    pipeline = best_info["best_pipeline"]
    if pipeline is None:
        raise ValueError("No best pipeline found; training may have failed.")

    joblib.dump(pipeline, model_path)

    metadata = {
        "best_model_name": best_info["best_model_name"],
        "best_rmse": best_info["best_rmse"],
        "best_metrics": best_info["best_metrics"],
        "best_run_id": best_info["best_run_id"],
        "model_path": str(model_path),
        # NEW: store feature names
        "features": best_info.get("features", []),
    }

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"[SAVE] Saved best model pipeline to {model_path}")
    logger.info(f"[SAVE] Saved model metadata to {metadata_path}")


# -------------------------------------------------------------------
# 7. Orchestration
# -------------------------------------------------------------------


def run_training_pipeline(config_path: str | Path = "config_v2.yaml") -> None:
    """
    End-to-end v2 training pipeline:

    1. Load project config (paths etc.) + ml/mlflow config from YAML
    2. Read cleaned sales data from v2 parquet
    3. Build (X, y) matrices
    4. Train several models, log runs to MLflow (with input_example)
    5. Save the best model locally (with feature list in metadata)
    """
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / config_path

    # v2 config for paths + cleaning/parquet
    config = load_project_config(config_path)

    # ML + MLflow config from YAML
    ml_cfg, mlflow_cfg = _load_ml_configs(config_path)

    # Load data from Parquet
    df = load_sales_from_parquet(config.paths.processed_parquet_dir)

    # Build X, y, feature definitions
    X, y, num_feats, cat_feats = polars_to_sklearn_matrices(df, ml_cfg)

    # Setup MLflow tracking
    setup_mlflow(mlflow_cfg)

    # Model search space
    models_def = get_model_search_space()

    # Train + log
    best_info = train_models_with_mlflow(
        X=X,
        y=y,
        numeric_features=num_feats,
        categorical_features=cat_feats,
        models_def=models_def,
        random_state=42,
        test_size=0.2,
    )

    # Save best model locally
    save_best_model(project_root, best_info)

    logger.info("[DONE] v2 training pipeline with MLflow completed.")


if __name__ == "__main__":
    run_training_pipeline()
