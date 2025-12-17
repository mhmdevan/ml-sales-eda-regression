"""
sales_regression.py

Baseline + tuned regression models on SALES using the same cleaning / IQR logic
from sales_eda.py.

Models:
- LinearRegression (with numeric + categorical features, in a Pipeline)
- RandomForestRegressor baseline
- RandomForestRegressor tuned with RandomizedSearchCV
- GradientBoostingRegressor baseline (boosting model)

Metrics:
- MSE, RMSE, MAE, R^2
"""

from __future__ import annotations

from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sales_eda import (
    PROJECT_ROOT,
    load_config,
    load_data,
    add_date_features,
    clean_missing_values,
    remove_outliers,
)


# -------------------------------------------------------------------------
# 1. Feature engineering helpers
# -------------------------------------------------------------------------


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features for regression:
      - YEAR (already added by add_date_features)
      - MONTH (already added)
      - QUARTER (1..4)
      - SEASON (categorical: 'winter', 'spring', 'summer', 'autumn')
    """
    result = df.copy()
    if "ORDERDATE" in result.columns and not np.issubdtype(
        result["ORDERDATE"].dtype, np.datetime64
    ):
        # Safety: ensure ORDERDATE is datetime
        result["ORDERDATE"] = pd.to_datetime(result["ORDERDATE"])

    if "YEAR" not in result.columns:
        result["YEAR"] = result["ORDERDATE"].dt.year
    if "MONTH" not in result.columns:
        result["MONTH"] = result["ORDERDATE"].dt.month

    result["QUARTER"] = result["ORDERDATE"].dt.quarter

    def month_to_season(m: int) -> str:
        if m in (12, 1, 2):
            return "winter"
        if m in (3, 4, 5):
            return "spring"
        if m in (6, 7, 8):
            return "summer"
        return "autumn"

    result["SEASON"] = result["MONTH"].apply(month_to_season)
    return result


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add numeric ratio features that make sense for regression.

    IMPORTANT:
    - We do NOT use `SALES / QUANTITYORDERED` as a feature for predicting SALES,
      because that would be target leakage.
    - Instead, we build ratios derived only from input features, e.g. PRICEEACH / MSRP.
    """
    result = df.copy()

    if "PRICEEACH" in result.columns and "MSRP" in result.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = result["PRICEEACH"] / result["MSRP"]
            ratio = ratio.replace([np.inf, -np.inf], np.nan)
        result["PRICE_TO_MSRP_RATIO"] = ratio.fillna(ratio.median())

    if "QUANTITYORDERED" in result.columns and "PRICEEACH" in result.columns:
        # LINE_TOTAL_APPROX ~ QUANTITY * PRICEEACH.
        # This is still a feature from inputs, not from SALES itself.
        result["LINE_TOTAL_APPROX"] = result["QUANTITYORDERED"] * result["PRICEEACH"]

    return result


def create_regression_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Take a cleaned DataFrame (after missing-values + outliers) and:
      - add time-based features,
      - add engineered ratio features,
      - define numeric + categorical feature columns for modeling.

    Returns:
        df_features: DataFrame with additional feature columns
        numeric_features: list of numeric feature column names
        categorical_features: list of categorical feature column names
    """
    df_features = add_time_features(df)
    df_features = add_ratio_features(df_features)

    numeric_features: List[str] = []
    for col in [
        "QUANTITYORDERED",
        "PRICEEACH",
        "MSRP",
        "YEAR",
        "MONTH",
        "QUARTER",
        "PRICE_TO_MSRP_RATIO",
        "LINE_TOTAL_APPROX",
    ]:
        if col in df_features.columns:
            numeric_features.append(col)

    categorical_features: List[str] = []
    for col in ["PRODUCTLINE", "DEALSIZE", "COUNTRY", "SEASON"]:
        if col in df_features.columns:
            categorical_features.append(col)

    return df_features, numeric_features, categorical_features


# -------------------------------------------------------------------------
# 2. Dataset preparation
# -------------------------------------------------------------------------


def build_regression_dataset(
    limit_rows: int | None = None,
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Load, clean, and prepare the regression dataset using the EDA pipeline.

    Steps:
      - load CSV using config.yaml
      - add date features
      - clean missing values
      - remove outliers using IQR
      - add feature engineering (time + ratios)
      - separate features (X) and target (y=SALES)
    """
    config = load_config()

    data_path = PROJECT_ROOT / config["data_path"]
    numeric_cols = config["numeric_columns"]
    cleaning_cfg = config["cleaning"]
    iqr_factor = float(config["outliers"]["iqr_factor"])

    df_raw = load_data(data_path)
    if limit_rows is not None:
        df_raw = df_raw.head(limit_rows)

    df_dates = add_date_features(df_raw)
    df_clean_missing = clean_missing_values(df_dates, cleaning_cfg=cleaning_cfg)
    df_clean = remove_outliers(
        df_clean_missing, numeric_cols=numeric_cols, factor=iqr_factor
    )

    if "SALES" not in df_clean.columns:
        raise KeyError("Expected target column 'SALES' is missing from cleaned data.")

    df_features, numeric_features, categorical_features = create_regression_features(
        df_clean
    )

    feature_cols = numeric_features + categorical_features
    X = df_features[feature_cols].copy()
    y = df_features["SALES"].astype(float)

    print(f"[REG] Number of samples after cleaning: {len(X)}")
    print(f"[REG] Numeric features: {numeric_features}")
    print(f"[REG] Categorical features: {categorical_features}")

    return X, y, numeric_features, categorical_features


# -------------------------------------------------------------------------
# 3. Model training & evaluation
# -------------------------------------------------------------------------


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
      - scales numeric features (StandardScaler)
      - one-hot encodes categorical features (OneHotEncoder)
    """
    transformers = []
    if numeric_features:
        transformers.append(("num", StandardScaler(), numeric_features))
    if categorical_features:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )
    return preprocessor


def evaluate_model(
    name: str,
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    """
    Compute regression metrics for a fitted model on test data.
    """
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(
        f"[REG] {name} – "
        f"MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.3f}"
    )

    return {
        "model": name,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def train_baseline_and_tuned_models(limit_rows: int | None = None) -> pd.DataFrame:
    """
    Train:
      - LinearRegression baseline
      - RandomForestRegressor baseline
      - GradientBoostingRegressor baseline (boosting)
      - RandomForestRegressor tuned with RandomizedSearchCV

    Returns:
        metrics_df: DataFrame with metrics for each model (indexed by model name).
    """
    config = load_config()
    output_dir = PROJECT_ROOT / config["output_dir"]

    # 1) Prepare dataset
    X, y, numeric_features, categorical_features = build_regression_dataset(
        limit_rows=limit_rows
    )

    # 2) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3) Preprocessor
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # 4) Define pipelines
    lin_pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LinearRegression()),
        ]
    )

    rf_pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,
                    max_depth=None,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    gbr_pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", GradientBoostingRegressor(random_state=42)),
        ]
    )

    # 5) Fit baseline models
    lin_pipeline.fit(X_train, y_train)
    rf_pipeline.fit(X_train, y_train)
    gbr_pipeline.fit(X_train, y_train)

    # 6) Evaluate baseline models
    rows = []
    rows.append(
        evaluate_model("LinearRegression_baseline", lin_pipeline, X_test, y_test)
    )
    rows.append(
        evaluate_model("RandomForest_baseline", rf_pipeline, X_test, y_test)
    )
    rows.append(
        evaluate_model("GradientBoosting_baseline", gbr_pipeline, X_test, y_test)
    )

    # 7) Tuned RandomForest with RandomizedSearchCV
    rf_tuned_pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                RandomForestRegressor(random_state=42, n_jobs=-1),
            ),
        ]
    )

    param_distributions = {
        "model__n_estimators": [100, 200, 300, 400],
        "model__max_depth": [None, 5, 10, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", None],
    }

    search = RandomizedSearchCV(
        estimator=rf_tuned_pipeline,
        param_distributions=param_distributions,
        n_iter=15,
        scoring="neg_root_mean_squared_error",
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    print("[REG] Starting RandomizedSearchCV for RandomForestRegressor...")
    search.fit(X_train, y_train)

    print(f"[REG] Best params from RandomizedSearchCV: {search.best_params_}")
    best_rf_pipeline = search.best_estimator_

    rows.append(
        evaluate_model(
            "RandomForest_tuned",
            best_rf_pipeline,
            X_test,
            y_test,
        )
    )

    # 8) Collect metrics
    metrics_df = pd.DataFrame(rows).set_index("model")

    # 9) Persist metrics & best model
    output_dir.mkdir(parents=True, exist_ok=True)
    reg_dir = output_dir / "regression"
    reg_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = reg_dir / "advanced_metrics.csv"
    metrics_df.to_csv(metrics_csv)
    print(f"[REG] Saved regression metrics to {metrics_csv}")

    try:
        metrics_parquet = reg_dir / "advanced_metrics.parquet"
        metrics_df.to_parquet(metrics_parquet)
        print(f"[REG] Saved regression metrics to {metrics_parquet}")
    except Exception as exc:
        print(f"[WARN] Could not write Parquet for regression metrics: {exc}")

    # best model by RMSE
    best_model_name = metrics_df["rmse"].idxmin()
    if best_model_name == "LinearRegression_baseline":
        best_model = lin_pipeline
    elif best_model_name == "RandomForest_baseline":
        best_model = rf_pipeline
    elif best_model_name == "GradientBoosting_baseline":
        best_model = gbr_pipeline
    else:
        best_model = best_rf_pipeline

    model_path = reg_dir / f"sales_regression_best_{best_model_name}.joblib"
    joblib.dump(best_model, model_path)
    print(f"[REG] Saved best model ({best_model_name}) to {model_path}")

    return metrics_df


def main() -> int:
    try:
        train_baseline_and_tuned_models(limit_rows=None)
        print("[DONE] Regression training (baseline + tuned) completed successfully.")
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1


__all__ = [
    "add_time_features",
    "add_ratio_features",
    "create_regression_features",
    "build_regression_dataset",
    "build_preprocessor",
    "train_baseline_and_tuned_models",
]


if __name__ == "__main__":
    raise SystemExit(main())
