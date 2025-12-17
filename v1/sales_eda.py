"""
sales_eda.py

Config-driven EDA pipeline for the Kaggle Sample Sales dataset.

Features:
- Load CSV from a configurable path (config.yaml).
- Clean missing values based on explicit rules.
- Detect and remove outliers using the IQR rule.
- Compute descriptive and grouped statistics.
- Export summary tables to CSV / Parquet for downstream ML.
- Generate histograms, scatter plots, and boxplots based on config.
- Designed as a reusable module (functions can be imported from other projects).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
import copy


sns.set(style="whitegrid")

# -----------------------------------------------------------------------------
# 1. Paths & default config
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

DEFAULT_CONFIG = {
    "data_path": "data/sales_data_sample.csv",
    "output_dir": "output",
    "numeric_columns": ["QUANTITYORDERED", "PRICEEACH", "SALES", "MSRP"],
    "cleaning": {
        "drop_columns": ["ADDRESSLINE2"],
        "fill_unknown_columns": ["STATE", "TERRITORY"],
        "postalcode_column": "POSTALCODE",
    },
    "outliers": {
        "iqr_factor": 1.5,
    },
    "plots": {
        "histograms": ["SALES", "QUANTITYORDERED"],
        "scatters": [
            ["QUANTITYORDERED", "SALES"],
            ["PRICEEACH", "SALES"],
        ],
        "boxplots": [
            ["PRODUCTLINE", "SALES"],
            ["DEALSIZE", "QUANTITYORDERED"],
        ],
    },
}


# -----------------------------------------------------------------------------
# 2. Config loading
# -----------------------------------------------------------------------------

def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Dict:
    """
    Load configuration from a YAML file and merge with DEFAULT_CONFIG.

    Supports optional multi-dataset configs via:
      - active_dataset: <name>
      - datasets:
          <name>:
            data_path: ...
            numeric_columns: ...
            cleaning: ...
            outliers: ...
            plots: ...
    """
    config = copy.deepcopy(DEFAULT_CONFIG)

    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        # shallow merge with a bit of nested handling
        for key, value in raw.items():
            if (
                key in config
                and isinstance(config[key], dict)
                and isinstance(value, dict)
            ):
                cfg = config[key].copy()
                cfg.update(value)
                config[key] = cfg
            else:
                config[key] = value

        print(f"[INFO] Loaded config from {path}")
    else:
        print(f"[WARN] Config file not found at {path}, using DEFAULT_CONFIG")

    # -------- multi-dataset override --------
    active_dataset = config.get("active_dataset")
    datasets_cfg = config.get("datasets")

    if active_dataset and isinstance(datasets_cfg, dict):
        ds = datasets_cfg.get(active_dataset)
        if ds is None:
            print(
                f"[WARN] active_dataset '{active_dataset}' not found in 'datasets', "
                "using top-level config only."
            )
        else:
            overridable_keys = [
                "data_path",
                "numeric_columns",
                "cleaning",
                "outliers",
                "plots",
            ]
            for key in overridable_keys:
                if key in ds:
                    if isinstance(config.get(key), dict) and isinstance(
                        ds[key], dict
                    ):
                        merged = config[key].copy()
                        merged.update(ds[key])
                        config[key] = merged
                    else:
                        config[key] = ds[key]
            print(f"[INFO] Applied dataset-specific config for '{active_dataset}'")

    return config


# -----------------------------------------------------------------------------
# 3. Data loading
# -----------------------------------------------------------------------------

def load_data(csv_path: Path) -> pd.DataFrame:
    """
    Load the sales dataset from a CSV file.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        A pandas DataFrame with the raw data.

    Raises:
        FileNotFoundError: if the file does not exist.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # encoding="latin1" is required for this Kaggle dataset
    df = pd.read_csv(csv_path, encoding="latin1")
    print(f"[INFO] Loaded data from {csv_path}")
    print(f"[INFO] Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# -----------------------------------------------------------------------------
# 4. Basic preprocessing
# -----------------------------------------------------------------------------

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert ORDERDATE to datetime and add YEAR and MONTH columns.

    Args:
        df: Input DataFrame, expected to contain 'ORDERDATE'.

    Returns:
        A new DataFrame with YEAR and MONTH added.

    Raises:
        KeyError: if 'ORDERDATE' is missing.
    """
    result = df.copy()

    if "ORDERDATE" not in result.columns:
        raise KeyError("Expected column 'ORDERDATE' is missing from DataFrame.")

    result["ORDERDATE"] = pd.to_datetime(result["ORDERDATE"])
    result["YEAR"] = result["ORDERDATE"].dt.year
    result["MONTH"] = result["ORDERDATE"].dt.month

    print("[INFO] Converted ORDERDATE to datetime and added YEAR, MONTH")
    return result


# -----------------------------------------------------------------------------
# 5. Missing values handling
# -----------------------------------------------------------------------------

def clean_missing_values(df: pd.DataFrame, cleaning_cfg: Dict) -> pd.DataFrame:
    """
    Handle missing values based on configurable, explicit rules.

    cleaning_cfg expects keys:
      - drop_columns: list[str]
      - fill_unknown_columns: list[str]
      - postalcode_column: str
    """
    result = df.copy()

    print("\n[INFO] Missing values BEFORE cleaning:")
    print(result.isna().sum().sort_values(ascending=False).head(10))

    drop_cols = cleaning_cfg.get("drop_columns", [])
    fill_unknown_cols = cleaning_cfg.get("fill_unknown_columns", [])
    postal_col = cleaning_cfg.get("postalcode_column", "POSTALCODE")

    # 1) Drop configured columns
    for col in drop_cols:
        if col in result.columns:
            result = result.drop(columns=[col])
            print(f"[CLEAN] Dropped column '{col}'")

    # 2) Fill configured columns with 'Unknown'
    for col in fill_unknown_cols:
        if col in result.columns:
            missing_before = result[col].isna().sum()
            result[col] = result[col].fillna("Unknown")
            missing_after = result[col].isna().sum()
            filled = missing_before - missing_after
            print(f"[CLEAN] Filled {filled} missing values in '{col}' with 'Unknown'")

    # 3) POSTALCODE as string + fill missing
    if postal_col in result.columns:
        missing_before = result[postal_col].isna().sum()
        result[postal_col] = result[postal_col].astype("object").fillna("Unknown")
        missing_after = result[postal_col].isna().sum()
        filled = missing_before - missing_after
        print(f"[CLEAN] Filled {filled} missing values in '{postal_col}' with 'Unknown'")

    print("\n[INFO] Missing values AFTER cleaning:")
    print(result.isna().sum().sort_values(ascending=False).head(10))

    return result


# -----------------------------------------------------------------------------
# 6. Outlier detection & removal (IQR)
# -----------------------------------------------------------------------------

def detect_outliers_iqr(series: pd.Series, factor: float = 1.5) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Detect outliers in a numeric Series using the IQR rule.

    Args:
        series: Numeric pandas Series.
        factor: IQR factor (1.5 is standard; use 3 for more conservative).

    Returns:
        mask: boolean Series (True = outlier)
        bounds: dict with q1, q3, iqr, lower_bound, upper_bound
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr

    mask = (series < lower_bound) | (series > upper_bound)

    bounds = {
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
    }

    return mask, bounds


def remove_outliers(
    df: pd.DataFrame,
    numeric_cols: Iterable[str],
    factor: float = 1.5,
) -> pd.DataFrame:
    """
    Remove outliers from the DataFrame for each column in numeric_cols
    based on the IQR rule.

    For each numeric column, we:
      - compute IQR,
      - mark outliers outside [Q1 - factor * IQR, Q3 + factor * IQR],
      - drop rows where that column is an outlier.
    """
    result = df.copy()
    print("\n[INFO] Removing outliers using IQR rule:")

    for col in numeric_cols:
        if col not in result.columns:
            raise KeyError(f"Expected numeric column '{col}' is missing.")

        mask, bounds = detect_outliers_iqr(result[col], factor=factor)
        outlier_count = int(mask.sum())
        row_count_before = result.shape[0]

        print(
            f"  [OUTLIERS] Column '{col}': "
            f"{outlier_count} rows outside "
            f"[{bounds['lower_bound']:.2f}, {bounds['upper_bound']:.2f}]"
        )

        # Keep only non-outliers for this column
        result = result.loc[~mask]

        row_count_after = result.shape[0]
        print(
            f"  [OUTLIERS] Rows before: {row_count_before}, "
            f"after: {row_count_after}"
        )

    print(f"[INFO] Final shape after outlier removal: {result.shape}")
    return result


# -----------------------------------------------------------------------------
# 7. Descriptive & grouped statistics
# -----------------------------------------------------------------------------

def get_descriptive_stats(df: pd.DataFrame, numeric_cols: Iterable[str]) -> pd.DataFrame:
    """
    Compute descriptive statistics for the selected numeric columns.
    """
    stats = df[list(numeric_cols)].describe().T
    return stats


def get_grouped_stats(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Compute a few useful grouped statistics:
      - Sales by PRODUCTLINE
      - Top 10 countries by sales
      - Sales by YEAR

    Returns:
        A dict mapping a name -> DataFrame.
    """
    stats: Dict[str, pd.DataFrame] = {}

    if "SALES" not in df.columns:
        raise KeyError("Expected column 'SALES' is missing.")

    # Sales by PRODUCTLINE
    if "PRODUCTLINE" in df.columns:
        productline_stats = (
            df.groupby("PRODUCTLINE")
            .agg(
                total_sales=("SALES", "sum"),
                avg_sales=("SALES", "mean"),
                orders_count=("ORDERNUMBER", "nunique"),
            )
            .sort_values("total_sales", ascending=False)
        )
        stats["sales_by_productline"] = productline_stats

    # Top 10 countries by total sales
    if "COUNTRY" in df.columns:
        country_stats = (
            df.groupby("COUNTRY")
            .agg(
                total_sales=("SALES", "sum"),
                avg_sales=("SALES", "mean"),
                orders_count=("ORDERNUMBER", "nunique"),
            )
            .sort_values("total_sales", ascending=False)
            .head(10)
        )
        stats["top_countries_by_sales"] = country_stats

    # Sales by YEAR
    if "YEAR" in df.columns:
        year_stats = (
            df.groupby("YEAR")
            .agg(
                total_sales=("SALES", "sum"),
                avg_sales=("SALES", "mean"),
                orders_count=("ORDERNUMBER", "nunique"),
            )
            .sort_values("YEAR")
        )
        stats["sales_by_year"] = year_stats

    return stats


def print_stats_to_console(
    numeric_stats: pd.DataFrame,
    grouped_stats: Dict[str, pd.DataFrame],
) -> None:
    """
    Pretty-print numeric and grouped stats to the console.
    """
    print("\n[INFO] Descriptive statistics for numeric columns:")
    print(numeric_stats)

    for name, table in grouped_stats.items():
        print(f"\n[INFO] {name.replace('_', ' ').title()}:")
        print(table)


def save_summary_tables(
    numeric_stats: pd.DataFrame,
    grouped_stats: Dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """
    Save numeric and grouped stats to CSV and, if possible, Parquet.
    """
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    # Numeric stats
    csv_path = summary_dir / "numeric_descriptive_stats.csv"
    numeric_stats.to_csv(csv_path, index=True)
    print(f"[SUMMARY] Saved numeric stats to {csv_path}")

    try:
        parquet_path = summary_dir / "numeric_descriptive_stats.parquet"
        numeric_stats.to_parquet(parquet_path, index=True)
        print(f"[SUMMARY] Saved numeric stats to {parquet_path}")
    except Exception as exc:
        print(f"[WARN] Could not write Parquet for numeric stats: {exc}")

    # Grouped stats
    for name, table in grouped_stats.items():
        csv_path = summary_dir / f"{name}.csv"
        table.to_csv(csv_path, index=True)
        print(f"[SUMMARY] Saved {name} to {csv_path}")

        try:
            parquet_path = summary_dir / f"{name}.parquet"
            table.to_parquet(parquet_path, index=True)
            print(f"[SUMMARY] Saved {name} to {parquet_path}")
        except Exception as exc:
            print(f"[WARN] Could not write Parquet for {name}: {exc}")


# -----------------------------------------------------------------------------
# 8. Plotting functions
# -----------------------------------------------------------------------------

def ensure_output_dir(output_dir: Path) -> None:
    """
    Create the output directory if it does not exist.
    """
    output_dir.mkdir(parents=True, exist_ok=True)


def plot_histogram(df: pd.DataFrame, column: str, output_dir: Path) -> None:
    ensure_output_dir(output_dir)

    plt.figure(figsize=(9, 5))
    plt.hist(df[column], bins=30, edgecolor="black")
    plt.title(f"Histogram of {column}", fontsize=14)
    plt.xlabel(column, fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.tight_layout()

    file_path = output_dir / f"hist_{column.lower()}.png"
    plt.savefig(file_path, dpi=140)
    plt.close()

    print(f"[PLOT] Saved histogram for '{column}' to {file_path}")


def plot_scatter(df: pd.DataFrame, x: str, y: str, output_dir: Path) -> None:
    ensure_output_dir(output_dir)

    plt.figure(figsize=(9, 5))
    plt.scatter(df[x], df[y], alpha=0.6)
    plt.title(f"{x} vs {y}", fontsize=14)
    plt.xlabel(x, fontsize=12)
    plt.ylabel(y, fontsize=12)
    plt.tight_layout()

    file_path = output_dir / f"scatter_{x.lower()}_vs_{y.lower()}.png"
    plt.savefig(file_path, dpi=140)
    plt.close()

    corr_value = df[x].corr(df[y])
    print(f"[PLOT] Saved scatter plot {x} vs {y} to {file_path}")
    print(f"[INFO] Correlation between {x} and {y}: {corr_value:.3f}")


def plot_boxplot(df: pd.DataFrame, x_cat: str, y_num: str, output_dir: Path) -> None:
    ensure_output_dir(output_dir)

    plt.figure(figsize=(11, 6))
    sns.boxplot(data=df, x=x_cat, y=y_num)
    plt.title(f"{y_num} by {x_cat}", fontsize=14)
    plt.xlabel(x_cat, fontsize=12)
    plt.ylabel(y_num, fontsize=12)
    plt.xticks(rotation=35)
    plt.tight_layout()

    file_path = output_dir / f"box_{y_num.lower()}_by_{x_cat.lower()}.png"
    plt.savefig(file_path, dpi=140)
    plt.close()

    print(f"[PLOT] Saved boxplot {y_num} by {x_cat} to {file_path}")


def generate_plots(df: pd.DataFrame, plots_cfg: Dict, output_dir: Path) -> None:
    """
    Generate all plots specified in the config:
      - histograms
      - scatters
      - boxplots
    """
    hist_cols = plots_cfg.get("histograms", [])
    scatter_pairs = plots_cfg.get("scatters", [])
    box_pairs = plots_cfg.get("boxplots", [])

    # Histograms
    for col in hist_cols:
        if col in df.columns:
            plot_histogram(df, col, output_dir)
        else:
            print(f"[WARN] Column '{col}' not found for histogram, skipping.")

    # Scatters
    for pair in scatter_pairs:
        if len(pair) != 2:
            continue
        x, y = pair
        if x in df.columns and y in df.columns:
            plot_scatter(df, x, y, output_dir)
        else:
            print(f"[WARN] Columns '{x}', '{y}' not found for scatter, skipping.")

    # Boxplots
    for pair in box_pairs:
        if len(pair) != 2:
            continue
        x_cat, y_num = pair
        if x_cat in df.columns and y_num in df.columns:
            plot_boxplot(df, x_cat, y_num, output_dir)
        else:
            print(f"[WARN] Columns '{x_cat}', '{y_num}' not found for boxplot, skipping.")


# -----------------------------------------------------------------------------
# 9. Main pipeline
# -----------------------------------------------------------------------------

def main() -> int:
    try:
        config = load_config()

        data_path = PROJECT_ROOT / config["data_path"]
        output_dir = PROJECT_ROOT / config["output_dir"]
        numeric_cols = config["numeric_columns"]
        cleaning_cfg = config["cleaning"]
        iqr_factor = float(config["outliers"]["iqr_factor"])
        plots_cfg = config["plots"]

        # 1) Load raw data
        df_raw = load_data(data_path)

        # 2) Add date-related features
        df_with_dates = add_date_features(df_raw)

        # 3) Clean missing values
        df_clean_missing = clean_missing_values(df_with_dates, cleaning_cfg=cleaning_cfg)

        # 4) Validate numeric columns
        for col in numeric_cols:
            if col not in df_clean_missing.columns:
                raise KeyError(f"Expected numeric column '{col}' is missing from data.")

        # 5) Remove outliers
        df_clean = remove_outliers(
            df_clean_missing,
            numeric_cols=numeric_cols,
            factor=iqr_factor,
        )

        # 6) Descriptive statistics
        numeric_stats = get_descriptive_stats(df_clean, numeric_cols=numeric_cols)

        # 7) Grouped statistics
        grouped_stats = get_grouped_stats(df_clean)

        # 8) Print stats to console
        print_stats_to_console(numeric_stats, grouped_stats)

        # 9) Save summary tables for downstream ML / reporting
        save_summary_tables(numeric_stats, grouped_stats, output_dir=output_dir)

        # 10) Plots
        generate_plots(df_clean, plots_cfg=plots_cfg, output_dir=output_dir)

        print("\n[DONE] EDA completed successfully.")
        return 0

    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1


# -----------------------------------------------------------------------------
# 10. Public API (for reuse as a module)
# -----------------------------------------------------------------------------

__all__ = [
    "PROJECT_ROOT",
    "load_config",
    "load_data",
    "add_date_features",
    "clean_missing_values",
    "detect_outliers_iqr",
    "remove_outliers",
    "get_descriptive_stats",
    "get_grouped_stats",
]


if __name__ == "__main__":
    sys.exit(main())
