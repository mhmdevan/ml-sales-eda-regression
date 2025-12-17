# src/data_parquet.py

from __future__ import annotations

from pathlib import Path
from typing import Optional, List

import polars as pl

from .config_v2 import load_project_config, ProjectConfig
from .logging_utils import get_logger

logger = get_logger(__name__)


def _ensure_dir(path: Path) -> None:
    """
    Create directory (and parents) if it does not exist.
    """
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def load_raw_sales(config: ProjectConfig) -> pl.DataFrame:
    """
    Load the raw CSV into a Polars DataFrame.

    - First try strict UTF-8
    - If it fails with invalid UTF-8, fall back to a more permissive encoding.

    This is common with Kaggle / Excel exports that contain weird characters.
    """
    raw_path = config.paths.raw_csv
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {raw_path}")

    logger.info(f"[DATA] Loading raw sales CSV from {raw_path}")

    try:
        # strict UTF-8
        df = pl.read_csv(raw_path)
    except pl.exceptions.ComputeError as e:
        logger.warning(
            "[DATA] Failed to read CSV with default UTF-8 "
            f"({e}). Retrying with encoding='utf8-lossy'..."
        )
        # utf8-lossy: invalid sequences will be replaced with �
        df = pl.read_csv(raw_path, encoding="utf8-lossy")

    logger.info(f"[DATA] Raw shape: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def clean_sales_df(df: pl.DataFrame, config: ProjectConfig) -> pl.DataFrame:
    """
    Apply basic cleaning and feature engineering:

    - Drop columns marked in config.cleaning.drop_columns
    - Fill NULLs for selected columns with 'Unknown'
    - Parse ORDERDATE to a proper Date / Datetime
    - Add YEAR, MONTH numeric columns
    """
    cleaning = config.cleaning

    # Drop configured columns if they exist
    to_drop: List[str] = [
        col for col in cleaning.drop_columns if col in df.columns
    ]
    if to_drop:
        logger.info(f"[CLEAN] Dropping columns: {to_drop}")
        df = df.drop(to_drop)

    # Fill NULLs for columns that should get 'Unknown'
    for col in cleaning.fill_unknown_columns:
        if col in df.columns:
            null_count = df[col].null_count()
            if null_count > 0:
                logger.info(f"[CLEAN] Filling {null_count} NULLs in '{col}' with 'Unknown'")
                df = df.with_columns(
                    pl.when(pl.col(col).is_null())
                    .then(pl.lit("Unknown"))
                    .otherwise(pl.col(col))
                    .alias(col)
                )

    # Parse ORDERDATE (string -> Datetime -> Date)
    order_col = cleaning.orderdate_column
    if order_col not in df.columns:
        raise KeyError(f"Expected order date column '{order_col}' not found in data")

    logger.info(f"[CLEAN] Parsing order date column '{order_col}'")

    # We want final column type = Date, but raw CSV usually has
    # strings like "2/24/2003 0:00" (datetime with hours/minutes).
    # Strategy:
    #   1) If config has an explicit format → use strptime(Datetime, fmt).dt.date()
    #   2) Else: try to_datetime(strict=False).dt.date()
    #   3) If that still fails → fallback to a known pattern "%m/%d/%Y %H:%M"

    try:
        if cleaning.orderdate_format:
            # Explicit format from config, parse as Datetime then cast to Date
            df = df.with_columns(
                pl.col(order_col)
                .str.strptime(pl.Datetime, fmt=cleaning.orderdate_format, strict=False)
                .dt.date()
                .alias(order_col)
            )
        else:
            # Let Polars infer datetime format first, then take only the date part
            df = df.with_columns(
                pl.col(order_col)
                .str.to_datetime(strict=False)
                .dt.date()
                .alias(order_col)
            )
    except pl.exceptions.ComputeError as e:
        # Fallback for the classic "Sales Data Sample" pattern like "2/24/2003 0:00"
        logger.warning(
            "[CLEAN] Failed to parse ORDERDATE with automatic format "
            f"({e}). Falling back to '%m/%d/%Y %H:%M'..."
        )
        df = df.with_columns(
            pl.col(order_col)
            .str.strptime(pl.Datetime, fmt="%m/%d/%Y %H:%M", strict=False)
            .dt.date()
            .alias(order_col)
        )

    # Add YEAR, MONTH integer columns
    df = df.with_columns(
        [
            pl.col(order_col).dt.year().alias("YEAR"),
            pl.col(order_col).dt.month().alias("MONTH"),
        ]
    )

    logger.info(
        f"[CLEAN] Finished cleaning; resulting shape: {df.shape[0]} rows × {df.shape[1]} columns"
    )
    return df



def export_to_parquet(config: ProjectConfig, df: pl.DataFrame) -> None:
    """
    Export cleaned sales data to Parquet files in the configured directory.

    We optionally "partition" by one or more columns (e.g., YEAR) by
    writing one parquet file per unique combination of partition columns.

    This makes it easier to query with DuckDB using a glob path.
    """
    parquet_dir = config.paths.processed_parquet_dir
    _ensure_dir(parquet_dir)

    part_cols = config.parquet.partition_by

    if not part_cols:
        # Single parquet file (no partitioning)
        out_path = parquet_dir / "sales_clean.parquet"
        logger.info(f"[PARQUET] Writing single parquet to {out_path}")
        df.write_parquet(out_path)
        return

    # Partitioned write: one file per combination of partition columns
    logger.info(f"[PARQUET] Writing partitioned parquet files by columns: {part_cols}")

    # We only support 1 or 2 partition columns here for simplicity
    # but you can extend this logic if needed.
    # Implementation: groupby and write one parquet per group.
    grouped = df.group_by(part_cols, maintain_order=True)

    for group_vals, group_df in grouped:
        # group_vals is a list of (col_name, value) pairs
        # e.g. [("YEAR", 2003)]
        # Build a file name suffix based on partition values
        # Example: sales_YEAR=2003.parquet
        suffix_parts = []
        for col_name, value in zip(part_cols, group_vals):
            suffix_parts.append(f"{col_name}={value}")
        suffix = "_".join(suffix_parts)
        out_path = parquet_dir / f"sales_{suffix}.parquet"

        logger.info(
            f"[PARQUET] Writing partition file for {dict(zip(part_cols, group_vals))} "
            f"to {out_path}"
        )
        group_df.write_parquet(out_path)


def run_data_to_parquet_pipeline(config_path: str | Path = "config_v2.yaml") -> None:
    """
    Orchestrate the data -> parquet pipeline:

    1. Load config
    2. Load raw CSV with Polars
    3. Clean + feature engineering (YEAR, MONTH)
    4. Write partitioned Parquet files
    """
    config = load_project_config(config_path)

    _ensure_dir(config.paths.processed_parquet_dir)

    df_raw = load_raw_sales(config)
    df_clean = clean_sales_df(df_raw, config)
    export_to_parquet(config, df_clean)

    logger.info("[DONE] Data -> Parquet pipeline completed.")


if __name__ == "__main__":
    run_data_to_parquet_pipeline()
