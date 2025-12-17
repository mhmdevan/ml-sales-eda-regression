# src/eda_polars_duckdb.py

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import polars as pl

from .config_v2 import load_project_config, ProjectConfig
from .data_parquet import run_data_to_parquet_pipeline
from .duckdb_utils import summarize_duckdb
from .logging_utils import get_logger

logger = get_logger(__name__)


def _ensure_output_dirs(config: ProjectConfig) -> None:
    for path in [
        config.paths.output_plots_dir,
        config.paths.output_summary_dir,
        config.paths.output_metrics_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def save_summary_tables(
    config: ProjectConfig,
    results: Dict[str, pl.DataFrame],
) -> None:
    """
    Save EDA summary tables (Polars DataFrames) to CSV and Parquet
    in the configured summary directory.
    """
    summary_dir = config.paths.output_summary_dir
    summary_dir.mkdir(parents=True, exist_ok=True)

    for name, df in results.items():
        csv_path = summary_dir / f"{name}.csv"
        parquet_path = summary_dir / f"{name}.parquet"

        logger.info(f"[SUMMARY] Saving {name} to {csv_path} and {parquet_path}")
        df.write_csv(csv_path)
        df.write_parquet(parquet_path)


def plot_total_sales_by_productline(
    config: ProjectConfig,
    by_productline: pl.DataFrame,
) -> None:
    """
    Create a simple bar chart for total sales by productline.

    Polars → pandas conversion is used only at the plotting edge.
    """
    plots_dir = config.paths.output_plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Convert Polars DataFrame to pandas for plotting
    pdf = by_productline.to_pandas()
    pdf = pdf.sort_values("total_sales", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(pdf["PRODUCTLINE"], pdf["total_sales"])
    ax.set_title("Total Sales by Productline (DuckDB + Parquet + Polars)")
    ax.set_xlabel("Productline")
    ax.set_ylabel("Total Sales")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    out_path = plots_dir / "total_sales_by_productline_duckdb.png"
    plt.savefig(out_path)
    plt.close(fig)

    logger.info(f"[PLOT] Saved total_sales_by_productline chart to {out_path}")


def run_eda_v2(config_path: str | Path = "config_v2.yaml") -> None:
    """
    Orchestrate the full v2 data layer EDA:

    1. Ensure data → parquet pipeline has run
    2. Initialize DuckDB and aggregate EDA tables
    3. Save EDA tables (CSV + Parquet)
    4. Save at least one chart (total sales by productline)
    """
    config = load_project_config(config_path)
    _ensure_output_dirs(config)

    # Step 1 – run data → parquet (idempotent: overwrites parquet files)
    logger.info("[PIPELINE] Running data → parquet step for v2")
    run_data_to_parquet_pipeline(config_path=config_path)

    # Step 2 – DuckDB EDA
    logger.info("[PIPELINE] Initializing DuckDB and running EDA queries")
    results = summarize_duckdb(config_path=config_path)

    # Step 3 – Save summary tables
    save_summary_tables(config, results)

    # Step 4 – Plot total sales by productline
    by_productline = results["by_productline"]
    plot_total_sales_by_productline(config, by_productline)

    logger.info("[DONE] v2 EDA pipeline (Parquet + DuckDB + Polars) completed.")


if __name__ == "__main__":
    run_eda_v2()
