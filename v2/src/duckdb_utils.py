# src/duckdb_utils.py

from __future__ import annotations

from pathlib import Path
from typing import Dict

import duckdb
import polars as pl

from .config_v2 import load_project_config, ProjectConfig
from .logging_utils import get_logger

logger = get_logger(__name__)


def init_duckdb(config: ProjectConfig) -> duckdb.DuckDBPyConnection:
    """
    Initialize (or open) a DuckDB database and create/replace the `sales` table
    based on parquet files written by the data pipeline.

    Returns an open DuckDB connection.
    """
    db_path = config.paths.duckdb_path
    parquet_glob = config.duckdb.parquet_glob
    table_name = config.duckdb.table_name

    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"[DUCKDB] Connecting to DuckDB at {db_path}")
    conn = duckdb.connect(str(db_path))

    # Use project root as current working directory for DuckDB
    project_root = Path(__file__).resolve().parents[1]
    conn.execute(f"SET home_directory='{project_root.as_posix()}'")

    # Create table from parquet (CREATE OR REPLACE)
    logger.info(
        f"[DUCKDB] Creating/refreshing table '{table_name}' "
        f"from parquet glob '{parquet_glob}'"
    )

    create_sql = f"""
    CREATE OR REPLACE TABLE {table_name} AS
    SELECT *
    FROM read_parquet('{parquet_glob}');
    """
    conn.execute(create_sql)

    # Optional: create some useful indices
    # DuckDB uses zone-maps and statistics under the hood; explicit indexing is limited.
    logger.info("[DUCKDB] Table created successfully.")
    return conn


def sales_by_productline(conn: duckdb.DuckDBPyConnection, table_name: str) -> pl.DataFrame:
    """
    Aggregate total and average sales per PRODUCTLINE.
    Returns a Polars DataFrame via DuckDB's .pl() integration.
    """
    logger.info("[DUCKDB] Running sales_by_productline aggregation")
    query = f"""
        SELECT
            PRODUCTLINE,
            SUM(SALES)      AS total_sales,
            AVG(SALES)      AS avg_sales,
            COUNT(*)        AS orders_count
        FROM {table_name}
        GROUP BY PRODUCTLINE
        ORDER BY total_sales DESC;
    """
    return conn.query(query).pl()


def top_countries_by_sales(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    limit: int = 10,
) -> pl.DataFrame:
    """
    Top N countries by total sales.
    """
    logger.info(f"[DUCKDB] Running top_countries_by_sales (limit={limit})")
    query = f"""
        SELECT
            COUNTRY,
            SUM(SALES) AS total_sales,
            AVG(SALES) AS avg_sales,
            COUNT(*)   AS orders_count
        FROM {table_name}
        GROUP BY COUNTRY
        ORDER BY total_sales DESC
        LIMIT {limit};
    """
    return conn.query(query).pl()


def sales_by_year(conn: duckdb.DuckDBPyConnection, table_name: str) -> pl.DataFrame:
    """
    Aggregate total and average sales per YEAR.
    """
    logger.info("[DUCKDB] Running sales_by_year aggregation")
    query = f"""
        SELECT
            YEAR,
            SUM(SALES) AS total_sales,
            AVG(SALES) AS avg_sales,
            COUNT(*)   AS orders_count
        FROM {table_name}
        GROUP BY YEAR
        ORDER BY YEAR;
    """
    return conn.query(query).pl()


def summarize_duckdb(config_path: str | Path = "config_v2.yaml") -> Dict[str, pl.DataFrame]:
    """
    Utility function to:

    1. Load config
    2. Initialize DuckDB and create/refresh `sales` table
    3. Run core EDA aggregations

    Returns a dict of Polars DataFrames:
      {
        "by_productline": pl.DataFrame,
        "top_countries":  pl.DataFrame,
        "by_year":        pl.DataFrame,
      }
    """
    config = load_project_config(config_path)
    conn = init_duckdb(config)
    table_name = config.duckdb.table_name

    by_pl = sales_by_productline(conn, table_name)
    top_c = top_countries_by_sales(conn, table_name)
    by_y = sales_by_year(conn, table_name)

    # It's safe to keep connection open for further interactive use,
    # but in this helper we close it at the end.
    conn.close()

    return {
        "by_productline": by_pl,
        "top_countries": top_c,
        "by_year": by_y,
    }


if __name__ == "__main__":
    results = summarize_duckdb()
    for name, df in results.items():
        logger.info(f"[DUCKDB] Preview of {name}:\n{df.head()}")
