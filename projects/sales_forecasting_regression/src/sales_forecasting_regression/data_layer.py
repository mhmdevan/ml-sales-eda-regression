from __future__ import annotations

import argparse
from pathlib import Path


def materialize_sales_data_layer(
    *,
    csv_path: Path,
    parquet_dir: Path,
    duckdb_path: Path,
) -> dict[str, str]:
    try:
        import duckdb
        import polars as pl
    except Exception as exc:
        raise RuntimeError("polars and duckdb are required for data layer materialization.") from exc

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")

    parquet_dir.mkdir(parents=True, exist_ok=True)
    duckdb_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pl.read_csv(csv_path, try_parse_dates=True)
    if "SALES" not in frame.columns and {"QUANTITYORDERED", "PRICEEACH"}.issubset(set(frame.columns)):
        frame = frame.with_columns((pl.col("QUANTITYORDERED") * pl.col("PRICEEACH")).alias("SALES"))

    if "YEAR_ID" in frame.columns:
        unique_years = frame.select("YEAR_ID").unique().to_series().to_list()
        for year in unique_years:
            partition = frame.filter(pl.col("YEAR_ID") == year)
            partition.write_parquet(parquet_dir / f"sales_year={year}.parquet")
    else:
        frame.write_parquet(parquet_dir / "sales.parquet")

    connection = duckdb.connect(str(duckdb_path))
    try:
        connection.execute(
            "CREATE OR REPLACE TABLE sales AS SELECT * FROM read_parquet(?)",
            [str(parquet_dir / "*.parquet")],
        )
    finally:
        connection.close()

    return {
        "parquet_dir": str(parquet_dir),
        "duckdb_path": str(duckdb_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create Parquet and DuckDB layers for sales data.")
    parser.add_argument("--csv-path", type=str, required=True)
    parser.add_argument("--parquet-dir", type=str, default="data/processed/sales_parquet")
    parser.add_argument("--duckdb-path", type=str, default="duckdb/sales.duckdb")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = materialize_sales_data_layer(
        csv_path=Path(args.csv_path),
        parquet_dir=Path(args.parquet_dir),
        duckdb_path=Path(args.duckdb_path),
    )
    print(result)


if __name__ == "__main__":
    main()
