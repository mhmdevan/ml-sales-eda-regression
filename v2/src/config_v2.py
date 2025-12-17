# src/config_v2.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import yaml

from .logging_utils import get_logger

logger = get_logger(__name__)

@dataclass
class PathsConfig:
    raw_csv: Path
    processed_parquet_dir: Path
    duckdb_path: Path
    output_plots_dir: Path
    output_summary_dir: Path
    output_metrics_dir: Path

@dataclass
class DuckDBConfig:
    table_name: str
    parquet_glob: str

@dataclass
class ParquetConfig:
    partition_by: List[str]

@dataclass
class CleaningConfig:
    drop_columns: List[str]
    fill_unknown_columns: List[str]
    orderdate_column: str
    orderdate_format: str | None

@dataclass
class ProjectConfig:
    paths: PathsConfig
    duckdb: DuckDBConfig
    parquet: ParquetConfig
    cleaning: CleaningConfig


def load_project_config(config_path: Path | str = "config_v2.yaml") -> ProjectConfig:
    """
    Load the v2 YAML config and return a typed ProjectConfig.

    Parameters
    ----------
    config_path:
        Path to the YAML config file relative to project root

    Returns
    -------
    ProjectConfig
    """
    project_root = Path(__file__).resolve().parents[1]
    config_file = project_root / config_path

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    logger.info(f"Loading v2 config from {config_file}")
    with config_file.open("r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    paths_cfg = cfg["paths"]
    duckdb_cfg = cfg["duckdb"]
    parquet_cfg = cfg["parquet"]
    cleaning_cfg = cfg["cleaning"]

    paths = PathsConfig(
        raw_csv=(project_root / paths_cfg["raw_csv"]).resolve(),
        processed_parquet_dir=(project_root / paths_cfg["processed_parquet_dir"]).resolve(),
        duckdb_path=(project_root / paths_cfg["duckdb_path"]).resolve(),
        output_plots_dir=(project_root / paths_cfg["output_plots_dir"]).resolve(),
        output_summary_dir=(project_root / paths_cfg["output_summary_dir"]).resolve(),
        output_metrics_dir=(project_root / paths_cfg["output_metrics_dir"]).resolve(),
    )

    duckdb_conf = DuckDBConfig(
        table_name=duckdb_cfg["table_name"],
        parquet_glob=duckdb_cfg["parquet_glob"],
    )

    parquet_conf = ParquetConfig(
        partition_by=list(parquet_cfg.get("partition_by", [])),
    )

    cleaning_conf = CleaningConfig(
        drop_columns=list(cleaning_cfg.get("drop_columns", [])),
        fill_unknown_columns=list(cleaning_cfg.get("fill_unknown_columns", [])),
        orderdate_column=str(cleaning_cfg["orderdate_column"]),
        orderdate_format=cleaning_cfg.get("orderdate_format"),
    )

    return ProjectConfig(
        paths=paths,
        duckdb=duckdb_conf,
        parquet=parquet_conf,
        cleaning=cleaning_conf,
    )
