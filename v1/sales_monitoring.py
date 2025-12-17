"""
sales_monitoring.py

Simple drift / monitoring demo on the Kaggle sales dataset.

Compares the distribution of SALES between two time periods
(e.g. earliest YEAR vs latest YEAR) using a basic PSI metric,
and saves a comparison plot.

This is NOT production-grade monitoring, but it shows that you
understand the idea of checking for distribution shifts over time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sales_eda import (
    PROJECT_ROOT,
    load_config,
    load_data,
    add_date_features,
    clean_missing_values,
    remove_outliers,
)


def calculate_psi(
    ref: np.ndarray,
    cur: np.ndarray,
    bins: int = 10,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Population Stability Index (PSI) between two 1D arrays.

    Steps:
      - build bin edges using ref distribution,
      - compute shares in each bin,
      - PSI = sum( (cur_share - ref_share) * ln(cur_share / ref_share) ).

    We add a small epsilon to avoid log(0) / division by zero.
    """
    ref = np.asarray(ref, dtype=float)
    cur = np.asarray(cur, dtype=float)

    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]

    if ref.size == 0 or cur.size == 0:
        raise ValueError("Empty reference or current array for PSI.")

    bin_edges = np.histogram_bin_edges(ref, bins=bins)

    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    cur_counts, _ = np.histogram(cur, bins=bin_edges)

    ref_props = ref_counts / max(ref_counts.sum(), 1)
    cur_props = cur_counts / max(cur_counts.sum(), 1)

    eps = 1e-6
    ref_props = np.clip(ref_props, eps, None)
    cur_props = np.clip(cur_props, eps, None)

    psi_values = (cur_props - ref_props) * np.log(cur_props / ref_props)
    psi_total = float(np.sum(psi_values))

    return psi_total, psi_values, ref_props, cur_props


def main() -> int:
    try:
        config = load_config()
        data_path = PROJECT_ROOT / config["data_path"]
        numeric_cols = config["numeric_columns"]
        cleaning_cfg = config["cleaning"]
        iqr_factor = float(config["outliers"]["iqr_factor"])

        df_raw = load_data(data_path)
        df_dates = add_date_features(df_raw)
        df_clean_missing = clean_missing_values(df_dates, cleaning_cfg=cleaning_cfg)
        df_clean = remove_outliers(
            df_clean_missing,
            numeric_cols=numeric_cols,
            factor=iqr_factor,
        )

        if "SALES" not in df_clean.columns:
            raise KeyError("Expected column 'SALES' is missing for monitoring.")

        # Choose two time periods:
        if "YEAR" in df_clean.columns:
            years = sorted(df_clean["YEAR"].unique())
            if len(years) >= 2:
                ref_year = years[0]
                cur_year = years[-1]
                ref_slice = df_clean.loc[df_clean["YEAR"] == ref_year, "SALES"]
                cur_slice = df_clean.loc[df_clean["YEAR"] == cur_year, "SALES"]
                label_ref = f"year_{ref_year}"
                label_cur = f"year_{cur_year}"
            else:
                # fallback: first half vs second half
                midpoint = len(df_clean) // 2
                ref_slice = df_clean["SALES"].iloc[:midpoint]
                cur_slice = df_clean["SALES"].iloc[midpoint:]
                label_ref = "first_half"
                label_cur = "second_half"
        else:
            # no YEAR column – fallback to first vs second half
            midpoint = len(df_clean) // 2
            ref_slice = df_clean["SALES"].iloc[:midpoint]
            cur_slice = df_clean["SALES"].iloc[midpoint:]
            label_ref = "first_half"
            label_cur = "second_half"

        psi_total, psi_values, ref_props, cur_props = calculate_psi(
            ref_slice.values,
            cur_slice.values,
            bins=10,
        )

        output_dir = PROJECT_ROOT / config["output_dir"]
        mon_dir = output_dir / "monitoring"
        mon_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON report
        report_path = mon_dir / f"psi_sales_{label_ref}_vs_{label_cur}.json"
        report = {
            "ref_label": label_ref,
            "cur_label": label_cur,
            "psi_total": psi_total,
            "psi_per_bin": psi_values.tolist(),
            "ref_props": ref_props.tolist(),
            "cur_props": cur_props.tolist(),
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[MON] Saved PSI report to {report_path}")
        print(f"[MON] PSI(SALES {label_ref} -> {label_cur}) = {psi_total:.4f}")

        # Save comparison plot
        plt.figure(figsize=(9, 5))
        bins = np.histogram_bin_edges(ref_slice.values, bins=10)
        plt.hist(
            ref_slice.values,
            bins=bins,
            alpha=0.5,
            label=label_ref,
            density=True,
        )
        plt.hist(
            cur_slice.values,
            bins=bins,
            alpha=0.5,
            label=label_cur,
            density=True,
        )
        plt.title(f"SALES distribution: {label_ref} vs {label_cur}", fontsize=14)
        plt.xlabel("SALES", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.legend()
        plt.tight_layout()

        plot_path = mon_dir / f"sales_{label_ref}_vs_{label_cur}.png"
        plt.savefig(plot_path, dpi=140)
        plt.close()
        print(f"[MON] Saved comparison plot to {plot_path}")

        return 0

    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
