from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_CORE_SRC = REPO_ROOT / "libs" / "ml_core" / "src"
SALES_SRC = REPO_ROOT / "projects" / "sales_forecasting_regression" / "src"
CALIFORNIA_SRC = REPO_ROOT / "projects" / "california_housing_template" / "src"

for source in [ML_CORE_SRC, SALES_SRC, CALIFORNIA_SRC]:
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from california_housing_template.eda import generate_california_eda_report
from sales_forecasting_regression.eda import generate_sales_eda_report


def _git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    value = result.stdout.strip()
    return value or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reproducible EDA reports for sales and california projects.")
    parser.add_argument("--project", choices=["sales", "california", "all"], default="all")
    parser.add_argument("--output-dir", type=str, default="reports/eda")
    parser.add_argument("--report-version", type=str, default="1.0.0")
    parser.add_argument("--sales-csv", type=str, default=None)
    parser.add_argument("--california-force-synthetic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = (REPO_ROOT / args.output_dir).resolve()
    report_version = args.report_version
    commit_hash = _git_commit_hash()
    if commit_hash is not None:
        report_version = f"{report_version}+{commit_hash[:8]}"

    payload: dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "report_version": report_version,
        "projects": {},
    }

    if args.project in {"sales", "all"}:
        sales_output_dir = output_root if args.project == "sales" else (output_root / "sales")
        sales = generate_sales_eda_report(
            output_dir=sales_output_dir,
            report_version=report_version,
            csv_path=Path(args.sales_csv) if args.sales_csv else None,
            use_synthetic_if_missing=True,
        )
        payload["projects"] = dict(payload["projects"]) | {"sales": sales}

    if args.project in {"california", "all"}:
        california_output_dir = output_root if args.project == "california" else (output_root / "california")
        california = generate_california_eda_report(
            output_dir=california_output_dir,
            report_version=report_version,
            force_synthetic=bool(args.california_force_synthetic),
            use_synthetic_if_fetch_fails=True,
        )
        payload["projects"] = dict(payload["projects"]) | {"california": california}

    manifest_path = output_root / "manifest.json"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
