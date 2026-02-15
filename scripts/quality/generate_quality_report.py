from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as element_tree

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_CORE_SRC = REPO_ROOT / "libs" / "ml_core" / "src"
SALES_SRC = REPO_ROOT / "projects" / "sales_forecasting_regression" / "src"
CALIFORNIA_SRC = REPO_ROOT / "projects" / "california_housing_template" / "src"

for source in [ML_CORE_SRC, SALES_SRC, CALIFORNIA_SRC]:
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from california_housing_template.data import CaliforniaHousingLoader
from california_housing_template.predict import load_assets as load_california_assets
from california_housing_template.train import train_california_housing_model
from ml_core.monitoring.calibration import regression_calibration_error
from ml_core.monitoring.drift import population_stability_index
from ml_core.monitoring.latency import benchmark_latency
from sales_forecasting_regression.data import SalesDatasetLoader
from sales_forecasting_regression.predict import load_assets as load_sales_assets
from sales_forecasting_regression.train import train_sales_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate test, coverage, and model quality reports.")
    parser.add_argument("--reports-dir", type=str, default="reports/quality")
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument("--latency-iterations", type=int, default=200)
    return parser.parse_args()


def run_pytest_with_reports(reports_dir: Path) -> dict[str, object]:
    junit_path = reports_dir / "pytest.junit.xml"
    coverage_path = reports_dir / "coverage.json"
    log_path = reports_dir / "pytest.log"

    pythonpath = os.pathsep.join([str(ML_CORE_SRC), str(SALES_SRC), str(CALIFORNIA_SRC)])
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{pythonpath}{os.pathsep}{env['PYTHONPATH']}"
        if env.get("PYTHONPATH")
        else pythonpath
    )

    command = [
        sys.executable,
        "-m",
        "pytest",
        "libs/ml_core/tests",
        "projects/sales_forecasting_regression/tests",
        "projects/california_housing_template/tests",
        "--cov=ml_core",
        "--cov=sales_forecasting_regression",
        "--cov=california_housing_template",
        "--cov-report=term",
        f"--cov-report=json:{coverage_path}",
        f"--junitxml={junit_path}",
    ]

    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    output_text = f"{completed.stdout}\n{completed.stderr}".strip()
    log_path.write_text(output_text, encoding="utf-8")

    if completed.returncode != 0:
        raise RuntimeError("Pytest failed. Check reports/quality/pytest.log")

    return {
        "junit_path": str(junit_path),
        "coverage_path": str(coverage_path),
        "log_path": str(log_path),
    }


def parse_junit_summary(junit_path: Path) -> dict[str, object]:
    root = element_tree.parse(junit_path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))

    tests = 0
    failures = 0
    errors = 0
    skipped = 0
    duration = 0.0

    for suite in suites:
        tests += int(float(suite.attrib.get("tests", 0)))
        failures += int(float(suite.attrib.get("failures", 0)))
        errors += int(float(suite.attrib.get("errors", 0)))
        skipped += int(float(suite.attrib.get("skipped", 0)))
        duration += float(suite.attrib.get("time", 0.0))

    passed = tests - failures - errors - skipped

    return {
        "tests": tests,
        "passed": passed,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
        "duration_seconds": round(duration, 3),
    }


def parse_coverage_summary(coverage_path: Path) -> dict[str, object]:
    payload = json.loads(coverage_path.read_text(encoding="utf-8"))
    totals = payload.get("totals", {})

    return {
        "covered_lines": int(totals.get("covered_lines", 0)),
        "num_statements": int(totals.get("num_statements", 0)),
        "percent_covered": round(float(totals.get("percent_covered", 0.0)), 2),
        "percent_covered_display": str(totals.get("percent_covered_display", "0")),
    }


def build_sales_quality(reports_dir: Path, latency_iterations: int) -> dict[str, object]:
    sales_reports_dir = reports_dir / "sales"
    sales_reports_dir.mkdir(parents=True, exist_ok=True)

    train_payload = train_sales_model(
        data_path=sales_reports_dir / "missing.csv",
        artifact_dir=sales_reports_dir / "models",
        report_path=sales_reports_dir / "train_metrics.json",
        enable_mlflow=False,
    )

    model, metadata = load_sales_assets(artifact_dir=sales_reports_dir / "models")
    feature_names = list(metadata["feature_names"])

    sample_input = {
        "QUANTITYORDERED": 34,
        "PRICEEACH": 99.2,
        "ORDERLINENUMBER": 3,
        "MSRP": 130.0,
        "QTR_ID": 2,
        "MONTH_ID": 5,
        "YEAR_ID": 2004,
        "PRODUCTLINE": "Classic Cars",
        "COUNTRY": "USA",
        "DEALSIZE": "Medium",
    }

    sample_frame = pd.DataFrame([{name: sample_input[name] for name in feature_names}])
    sample_prediction = float(model.predict(sample_frame)[0])

    sales_eval_frame = SalesDatasetLoader(random_state=42).synthetic_frame(600)
    sales_eval_x = sales_eval_frame.loc[:, feature_names]
    sales_eval_y = sales_eval_frame.loc[:, "SALES"].to_numpy()
    sales_eval_pred = model.predict(sales_eval_x)

    calibration_error = regression_calibration_error(sales_eval_y, sales_eval_pred)
    psi_value = population_stability_index(
        sales_eval_frame["PRICEEACH"].to_numpy(),
        (sales_eval_frame["PRICEEACH"] * 1.05).to_numpy(),
    )

    latency = benchmark_latency(
        predict_fn=lambda frame: model.predict(frame),
        payload=sample_frame,
        iterations=latency_iterations,
    )

    payload = {
        "best_model": train_payload["best_model"],
        "best_metrics": train_payload["best_metrics"],
        "sample_prediction": sample_prediction,
        "monitoring": {
            "regression_calibration_error": round(float(calibration_error), 6),
            "priceeach_psi": round(float(psi_value), 6),
        },
        "latency_ms": {key: round(float(value), 6) for key, value in latency.items()},
    }

    (sales_reports_dir / "quality_metrics.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    return payload


def build_california_quality(reports_dir: Path, latency_iterations: int) -> dict[str, object]:
    california_reports_dir = reports_dir / "california"
    california_reports_dir.mkdir(parents=True, exist_ok=True)

    train_payload = train_california_housing_model(
        artifact_dir=california_reports_dir / "models",
        use_synthetic_if_fetch_fails=True,
        force_synthetic=True,
    )

    (california_reports_dir / "train_metrics.json").write_text(
        json.dumps(train_payload, indent=2),
        encoding="utf-8",
    )

    model, metadata = load_california_assets(artifact_dir=california_reports_dir / "models")
    feature_names = list(metadata["feature_names"])

    sample_input = {
        "MedInc": 8.1,
        "HouseAge": 21,
        "AveRooms": 6.2,
        "AveBedrms": 1.1,
        "Population": 680,
        "AveOccup": 2.4,
        "Latitude": 37.75,
        "Longitude": -122.45,
    }

    sample_frame = pd.DataFrame([{name: sample_input[name] for name in feature_names}])
    sample_prediction = float(model.predict(sample_frame)[0])

    california_eval_frame = CaliforniaHousingLoader(random_state=42).load(force_synthetic=True)
    california_eval_x = california_eval_frame.loc[:, feature_names]
    california_eval_y = california_eval_frame.loc[:, "MedHouseVal"].to_numpy()
    california_eval_pred = model.predict(california_eval_x)

    calibration_error = regression_calibration_error(california_eval_y, california_eval_pred)
    psi_value = population_stability_index(
        california_eval_frame["MedInc"].to_numpy(),
        (california_eval_frame["MedInc"] + 0.5).to_numpy(),
    )

    latency = benchmark_latency(
        predict_fn=lambda frame: model.predict(frame),
        payload=sample_frame,
        iterations=latency_iterations,
    )

    payload = {
        "best_model": train_payload["best_model"],
        "best_metrics": train_payload["best_metrics"],
        "sample_prediction": sample_prediction,
        "monitoring": {
            "regression_calibration_error": round(float(calibration_error), 6),
            "medinc_psi": round(float(psi_value), 6),
        },
        "latency_ms": {key: round(float(value), 6) for key, value in latency.items()},
    }

    (california_reports_dir / "quality_metrics.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    return payload


def build_readme_snapshot(report: dict[str, object]) -> str:
    tests = report["tests"]
    coverage = report["coverage"]
    sales = report["sales"]
    california = report["california"]

    lines = [
        "# Quality Snapshot",
        "",
        f"- Generated at (UTC): {report['generated_at']}",
        f"- Test status: {tests['passed']}/{tests['tests']} passed",
        f"- Coverage: {coverage['percent_covered']}% ({coverage['covered_lines']}/{coverage['num_statements']} lines)",
        "",
        "## Sales Forecasting Regression",
        f"- Best model: {sales['best_model']}",
        f"- RMSE: {sales['best_metrics']['rmse']:.4f}",
        f"- MAE: {sales['best_metrics']['mae']:.4f}",
        f"- R2: {sales['best_metrics']['r2']:.4f}",
        f"- Sample prediction: {sales['sample_prediction']:.4f}",
        f"- Latency p95 (ms): {sales['latency_ms']['p95_ms']:.4f}",
        f"- Calibration error: {sales['monitoring']['regression_calibration_error']:.4f}",
        f"- PSI (PRICEEACH): {sales['monitoring']['priceeach_psi']:.4f}",
        "",
        "## California Housing Template",
        f"- Best model: {california['best_model']}",
        f"- RMSE: {california['best_metrics']['rmse']:.4f}",
        f"- MAE: {california['best_metrics']['mae']:.4f}",
        f"- R2: {california['best_metrics']['r2']:.4f}",
        f"- Sample prediction: {california['sample_prediction']:.4f}",
        f"- Latency p95 (ms): {california['latency_ms']['p95_ms']:.4f}",
        f"- Calibration error: {california['monitoring']['regression_calibration_error']:.4f}",
        f"- PSI (MedInc): {california['monitoring']['medinc_psi']:.4f}",
    ]

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    reports_dir = (REPO_ROOT / args.reports_dir).resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)

    tests_summary: dict[str, object]
    coverage_summary: dict[str, object]

    if args.skip_pytest:
        junit_path = reports_dir / "pytest.junit.xml"
        coverage_path = reports_dir / "coverage.json"
        if not junit_path.exists() or not coverage_path.exists():
            raise FileNotFoundError("Use --skip-pytest only when junit and coverage reports already exist.")
    else:
        run_pytest_with_reports(reports_dir)

    tests_summary = parse_junit_summary(reports_dir / "pytest.junit.xml")
    coverage_summary = parse_coverage_summary(reports_dir / "coverage.json")

    sales_summary = build_sales_quality(reports_dir, latency_iterations=args.latency_iterations)
    california_summary = build_california_quality(reports_dir, latency_iterations=args.latency_iterations)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tests": tests_summary,
        "coverage": coverage_summary,
        "sales": sales_summary,
        "california": california_summary,
    }

    (reports_dir / "quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (reports_dir / "README_SNAPSHOT.md").write_text(build_readme_snapshot(report), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
