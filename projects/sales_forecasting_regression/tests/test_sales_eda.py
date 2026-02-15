from pathlib import Path

from sales_forecasting_regression.eda import generate_sales_eda_report


def test_generate_sales_eda_report_writes_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "eda" / "sales"
    payload = generate_sales_eda_report(
        output_dir=output_dir,
        report_version="test-v1",
        csv_path=tmp_path / "missing.csv",
        use_synthetic_if_missing=True,
    )

    assert (output_dir / "eda_report.json").exists()
    assert (output_dir / "EDA_REPORT.md").exists()
    assert (output_dir / "plots.json").exists()
    report = payload["report"]
    assert report["project_name"] == "sales_forecasting_regression"
    assert report["report_version"] == "test-v1"
    assert report["row_count"] > 0
    assert "dataset_fingerprint" in report
