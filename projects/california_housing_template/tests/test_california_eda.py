from pathlib import Path

from california_housing_template.eda import generate_california_eda_report


def test_generate_california_eda_report_writes_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "eda" / "california"
    payload = generate_california_eda_report(
        output_dir=output_dir,
        report_version="test-v1",
        force_synthetic=True,
        use_synthetic_if_fetch_fails=True,
    )

    assert (output_dir / "eda_report.json").exists()
    assert (output_dir / "EDA_REPORT.md").exists()
    assert (output_dir / "plots.json").exists()
    report = payload["report"]
    assert report["project_name"] == "california_housing_template"
    assert report["report_version"] == "test-v1"
    assert report["row_count"] > 0
    assert "dataset_fingerprint" in report
