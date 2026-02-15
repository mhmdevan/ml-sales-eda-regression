import json
from pathlib import Path

from sales_forecasting_regression.data import SalesDatasetLoader
from sales_forecasting_regression.time_series import generate_segment_time_series_report


def test_generate_segment_time_series_report_writes_json(tmp_path: Path) -> None:
    frame = SalesDatasetLoader(random_state=42).synthetic_frame(2500)
    output_path = tmp_path / "reports" / "time_series_risk.json"

    report = generate_segment_time_series_report(
        frame=frame,
        output_path=output_path,
        product_line="Classic Cars",
        country="USA",
        holdout_periods=6,
        forecast_horizon=3,
    )

    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["segment"]["product_line"] == "Classic Cars"
    assert payload["segment"]["country"] == "USA"
    assert payload["selection"]["best_model"] in {"naive", "sarima", None}
    assert payload["forecast"]["horizon"] == 3
    assert len(payload["forecast"]["values"]) in {0, 3}
    assert "risk" in payload
    assert report["segment"]["product_line"] == "Classic Cars"


def test_generate_segment_time_series_report_handles_empty_segment(tmp_path: Path) -> None:
    frame = SalesDatasetLoader(random_state=42).synthetic_frame(500)
    output_path = tmp_path / "reports" / "time_series_empty.json"

    report = generate_segment_time_series_report(
        frame=frame,
        output_path=output_path,
        product_line="NoSuchProductLine",
        country="Nowhere",
        holdout_periods=4,
        forecast_horizon=2,
    )

    assert output_path.exists()
    assert report["series_points"] == 0
    assert report["selection"]["best_model"] is None
    assert report["forecast"]["values"] == []
