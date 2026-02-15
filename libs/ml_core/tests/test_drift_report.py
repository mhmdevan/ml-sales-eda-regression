from pathlib import Path

import numpy as np
import pandas as pd

from ml_core.monitoring.drift_report import (
    build_drift_alert_payload,
    compute_numeric_drift_summary,
    write_alert_artifacts,
)


def test_drift_report_flags_shifted_feature(tmp_path: Path) -> None:
    reference = pd.DataFrame(
        {
            "x1": np.linspace(0, 10, 200),
            "x2": np.linspace(0, 5, 200),
        }
    )
    current = pd.DataFrame(
        {
            "x1": np.linspace(50, 60, 200),
            "x2": np.linspace(0, 5, 200),
        }
    )

    summary = compute_numeric_drift_summary(
        reference_frame=reference,
        current_frame=current,
        numeric_features=["x1", "x2"],
        psi_threshold=0.2,
    )

    assert "x1" in summary["drifted_features"]
    assert "x1" in summary["feature_ks"]
    assert summary["feature_ks"]["x1"]["drift_detected"] is True

    payload = build_drift_alert_payload(
        project_name="sales",
        drift_summary=summary,
        alert_ratio_threshold=0.3,
    )

    outputs = write_alert_artifacts(output_dir=tmp_path / "monitoring", payload=payload)

    assert Path(outputs["summary_path"]).exists()
    assert Path(outputs["alert_path"]).exists()
