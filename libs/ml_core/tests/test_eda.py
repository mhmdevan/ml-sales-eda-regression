from pathlib import Path

import pandas as pd

from ml_core.monitoring.eda import build_tabular_eda_payload, write_eda_artifacts


def test_build_tabular_eda_payload_and_write_artifacts(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0, 4.0],
            "x2": [10.0, 11.0, 12.0, 13.0],
            "cat": ["a", "a", "b", "b"],
            "target": [100.0, 102.0, 104.0, 106.0],
        }
    )
    payload = build_tabular_eda_payload(
        frame=frame,
        project_name="test_project",
        report_version="v-test",
        numeric_features=["x1", "x2"],
        categorical_features=["cat"],
        target_name="target",
    )
    paths = write_eda_artifacts(
        frame=frame,
        numeric_features=["x1", "x2"],
        target_name="target",
        output_dir=tmp_path / "eda",
        payload=payload,
    )

    assert payload["project_name"] == "test_project"
    assert payload["report_version"] == "v-test"
    assert payload["row_count"] == 4
    assert Path(paths["json_path"]).exists()
    assert Path(paths["markdown_path"]).exists()
    assert Path(paths["plots_path"]).exists()
