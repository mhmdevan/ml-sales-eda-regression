from pathlib import Path

import numpy as np
import pandas as pd

from ml_core.monitoring.calibration import regression_calibration_error
from ml_core.monitoring.conformal import conformal_interval, fit_split_conformal_interval
from ml_core.monitoring.distributions import build_feature_distribution_log, summarize_feature_distributions
from ml_core.monitoring.drift import kolmogorov_smirnov_test, population_stability_index
from ml_core.monitoring.latency import benchmark_latency
from ml_core.monitoring.schema_alert import (
    build_schema_alert_payload,
    detect_schema_changes,
    write_schema_alert_artifacts,
)


def test_population_stability_index_non_negative() -> None:
    rng = np.random.default_rng(7)
    ref = rng.normal(0, 1, 500)
    cur = rng.normal(0.3, 1.1, 500)
    psi = population_stability_index(ref, cur, bins=10)
    assert psi >= 0


def test_kolmogorov_smirnov_test_detects_shift() -> None:
    rng = np.random.default_rng(11)
    ref = rng.normal(0.0, 1.0, 500)
    cur = rng.normal(1.2, 1.0, 500)
    result = kolmogorov_smirnov_test(ref, cur, alpha=0.05)

    assert 0 <= result["statistic"] <= 1
    assert 0 <= result["p_value"] <= 1
    assert result["drift_detected"] is True


def test_regression_calibration_error_is_finite() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.2, 1.8, 3.1, 3.9, 4.7])
    error = regression_calibration_error(y_true, y_pred, n_bins=3)
    assert error >= 0


def test_latency_benchmark_returns_expected_keys() -> None:
    result = benchmark_latency(lambda x: x, payload={"x": 1}, iterations=5)
    assert set(result.keys()) == {"mean_ms", "p50_ms", "p95_ms", "max_ms"}


def test_split_conformal_interval_returns_expected_contract() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.8, 3.7, 5.1])
    payload = fit_split_conformal_interval(y_true, y_pred, alpha=0.1)

    assert payload["method"] == "split_conformal_abs_residual"
    assert payload["alpha"] == 0.1
    assert payload["q_hat"] >= 0
    assert payload["calibration_size"] == 5
    assert 0 <= payload["empirical_coverage"] <= 1


def test_conformal_interval_returns_symmetric_bounds() -> None:
    lower, upper = conformal_interval(prediction=10.0, q_hat=1.5)
    assert lower == 8.5
    assert upper == 11.5


def test_summarize_feature_distributions_returns_numeric_and_categorical() -> None:
    frame = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, np.nan],
            "cat": ["a", "a", None, "b"],
        }
    )

    summary = summarize_feature_distributions(frame=frame, feature_names=["num", "cat", "missing"])
    log_payload = build_feature_distribution_log(
        reference_frame=frame,
        current_frame=frame,
        feature_names=["num", "cat", "missing"],
    )

    assert summary["num"]["type"] == "numeric"
    assert summary["cat"]["type"] == "categorical"
    assert summary["missing"]["present"] is False
    assert set(log_payload.keys()) == {"generated_at", "feature_count", "reference", "current"}


def test_schema_alert_detects_changes_and_writes_artifacts(tmp_path: Path) -> None:
    current = pd.DataFrame(
        {
            "x1": pd.Series([1, 2], dtype="int64"),
            "x3": ["a", "b"],
        }
    )

    schema_check = detect_schema_changes(
        expected_features=["x1", "x2"],
        current_frame=current,
        expected_dtypes={"x1": "float64", "x2": "int64"},
    )

    payload = build_schema_alert_payload(project_name="sales", schema_check=schema_check)
    artifacts = write_schema_alert_artifacts(output_dir=tmp_path, payload=payload)

    assert payload["alert"] is True
    assert schema_check["missing_features"] == ["x2"]
    assert schema_check["unexpected_features"] == ["x3"]
    assert "x1" in schema_check["dtype_mismatches"]
    assert Path(artifacts["summary_path"]).exists()
    assert Path(artifacts["alert_path"]).exists()
