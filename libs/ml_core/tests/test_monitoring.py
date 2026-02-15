import numpy as np

from ml_core.monitoring.calibration import regression_calibration_error
from ml_core.monitoring.drift import population_stability_index
from ml_core.monitoring.latency import benchmark_latency


def test_population_stability_index_non_negative() -> None:
    rng = np.random.default_rng(7)
    ref = rng.normal(0, 1, 500)
    cur = rng.normal(0.3, 1.1, 500)
    psi = population_stability_index(ref, cur, bins=10)
    assert psi >= 0


def test_regression_calibration_error_is_finite() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.2, 1.8, 3.1, 3.9, 4.7])
    error = regression_calibration_error(y_true, y_pred, n_bins=3)
    assert error >= 0


def test_latency_benchmark_returns_expected_keys() -> None:
    result = benchmark_latency(lambda x: x, payload={"x": 1}, iterations=5)
    assert set(result.keys()) == {"mean_ms", "p50_ms", "p95_ms", "max_ms"}
