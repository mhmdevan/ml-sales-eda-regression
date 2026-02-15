from pathlib import Path

from california_housing_template.train import train_california_housing_model


def test_training_creates_artifacts(tmp_path: Path) -> None:
    payload = train_california_housing_model(
        artifact_dir=tmp_path / "models",
        use_synthetic_if_fetch_fails=True,
        force_synthetic=True,
        selection_mode="baseline",
    )

    assert payload["best_metrics"]["rmse"] > 0
    assert (tmp_path / "models" / "california_model.joblib").exists()
    assert (tmp_path / "models" / "california_metadata.json").exists()
    assert (tmp_path / "models" / "explainability" / "shap_summary.json").exists()
    uncertainty = payload["metadata"]["extra"]["uncertainty"]
    assert uncertainty["method"] == "residual_quantiles"
    assert "residual_q10" in uncertainty
    assert "residual_q90" in uncertainty
    conformal = payload["metadata"]["extra"]["conformal_intervals"]
    assert conformal["method"] == "split_conformal_abs_residual"
    assert conformal["alpha"] == 0.1
    assert conformal["q_hat"] >= 0
    assert conformal["calibration_size"] > 0
    assert "available" in payload["metadata"]["extra"]["shap_explainability"]
