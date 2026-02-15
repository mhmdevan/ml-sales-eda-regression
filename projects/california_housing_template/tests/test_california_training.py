from pathlib import Path

from california_housing_template.train import train_california_housing_model


def test_training_creates_artifacts(tmp_path: Path) -> None:
    payload = train_california_housing_model(
        artifact_dir=tmp_path / "models",
        use_synthetic_if_fetch_fails=True,
        force_synthetic=True,
    )

    assert payload["best_metrics"]["rmse"] > 0
    assert (tmp_path / "models" / "california_model.joblib").exists()
    assert (tmp_path / "models" / "california_metadata.json").exists()
