from pathlib import Path

from california_housing_template.train import train_california_housing_model


def test_california_training_registers_model_versions(tmp_path: Path) -> None:
    payload = train_california_housing_model(
        artifact_dir=tmp_path / "models",
        use_synthetic_if_fetch_fails=True,
        force_synthetic=True,
        selection_mode="baseline",
        registry_dir=tmp_path / "registry",
        promote_staging=True,
        promote_prod=True,
    )

    assert payload["registry"] is not None
    registration = payload["registry"]["registration"]
    assert registration["version"].startswith("v")
    aliases = payload["registry"]["aliases"]
    assert aliases["prod"] == registration["version"]
