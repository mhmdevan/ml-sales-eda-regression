from pathlib import Path

from sales_forecasting_regression.train import train_sales_model


def test_sales_training_registers_model_versions(tmp_path: Path) -> None:
    payload = train_sales_model(
        data_path=tmp_path / "missing.csv",
        artifact_dir=tmp_path / "models",
        report_path=tmp_path / "reports" / "metrics.json",
        selection_mode="baseline",
        enable_mlflow=False,
        registry_dir=tmp_path / "registry",
        promote_staging=True,
        promote_prod=True,
    )

    assert payload["registry"] is not None
    registration = payload["registry"]["registration"]
    assert registration["version"].startswith("v")
    aliases = payload["registry"]["aliases"]
    assert aliases["prod"] == registration["version"]
