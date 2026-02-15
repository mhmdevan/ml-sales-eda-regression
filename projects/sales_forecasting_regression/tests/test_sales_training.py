import json
from pathlib import Path

from sales_forecasting_regression.train import train_sales_model


def test_train_sales_model_writes_artifacts(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "models"
    report_path = tmp_path / "reports" / "metrics.json"

    payload = train_sales_model(
        data_path=tmp_path / "missing.csv",
        artifact_dir=artifact_dir,
        report_path=report_path,
        selection_mode="baseline",
        enable_mlflow=False,
    )

    assert payload["best_metrics"]["rmse"] > 0
    assert (artifact_dir / "sales_regressor.joblib").exists()
    assert (artifact_dir / "sales_regressor_metadata.json").exists()
    assert (artifact_dir / "schema.json").exists()
    assert (artifact_dir / "explainability" / "shap_summary.json").exists()
    assert report_path.exists()
    time_series_path = report_path.parent / "time_series_risk.json"
    assert time_series_path.exists()

    metadata = payload["metadata"]
    extra = metadata["extra"]
    shap_summary = json.loads((artifact_dir / "explainability" / "shap_summary.json").read_text(encoding="utf-8"))
    time_series_report = json.loads(time_series_path.read_text(encoding="utf-8"))

    assert extra["dataset_fingerprint"]["algorithm"] == "sha256"
    assert len(extra["dataset_fingerprint"]["value"]) == 64
    assert extra["training_context"]["selection_mode"] == "baseline"
    assert extra["training_context"]["duration_seconds"] >= 0
    assert "started_at" in extra["training_context"]
    assert "finished_at" in extra["training_context"]
    assert extra["feature_contract"]["feature_names"] == payload["feature_names"]
    assert extra["feature_contract"]["target_name"] == payload["target"]
    assert set(extra["feature_contract"]["dtypes"]) == set(payload["feature_names"])
    assert isinstance(extra["model_hyperparameters"], dict)
    assert set(extra["inference_input_example"]) == set(payload["feature_names"])
    assert extra["metric_summary"]["best_model"] == payload["best_model"]
    assert extra["metric_summary"]["primary_metric"] == "rmse"
    assert extra["metric_summary"]["best_value"] == payload["best_metrics"]["rmse"]
    assert extra["uncertainty"]["method"] == "residual_quantiles"
    assert "residual_q10" in extra["uncertainty"]
    assert "residual_q90" in extra["uncertainty"]
    assert extra["conformal_intervals"]["method"] == "split_conformal_abs_residual"
    assert extra["conformal_intervals"]["alpha"] == 0.1
    assert extra["conformal_intervals"]["q_hat"] >= 0
    assert extra["conformal_intervals"]["calibration_size"] > 0
    assert "available" in extra["shap_explainability"]
    assert extra["shap_explainability"]["available"] == shap_summary["available"]
    assert payload["artifacts"]["shap_summary"] == str(artifact_dir / "explainability" / "shap_summary.json")
    assert payload["artifacts"]["time_series_risk_report"] == str(time_series_path)
    assert time_series_report["selection"]["best_model"] in {"naive", "sarima", None}
    assert "risk" in time_series_report
