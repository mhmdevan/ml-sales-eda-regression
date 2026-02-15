from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from ml_core.features.preprocessor import build_preprocessor
from ml_core.features.schema import FeatureSchema
from ml_core.monitoring.explainability import explain_single_prediction, generate_shap_artifacts


def test_explain_single_prediction_returns_unavailable_for_non_tree_model() -> None:
    schema = FeatureSchema.create(["x"], [], "y")
    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(schema)),
            ("regressor", LinearRegression()),
        ]
    )
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]})
    model.fit(frame[["x"]], frame["y"])

    result = explain_single_prediction(
        pipeline=model,
        frame=pd.DataFrame({"x": [4.0]}),
        schema=schema,
    )

    assert result["available"] is False
    assert result["reason"] == "best model is not tree-based"


def test_generate_shap_artifacts_writes_summary_for_tree_model(tmp_path: Path) -> None:
    schema = FeatureSchema.create(["x"], [], "y")
    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(schema)),
            ("regressor", RandomForestRegressor(random_state=42, n_estimators=20)),
        ]
    )
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [3.0, 5.0, 7.0, 9.0, 11.0]})
    model.fit(frame[["x"]], frame["y"])

    payload = generate_shap_artifacts(
        pipeline=model,
        frame=frame[["x"]],
        schema=schema,
        output_dir=tmp_path / "explainability",
    )

    assert Path(payload["summary_path"]).exists()
    assert "available" in payload["summary"]
