from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from ml_core.data.splitters import split_regression_frame
from ml_core.features.schema import FeatureSchema
from ml_core.modeling.contracts import CandidateModel
from ml_core.modeling.trainer import RegressionTrainer
from ml_core.registry.artifacts import ArtifactRegistry


def test_training_and_registry_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(
        {
            "x1": rng.normal(size=200),
            "x2": rng.normal(size=200),
            "cat": rng.choice(["a", "b", "c"], size=200),
        }
    )
    frame["target"] = 2.5 * frame["x1"] - 1.7 * frame["x2"] + rng.normal(0, 0.1, size=200)

    schema = FeatureSchema.create(["x1", "x2"], ["cat"], "target")
    split = split_regression_frame(frame, schema)

    candidates = [
        CandidateModel(
            name="LinearRegression",
            estimator_factory=lambda _: LinearRegression(),
            parameter_grid=({},),
        ),
        CandidateModel(
            name="RandomForestRegressor",
            estimator_factory=lambda params: RandomForestRegressor(random_state=42, **params),
            parameter_grid=({"n_estimators": 30},),
        ),
    ]

    trainer = RegressionTrainer(schema=schema, candidates=candidates)
    result = trainer.fit(split)

    registry = ArtifactRegistry(tmp_path / "models")
    metadata = registry.save_training_result(
        result,
        project_name="ml_core_test",
        export_onnx=True,
        sample_frame=split.x_train,
    )

    assert (tmp_path / "models" / "model.joblib").exists()
    assert (tmp_path / "models" / "metadata.json").exists()
    assert (tmp_path / "models" / "schema.json").exists()
    assert metadata["project_name"] == "ml_core_test"
    assert registry.model_exists()

    model = registry.load_model()
    predictions = model.predict(split.x_test.head(2))
    assert len(predictions) == 2
