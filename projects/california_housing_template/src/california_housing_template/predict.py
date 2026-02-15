from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from california_housing_template.config import CaliforniaPaths
from ml_core.registry.artifacts import ArtifactRegistry


def load_assets(artifact_dir: Path | None = None) -> tuple[Any, dict[str, Any]]:
    paths = CaliforniaPaths.default()
    registry = ArtifactRegistry(
        root_dir=artifact_dir or paths.models_dir,
        model_filename="california_model.joblib",
        metadata_filename="california_metadata.json",
        onnx_filename="california_model.onnx",
    )
    model = registry.load_model()
    metadata = registry.load_metadata()
    return model, metadata


def predict_one(features: dict[str, Any], artifact_dir: Path | None = None) -> float:
    model, metadata = load_assets(artifact_dir=artifact_dir)
    feature_names = list(metadata["feature_names"])
    missing = [name for name in feature_names if name not in features]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    row = {name: features[name] for name in feature_names}
    frame = pd.DataFrame([row])
    prediction = float(model.predict(frame)[0])
    return prediction


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict California house value using trained artifact.")
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument("--json-file", type=str, default=None)
    parser.add_argument("--artifact-dir", type=str, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if bool(args.json) == bool(args.json_file):
        raise ValueError("Provide either --json or --json-file.")

    if args.json:
        payload = json.loads(args.json)
    else:
        payload = json.loads(Path(args.json_file).read_text(encoding="utf-8"))

    prediction = predict_one(
        payload,
        artifact_dir=Path(args.artifact_dir) if args.artifact_dir else None,
    )
    print(json.dumps({"prediction": prediction}, indent=2))


if __name__ == "__main__":
    main()
