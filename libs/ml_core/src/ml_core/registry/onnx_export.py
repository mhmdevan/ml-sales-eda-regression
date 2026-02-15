from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline


def export_pipeline_to_onnx(
    pipeline: Pipeline,
    sample_frame: pd.DataFrame,
    output_path: Path,
) -> Path | None:
    try:
        from skl2onnx import to_onnx
    except Exception:
        return None

    try:
        onnx_model = to_onnx(pipeline, sample_frame.iloc[:1], target_opset=17)
    except Exception:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(onnx_model.SerializeToString())
    return output_path
