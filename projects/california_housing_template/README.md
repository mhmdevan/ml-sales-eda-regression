# California Housing Template

This module is a reference template to demonstrate the same production contract (schema, artifacts, tests, API) on a small dataset.

## Run

```bash
python -m california_housing_template.train
uvicorn california_housing_template.api:app --reload --port 8001
```

## Artifacts

- `models/california_model.joblib`
- `models/california_metadata.json`
- `models/california_model.onnx` (if ONNX export dependencies are installed)
