# California Housing Template

This module is a reference template to demonstrate the same production contract (schema, artifacts, tests, API) on a small dataset.

## Run

```bash
make train APP=housing
make serve APP=housing

# Optional standardized selection mode
python -m california_housing_template.train --selection-mode both

# Optional reproducible EDA report
python scripts/eda/generate_eda_report.py --project california --output-dir reports/eda --report-version 1.0.0

# Optional DVC stage execution
dvc repro train_california eda_california monitor_california quality
```

## Artifacts

- `models/california_model.joblib`
- `models/california_metadata.json`
- `models/california_model.onnx` (if ONNX export dependencies are installed)
- `models/schema.json`
- `models/explainability/shap_summary.json`
- `models/explainability/shap_feature_importance.csv` (when SHAP is available and model is tree-based)
- `reports/eda/*` (versioned EDA json/markdown/plots/manifest)

## Swagger

- `http://127.0.0.1:8001/docs`

## curl

```bash
curl -s http://127.0.0.1:8001/health

curl -s -X POST http://127.0.0.1:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "MedInc": 8.0,
    "HouseAge": 22,
    "AveRooms": 5.5,
    "AveBedrms": 1.1,
    "Population": 700,
    "AveOccup": 2.1,
    "Latitude": 37.7,
    "Longitude": -122.4
  }'
```

Response fields include `prediction`, `y_hat`, `p10`, `p90`, and `interval_method`
(`conformal_intervals`, `ensemble_quantiles`, or `residual_quantiles`).

```bash
curl -s -X POST http://127.0.0.1:8001/explain \
  -H "Content-Type: application/json" \
  -d '{
    "MedInc": 8.0,
    "HouseAge": 22,
    "AveRooms": 5.5,
    "AveBedrms": 1.1,
    "Population": 700,
    "AveOccup": 2.1,
    "Latitude": 37.7,
    "Longitude": -122.4
  }'
```

`/explain` returns SHAP contributions for tree-based models when `shap` is installed.
