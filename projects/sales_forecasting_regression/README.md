# Sales Forecasting Regression

This is the flagship production use-case in the monorepo. It uses the shared `ml_core` contract for feature schema, training, artifact registry, API serving, and tests.

## Run

```bash
python -m sales_forecasting_regression.train --enable-mlflow
uvicorn sales_forecasting_regression.api:app --reload
```

## Artifacts

- `models/sales_regressor.joblib`
- `models/sales_regressor_metadata.json`
- `models/sales_regressor.onnx` (if ONNX export dependencies are installed)
- `reports/sales_metrics.json`
