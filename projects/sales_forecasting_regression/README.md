# Sales Forecasting Regression

This is the flagship production use-case in the monorepo. It uses the shared `ml_core` contract for feature schema, training, artifact registry, API serving, and tests.

## Run

```bash
make train APP=sales
make serve APP=sales

# Optional standardized selection mode
python -m sales_forecasting_regression.train --selection-mode both

# Optional MLflow logging with signature + artifact set
python -m sales_forecasting_regression.train \
  --enable-mlflow \
  --mlflow-tracking-uri file:mlruns \
  --mlflow-experiment-name sales_forecasting_regression

# Optional segment time-series + risk report
python -m sales_forecasting_regression.time_series --product-line "Classic Cars" --country "USA"

# Optional reproducible EDA report
python scripts/eda/generate_eda_report.py --project sales --output-dir reports/eda --report-version 1.0.0

# Optional DVC stage execution
dvc repro train_sales eda_sales monitor_sales quality
```

## Artifacts

- `models/sales_regressor.joblib`
- `models/sales_regressor_metadata.json`
- `models/sales_regressor.onnx` (if ONNX export dependencies are installed)
- `models/schema.json`
- `models/explainability/shap_summary.json`
- `models/explainability/shap_feature_importance.csv` (when SHAP is available and model is tree-based)
- `reports/sales_metrics.json`
- `reports/time_series_risk.json`
- `reports/eda/*` (versioned EDA json/markdown/plots/manifest)

## Swagger

- `http://127.0.0.1:8000/docs`

## curl

```bash
curl -s http://127.0.0.1:8000/health

curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "QUANTITYORDERED": 30,
    "PRICEEACH": 100.5,
    "ORDERLINENUMBER": 2,
    "MSRP": 120.0,
    "QTR_ID": 2,
    "MONTH_ID": 5,
    "YEAR_ID": 2004,
    "PRODUCTLINE": "Classic Cars",
    "COUNTRY": "USA",
    "DEALSIZE": "Medium"
  }'
```

Response fields include `prediction`, `y_hat`, `p10`, `p90`, and `interval_method`
(`conformal_intervals`, `ensemble_quantiles`, or `residual_quantiles`).

```bash
curl -s -X POST http://127.0.0.1:8000/explain \
  -H "Content-Type: application/json" \
  -d '{
    "QUANTITYORDERED": 30,
    "PRICEEACH": 100.5,
    "ORDERLINENUMBER": 2,
    "MSRP": 120.0,
    "QTR_ID": 2,
    "MONTH_ID": 5,
    "YEAR_ID": 2004,
    "PRODUCTLINE": "Classic Cars",
    "COUNTRY": "USA",
    "DEALSIZE": "Medium"
  }'
```

`/explain` returns SHAP contributions for tree-based models when `shap` is installed.
