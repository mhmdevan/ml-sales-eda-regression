# 📊 Sales Analytics & ML – From EDA Script to Mini MLOps Project

End‑to‑end **sales analytics + regression** project built on the Kaggle *Sample Sales Data*.

The idea is simple:

> Take one CSV and turn it into a **small, production‑style analytics & ML service**  
> – with proper data cleaning, EDA, regression models, experiment tracking, explainability, APIs, and even time‑series risk modeling.

This repo shows two evolutions of the same idea:

- **v1 – Script‑style, config‑driven pipeline**  
  Pure Python modules (no notebooks) for EDA, feature engineering, regression, monitoring, Streamlit dashboard, and a simple FastAPI service.
- **v2 – “Mini MLOps” version**  
  Data layer rebuilt with **Polars + Parquet + DuckDB**, training refactored into a **sklearn + MLflow** pipeline with multiple models (RF, GBT, XGB, LGBM, CatBoost), **SHAP** explainability, **ONNX** export, and a proper **FastAPI v2** endpoint + CLI.

Both versions are kept side‑by‑side so you can show an interviewer **how you evolve a quick analysis into a more robust, production‑shaped project**.

---

## 🎯 What Problem Does This Project Solve? (Resume Story)

> “Given a messy sales CSV from a business system, build a clean, reproducible pipeline that:  
> – runs EDA,  
> – learns to predict order‑level `SALES`,  
> – tracks experiments,  
> – exposes predictions over HTTP,  
> – and provides some time‑series & risk insight for a selected segment.”

Concretely:

1. **Data problem**  
   - Source: one CSV exported from a transactional system (`sales_data_sample.csv`).  
   - Issues: missing values, messy date strings, non‑UTF8 characters, outliers, high‑cardinality categorical columns.

2. **ML problem**  
   - Predict continuous `SALES` for a single order line based on quantity, price, product line, country, etc.
   - Compare different models: Linear Regression vs forests vs gradient boosting vs modern boosting libraries.

3. **MLOps / engineering problem**  
   - Turn the experiment into something **repeatable and deployable**:
     - config‑driven,
     - structured package,
     - versioned models & metrics,
     - online API + CLI,
     - interpretable (SHAP),
     - time‑series & volatility insight on top of the same data.

This README is written **resume‑first**: you can literally use the sections below as a script to explain the project in an interview.

---

## 🔗 Dataset

- **Name**: *Sample Sales Data*  
- **Source**: Kaggle – `sales_data_sample.csv`  
- **Typical columns**: `ORDERNUMBER`, `ORDERDATE`, `QUANTITYORDERED`, `PRICEEACH`, `SALES`, `MSRP`, `PRODUCTLINE`, `DEALSIZE`, `COUNTRY`, etc.
- **Target**: `SALES` (continuous).

⚠️ **Kaggle license note**:  
The raw CSV is **not** committed. You must download it manually (see *Setup*).

---

## 🧱 High‑Level Features (What This Repo Demonstrates)

**v1 core**

- Config‑driven **EDA** (`config.yaml`):
  - missing value handling,
  - IQR‑based outlier removal,
  - grouped statistics (by product line, country, year),
  - Matplotlib/Seaborn charts (histograms, scatter, boxplots).
- Feature engineering for regression (date features, ratios, line totals).
- Several regression models on `SALES` (Linear Regression, Random Forest, Gradient Boosting + basic tuning).
- Simple **drift / monitoring** demo (PSI on `SALES` over time).
- **Streamlit** dashboard for interactive exploration.
- **FastAPI** v1 for a quick prediction API.

**v2 extensions**

- **Polars** for fast CSV ingest + cleaning.
- **Partitioned Parquet** as the canonical cleaned data layer.
- **DuckDB** for SQL‑style analytics on top of Parquet (no separate DB server).
- **Config‑driven** project (`config_v2.yaml`) with typed loader (`src/config_v2.py`).
- Scikit‑learn pipelines with **ColumnTransformer** (numeric + categorical).
- Multiple ML models:
  - Linear Regression, RandomForest, GradientBoosting, XGBoost, LightGBM, CatBoost.
- **MLflow** tracking (file‑based `mlruns` store):
  - params, metrics (RMSE, MAE, R²), artifacts.
- **SHAP** global feature importance for tree‑based best model.
- **ONNX** export of the best pipeline (for serving in non‑Python runtimes).
- **CLI** for offline prediction (`python -m src.predict_sales`).
- **FastAPI v2** service (`src.api_sales_v2.py`) with `/health` and `/predict`.
- **Time‑series + risk module**:
  - SARIMA forecasting on aggregated monthly `SALES`,
  - naive baseline comparison,
  - GARCH(1,1) on monthly returns to estimate volatility.

---

## 🗂 Project Structure (Combined v1 + v2)

At a high level (names simplified):

```text
v1/                            # v1 (script-style)
  data/
    sales_data_sample.csv      # downloaded from Kaggle (gitignored)
  output/
    hist_*.png                 # v1 plots
    summary/                   # v1 summary tables
    regression/                # v1 regression metrics + models
    monitoring/                # PSI reports
  config.yaml                  # v1 config
  sales_eda.py                 # v1 EDA pipeline
  sales_regression.py          # v1 regression script
  sales_monitoring.py          # v1 drift / PSI demo
  sales_api.py                 # v1 FastAPI
  streamlit_app.py             # v1 dashboard
  pyproject.toml
  tests/
    test_feature_engineering.py
    test_regression_pipeline.py
  README.md                    # this file

v2/                            # v2 – data / MLflow / TS stack
  config_v2.yaml
  data/
    raw/
      sales_data_sample.csv
    processed/
      sales_parquet/
        sales_YEAR=2003.parquet
        sales_YEAR=2004.parquet
        ...
  duckdb/
    sales.duckdb
  models/
    sales_regressor.joblib
    sales_regressor_metadata.json
    sales_regressor.onnx
  output/
    summary/
      by_productline.{csv,parquet}
      top_countries.{csv,parquet}
      by_year.{csv,parquet}
    plots/
      total_sales_by_productline_duckdb.png
    metrics/
      sales_regression_metrics.json
    explainability/
      shap_global_importance.json
    mlflow_input_examples/
      *_input_example.parquet
    mlflow_models/
      *_pipeline.joblib
    timeseries/
      sales_ts_risk_*_*.json
  mlruns/                      # local MLflow store
  src/
    logging_utils.py
    config_v2.py
    data_parquet.py
    duckdb_utils.py
    eda_polars_duckdb.py

    api_sales_v2.py
    predict_sales.py
    train_sales_regression_mlflow.py
    train_sales_timeseries_risk.py

    sales_regression/
      schema.py
      preprocessing.py
      models.py
      mlflow_runner.py
      pipeline.py
      export.py
      explainability.py
      report.py
      timeseries.py
```

When talking to an interviewer, you can say:  
> “v1 is an educational, script‑style pipeline. v2 refactors it into a more realistic data+ML stack with Polars, Parquet, DuckDB and MLflow, while keeping the same business problem.”

---

## ⚙️ Setup & Installation

### 1. Create virtual environment & install

From the repo root (containing `pyproject.toml`):

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install both v1 & v2 package in editable mode
pip install -e .
```

Major dependencies include:

- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `polars`, `duckdb`
- `scikit-learn`, `joblib`
- `xgboost`, `lightgbm`, `catboost`
- `mlflow`, `shap`, `skl2onnx`, `onnx`, `onnxruntime`
- `fastapi`, `uvicorn`
- `pyyaml`

If you only want the minimal v1 stack, a rough requirements set is:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib pyarrow            pyyaml streamlit fastapi uvicorn
```

### 2. Download the Kaggle CSV

1. Open: <https://www.kaggle.com/datasets/kyanyoga/sample-sales-data>
2. Download `sales_data_sample.csv`.
3. For v1, place it here:

   ```text
   v1/data/sales_data_sample.csv
   ```

4. For v2, place it here:

   ```text
   v2/data/raw/sales_data_sample.csv
   ```

Paths are configurable via `config.yaml` (v1) and `config_v2.yaml` (v2).

---

## 🧾 Config‑Driven Pipeline (v1 vs v2)

### v1 – `config.yaml`

Controls:

- which dataset to use (`active_dataset`)
- list of numeric columns
- cleaning rules:
  - columns to drop,
  - columns to fill with `"Unknown"`,
  - postal code handling.
- outlier detection parameters (IQR factor).
- which histograms / scatter / boxplots to generate.

The EDA script reads `config.yaml`, merges top‑level settings with the selected dataset block, and drives the whole cleaning/EDA process from that.

### v2 – `config_v2.yaml`

Moves configuration into a **typed dataclass layer**:

- `paths`: raw CSV, processed Parquet directory, DuckDB path, summary / plots directories.
- `cleaning`: columns to drop, `fill_unknown_columns`, date column name + format.
- `parquet`: partitioning rule (e.g. `["YEAR"]`).
- `duckdb`: database file, table name, parquet glob.
- `aggregations`: e.g. `top_countries_limit`.
- `model`: target column, numeric/categorical features, test size, random state.
- `mlflow`: tracking URI, experiment name.

`src/config_v2.py` resolves paths relative to the config file and puts everything into a `ProjectConfig` dataclass. That gives you type‑safe config access everywhere else.

---

## 🔍 v1 – EDA, Cleaning & Simple Regression

### Data cleaning logic (v1)

On a real run on `sample_sales`:

- **Raw shape**: `2823` rows × `25` columns.
- After adding date features (`YEAR`, `MONTH`) and dropping `ADDRESSLINE2`:
  - **Shape before outlier removal**: `2823` rows × `26` columns.

Missing values:

- Major missing columns:
  - `ADDRESSLINE2`: 2521
  - `STATE`: 1486
  - `TERRITORY`: 1074
  - `POSTALCODE`: 76

Cleaning steps:

- Drop `ADDRESSLINE2`.
- Fill `STATE` & `TERRITORY` with `"Unknown"`.
- Cast `POSTALCODE` to string and fill missing with `"Unknown"`.

After cleaning, business‑critical columns used in analysis (orders, quantities, dates, geography) have **0 missing values**.

Outlier removal: IQR rule (per column, sequentially):

- For each numeric column (`QUANTITYORDERED`, `PRICEEACH`, `SALES`, `MSRP`):
  - compute Q1, Q3, IQR = Q3 − Q1
  - bounds: `lower = Q1 − iqr_factor * IQR`, `upper = Q3 + iqr_factor * IQR`
  - drop rows outside `[lower, upper]`.

Example counts:

- `QUANTITYORDERED`: 8 rows dropped.
- `PRICEEACH`: 0 rows dropped.
- `SALES`: 79 rows dropped.
- `MSRP`: 35 rows dropped.

Final dataset:

- **Shape after outlier removal**: `2701` rows × `26` columns.

Descriptive stats (after cleaning & outliers):

- `QUANTITYORDERED`: mean ≈ `34.66`, 75% ≈ `43`, max `66`.
- `PRICEEACH`: mean ≈ `82.99`, 75% ≈ `100`, max `100`.
- `SALES`: mean ≈ `3354.54`, 75% ≈ `4300.5`, max ≈ `7901.1`.
- `MSRP`: mean ≈ `97.43`, 75% ≈ `121`, max `194`.

All numeric stats are saved as:

```text
output/summary/numeric_descriptive_stats.{csv,parquet}
```

### Visual EDA (v1)

`python sales_eda.py` will:

- generate histograms for `SALES` and `QUANTITYORDERED`,
- scatter plots:
  - `QUANTITYORDERED` vs `SALES`,
  - `PRICEEACH` vs `SALES`,
- boxplots:
  - `SALES` by `PRODUCTLINE`,
  - `QUANTITYORDERED` by `DEALSIZE`.

Files end up in `output/`:

- `hist_sales.png`
- `hist_quantityordered.png`
- `scatter_quantityordered_vs_sales.png`
- `scatter_priceeach_vs_sales.png`
- `box_sales_by_productline.png`
- `box_quantityordered_by_dealsize.png`

For interviews, this shows you can:

- explain distribution of key metrics,
- talk about product mix (`PRODUCTLINE`),
- discuss how deal size relates to order quantity.

### v1 Regression on `SALES`

Script: `sales_regression.py`

Features include:

- date‑derived features: `YEAR`, `MONTH`, `QUARTER`, `SEASON`,
- ratio / interaction features:
  - `PRICE_TO_MSRP_RATIO = PRICEEACH / MSRP`,
  - `LINE_TOTAL_APPROX = QUANTITYORDERED * PRICEEACH`,
- categorical encodings: `PRODUCTLINE`, `DEALSIZE`, `COUNTRY`, `SEASON`.

Models trained:

1. `LinearRegression_baseline`
2. `RandomForest_baseline`
3. `GradientBoosting_baseline`
4. `RandomForest_tuned` (RandomizedSearchCV)

Example metrics from a run:

| Model                     | MSE       | RMSE   | MAE    | R²    |
|---------------------------|-----------|--------|--------|-------|
| LinearRegression_baseline | 233263.80 | 482.97 | 340.78 | 0.903 |
| RandomForest_baseline     | 134122.17 | 366.23 | 188.66 | 0.944 |
| GradientBoosting_baseline | 126395.65 | 355.52 | 201.49 | 0.947 |
| RandomForest_tuned        | 137114.12 | 370.29 | 190.44 | 0.943 |

Key talking point:

> “Basic Linear Regression already gets ~0.90 R², but moving to tree‑based models (RF/GBT) reduces RMSE by ~25–30%. On this dataset, a fairly standard GradientBoostingRegressor wins without heavy tuning, which is realistic for medium‑sized tabular data.”

Best model in this v1 run: `GradientBoosting_baseline`.  
Saved as:

```text
output/regression/sales_regression_best_GradientBoosting_baseline.joblib
```

All metrics:

```text
output/regression/advanced_metrics.{csv,parquet}
```

### v1 Monitoring / Drift Demo

Script: `sales_monitoring.py`

- Splits the cleaned dataset into two periods:
  - earliest `YEAR` vs latest `YEAR` (or first half vs second half).
- Computes **Population Stability Index (PSI)** on the `SALES` distribution.
- Saves:
  - JSON report: `output/monitoring/psi_sales_*.json`,
  - comparison plot: `output/monitoring/sales_*.png`.

This is not production‑grade, but it lets you talk about **data drift and retraining triggers**.

### v1 Streamlit Dashboard

Script: `streamlit_app.py`

Run:

```bash
python -m streamlit run streamlit_app.py
```

Provides:

- Numeric summary stats,
- Top product lines / countries by sales,
- Simple charts (sales distribution, sales by year),
- Optional filters by `PRODUCTLINE`, etc.

### v1 FastAPI Prediction Service

Script: `sales_api.py`

Run training first (to make sure the best model exists), then:

```bash
uvicorn sales_api:app --reload
```

- Swagger UI at `/docs`
- `POST /predict` endpoint taking raw features and returning predicted `SALES` using the v1 best model.

---

## 🦆 v2 – Polars, Parquet, DuckDB & MLflow Stack

v2 refactors the project into a more realistic MLOps‑style architecture.

### Step 1 – Data → Parquet (Polars)

Script: `src/data_parquet.py`

Run:

```bash
cd v2
python -m src.data_parquet
```

What happens:

1. Load raw CSV (`data/raw/sales_data_sample.csv`) using Polars:
   - try strict UTF‑8,
   - on failure, retry with `encoding="utf8-lossy"` (handles weird Excel exports).
2. Apply cleaning rules from `config_v2.yaml`:
   - drop configured columns (`ADDRESSLINE2`),
   - fill `STATE` and `TERRITORY` `NULL` → `"Unknown"`,
   - parse `ORDERDATE` to a proper `Date` and add `YEAR`, `MONTH`.
3. Write cleaned data to **partitioned Parquet** under `data/processed/sales_parquet`:
   - e.g. `sales_YEAR=2003.parquet`, `sales_YEAR=2004.parquet`.

### Step 2 – DuckDB EDA

Script: `src/eda_polars_duckdb.py`

Run:

```bash
python -m src.eda_polars_duckdb
```

Pipeline:

1. Initialize / refresh a DuckDB database (`duckdb/sales.duckdb`).
2. Create (or replace) the `sales` table from Parquet:

   ```sql
   CREATE OR REPLACE TABLE sales AS
   SELECT * FROM read_parquet('data/processed/sales_parquet/sales_*.parquet');
   ```

3. Run SQL aggregations:
   - total & avg `SALES` by `PRODUCTLINE`,
   - top N `COUNTRY` by `SALES`,
   - sales by `YEAR`.
4. Save results to `output/summary/` as CSV and Parquet.
5. Plot total sales by product line to  
   `output/plots/total_sales_by_productline_duckdb.png`.

### Step 3 – Regression Training with MLflow (v2)

Entry point: `src/train_sales_regression_mlflow.py`

Run:

```bash
python -m src.train_sales_regression_mlflow
```

Internally, `sales_regression.pipeline.run_training_pipeline` does:

1. Load `ProjectConfig` from `config_v2.yaml`.
2. Read cleaned sales from partitioned Parquet (`load_clean_sales_from_parquet`).
3. Build `(X, y)` via `sales_regression.schema.build_feature_matrix`:

   - **Numeric features** (`NUMERIC_FEATURES`):
     - `["QUANTITYORDERED", "PRICEEACH", "ORDERLINENUMBER", "MSRP",
        "QTR_ID", "MONTH_ID", "YEAR_ID"]`
   - **Categorical features** (`CATEGORICAL_FEATURES`):
     - `["PRODUCTLINE", "COUNTRY", "DEALSIZE"]`
   - **Target**: `SALES`

4. Train/test split (`test_size`, `random_state` from config).
5. Build `ColumnTransformer` preprocessor:
   - numeric: `SimpleImputer(strategy="median")` + `StandardScaler`,
   - categorical: `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`.

6. Define model specs (`sales_regression.models.build_model_specs()`):
   - Linear Regression
   - RandomForestRegressor (two configs)
   - GradientBoostingRegressor (two configs)
   - XGBRegressor
   - LGBMRegressor
   - CatBoostRegressor

7. For each model spec:
   - Wrap into `Pipeline(preprocessor, regressor)`.
   - Train and evaluate (RMSE, MAE, R²).
   - Log everything to MLflow (`mlflow_runner.train_and_log_one_model`).

8. Select best model by **RMSE**.

9. Persist artifacts:
   - `output/metrics/sales_regression_metrics.json` – metrics summary for all models.
   - `models/sales_regressor.joblib` – best pipeline.
   - `models/sales_regressor_metadata.json` – schema, metrics snapshot, run id.
   - `output/explainability/shap_global_importance.json` – SHAP global importances (tree‑based models only).
   - `models/sales_regressor.onnx` – ONNX export using `skl2onnx`.

This is the “serious” training pipeline you showcase for ML roles.

### Step 4 – MLflow UI

To inspect all runs:

```bash
mlflow ui --backend-store-uri file:mlruns
```

- You’ll see each model key (`RandomForest_300_depth10`, `XGBoost_300_hist`, etc.) as a run.
- You can sort by RMSE and cross‑check that the code’s selected best model matches MLflow’s best run.

### Step 5 – Explainability (SHAP) & Export (ONNX)

Script: `sales_regression.explainability`

- If the best model is tree‑based, calculate SHAP values for up to 500 training samples.
- Collapse one‑hot columns back to original features (e.g. `PRODUCTLINE`, `COUNTRY`).
- Save a ranked list of features by mean |SHAP| to `output/explainability/shap_global_importance.json`.

Script: `sales_regression.export.export_best_pipeline_to_onnx`

- Builds a small schema sample from `X_train`.
- Ensures numeric features are float.
- Infers ONNX types via `guess_data_type`, then patches Int32/64 → FloatTensorType.
- Exports the sklearn Pipeline to `models/sales_regressor.onnx`.

These two pieces are what you mention when someone asks:

> “How do you interpret your model?”  
> “How would you deploy it on a non‑Python stack?”

### Step 6 – CLI for Predictions (`src.predict_sales.py`)

Usage:

```bash
# Show required features and an example payload
python -m src.predict_sales --show-schema

# Inline JSON
python -m src.predict_sales --json '{"QUANTITYORDERED": 30, "PRICEEACH": 95.7, "ORDERLINENUMBER": 3,
                                     "MSRP": 120.0, "QTR_ID": 3, "MONTH_ID": 7, "YEAR_ID": 2004,
                                     "PRODUCTLINE": "Classic Cars", "COUNTRY": "USA", "DEALSIZE": "Medium"}'

# JSON file
python -m src.predict_sales --json-file features.json
```

The script:

- loads the best pipeline and metadata,
- resolves feature order,
- builds a 1‑row `pandas.DataFrame`,
- prints predicted `SALES` and model info.

### Step 7 – FastAPI v2 Service (`src.api_sales_v2.py`)

Start the server:

```bash
uvicorn src.api_sales_v2:app --reload
```

Endpoints:

- `GET /health` – model status + metadata (best model name, MLflow run id, feature list).
- `POST /predict` – body is a `SalesFeatures` schema matching original columns (`QUANTITYORDERED`, `PRICEEACH`, `ORDERLINENUMBER`, `MSRP`, `QTR_ID`, `MONTH_ID`, `YEAR_ID`, `PRODUCTLINE`, `COUNTRY`, `DEALSIZE`).

Response includes:

- predicted `SALES`,
- currency (`USD`),
- `model_name`,
- `mlflow_run_id`,
- echo of the input features used.

All heavy lifting (feature ordering, preprocessing, encoding) is reused from the training pipeline.

---

## 📉 Time‑Series & Risk Module (SARIMA + GARCH)

Entry point: `src.train_sales_timeseries_risk.py`  
Core code: `src/sales_regression/timeseries.py`

Problem:  
Given the same sales data, **aggregate to monthly sales for one segment** and run:

- a proper **SARIMA** model with a naive baseline comparison,
- a **GARCH(1,1)** model on monthly returns to estimate volatility.

Example usage:

```bash
python -m src.train_sales_timeseries_risk   --product-line "Classic Cars"   --country "USA"   --forecast-steps 6   --garch-horizon 6
```

Pipeline:

1. Build monthly series (`build_monthly_sales_series`):
   - filter by `PRODUCTLINE` and `COUNTRY`,
   - construct a date index from `YEAR_ID` + `MONTH_ID`,
   - aggregate `SALES` by month,
   - ensure `freq="MS"` and fill missing months with `0.0`.

2. SARIMA (`fit_sarima_with_candidates`):
   - train/test split: reserve last `N` months for test,
   - compare several SARIMA configs,
   - compute RMSE, MAE, MAPE, sMAPE on the test window,
   - compare against a naive baseline (last train value),
   - fit best model on full history and forecast next `forecast_steps` months.

3. GARCH (`fit_garch_on_returns`):
   - compute monthly percent returns,
   - fit GARCH(1,1) on cleaned return series,
   - extract long‑run monthly volatility and annualised volatility,
   - forecast conditional variance for `garch_horizon` months ahead.

4. Save all results to:

   ```text
   output/timeseries/sales_ts_risk_<product>_<country>.json
   ```

Talking point for interviews:

> “On top of the tabular regression, I added a small time‑series layer that can answer questions like: ‘What’s the expected volatility of monthly sales for Classic Cars in the US over the next 6 months, and is my SARIMA model actually better than a naive baseline?’”

---

## 🧪 Tests (v1 & v2)

Basic tests (v1) under `tests/`:

- `test_feature_engineering.py` – validates that engineered features exist and have expected types.
- `test_regression_pipeline.py` – runs a small slice through the regression pipeline and checks that metrics are produced and positive.

For v2 you can extend tests to:

- assert that `data_parquet.py` produces parquet files with expected columns,
- ensure `build_feature_matrix` throws a clear error when columns are missing,
- validate that `train_sales_regression_mlflow` produces a non‑empty metrics JSON and a model file,
- spin up `api_sales_v2` in a test client and verify `/health` returns `status="ok"` after training.

---

## 🧠 How to Talk About This Project in an Interview

You can treat this section as your personal cheat‑sheet.

### 1. “What was the problem you solved?”

> “I took a single Kaggle sales CSV and turned it into a mini analytics & ML platform:  
> cleaning + EDA, supervised regression on `SALES`, basic drift detection, a Streamlit dashboard, a FastAPI prediction API, and an extended v2 architecture with Polars/Parquet/DuckDB, MLflow, SHAP explainability, ONNX export, and a time‑series risk module on top.”

### 2. “What tools and libraries did you use, and why?”

- **pandas / matplotlib / seaborn** – v1 EDA and quick visualizations.
- **Polars** – fast CSV reading and transformations, with better performance on medium data than plain pandas.
- **Parquet** – columnar storage for efficient analytics & ML (small, compressed files).
- **DuckDB** – in‑process analytical database; perfect match for Parquet + SQL aggregations without managing an external DB server.
- **scikit‑learn (Pipeline + ColumnTransformer)** – for clean ML pipelines and safe preprocessing.
- **XGBoost, LightGBM, CatBoost** – modern tree/boosting models that typically outperform vanilla RandomForest/GBM on tabular data.
- **MLflow** – experiment tracking; keeps metrics, params and artifacts organised and reproducible.
- **SHAP** – model explainability for tree‑based models.
- **skl2onnx / ONNX** – export path for serving the model in non‑Python environments.
- **FastAPI** – performant, typed HTTP API for online predictions.
- **Streamlit** – quick interactive dashboard with almost no frontend work.
- **arch, statsmodels** – SARIMA + GARCH for time‑series and volatility modeling.

### 3. “Why this stack instead of just a notebook?”

- I wanted a **portfolio piece that looks like real work**, not just a notebook:
  - config files instead of hard‑coded paths,
  - modular Python packages,
  - typed config loader,
  - experiment tracking with MLflow,
  - CLI tools and APIs for inference,
  - proper logging and error handling,
  - explainability (SHAP) and exportability (ONNX).

### 4. “What metrics did you track?”

- For the **tabular regression**:
  - `RMSE` – main optimisation metric for sales prediction.
  - `MAE` – robustness to outliers.
  - `R²` – variance explained.
- For **time‑series (SARIMA)**:
  - `RMSE`, `MAE` on the test window,
  - `MAPE` and `sMAPE` (with safe handling of zeros),
  - comparison vs naive baseline (`last train value`), including **relative RMSE improvement**.
- For **risk (GARCH)**:
  - mean & standard deviation of returns,
  - long‑run monthly volatility,
  - annualised volatility,
  - forecasted variance over the next `N` periods.

### 5. “What were the hardest issues and how did you solve them?”

Some examples you can mention:

1. **CSV encoding & date parsing**  
   - Problem: Kaggle/Excel exports often have non‑UTF8 characters and non‑ISO date strings.  
   - Solution:
     - Polars CSV loader with fallback to `encoding="utf8-lossy"`.
     - Two‑stage date parsing:
       - try auto inference / configured format,
       - fallback to known patterns (`"%m/%d/%Y %H:%M"`) with clear logs.

2. **Consistent feature schema between training, CLI and API**  
   - Problem: It’s easy to break inference by changing feature names or order.  
   - Solution:
     - Centralised schema in `sales_regression.schema` (numeric + categorical features).
     - Training pipeline and APIs both import the same schema.
     - Metadata JSON saves the effective feature list used by the best model.

3. **Avoiding target leakage in engineered features**  
   - Problem: Some naive engineered features can sneak the target (`SALES`) back into inputs.  
   - Solution:
     - v1 features are carefully defined (e.g. `LINE_TOTAL_APPROX = QUANTITYORDERED * PRICEEACH`) using only input fields, not the actual `SALES` target.

4. **ONNX export type mismatches**  
   - Problem: `skl2onnx` is picky about input dtypes; mixed int/float types can fail conversion.  
   - Solution:
     - Build a small schema sample, enforce numeric features as float,
     - map Int32/64 tensor types to FloatTensorType before conversion.

5. **Time‑series with zeros and short histories**  
   - Problem: Monthly series can be short and contain zero sales months → MAPE blows up, SARIMA/GARCH may fail.  
   - Solution:
     - enforce minimum number of monthly points,
     - compute sMAPE (symmetric) and skip points with near‑zero denominators,
     - wrap SARIMA and GARCH in try/except, return detailed error messages in JSON.

6. **Data drift & monitoring**  
   - Problem: Static model quality decays if the data distribution shifts.  
   - Solution:
     - v1 includes PSI computation as a lightweight example,
     - v2 adds time‑series volatility as a richer way to think about dynamics of sales.

### 6. “If you had more time, what would you add next?”

You can mention:

- A small **scheduling layer** (e.g. cron + CLI) to re‑run the full pipeline daily / weekly.
- **Model registry** (MLflow Model Registry or custom) instead of manually picking joblib files.
- Better **dashboard** (Streamlit) that surfaces:
  - MLflow metrics,
  - SHAP importances,
  - time‑series forecasts and risk indicators.
- Real **CI tests** (GitHub Actions) running linters + unit tests on pushes.

---

## 📜 License

```text
MIT License

Copyright (c) 2025 Mohammad Eslamnia
...
```
