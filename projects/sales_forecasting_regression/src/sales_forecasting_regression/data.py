from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sales_forecasting_regression.config import (
    SALES_CATEGORICAL_FEATURES,
    SALES_NUMERIC_FEATURES,
    SALES_TARGET,
    sales_schema,
)


class SalesDatasetLoader:
    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def load(
        self,
        csv_path: Path | None = None,
        use_synthetic_if_missing: bool = True,
        n_synthetic_rows: int = 2000,
    ) -> pd.DataFrame:
        resolved = csv_path if csv_path is not None else self.default_csv_path()
        if resolved.exists():
            frame = pd.read_csv(resolved)
            return self._prepare(frame)
        if not use_synthetic_if_missing:
            raise FileNotFoundError(f"Sales CSV not found: {resolved}")
        return self.synthetic_frame(n_rows=n_synthetic_rows)

    def default_csv_path(self) -> Path:
        repository_root = Path(__file__).resolve().parents[4]
        return repository_root / "data" / "raw" / "sales_data_sample.csv"

    def synthetic_frame(self, n_rows: int = 2000) -> pd.DataFrame:
        rng = np.random.default_rng(self.random_state)
        quantity = rng.integers(1, 80, size=n_rows)
        price = rng.uniform(20, 150, size=n_rows)
        order_line = rng.integers(1, 12, size=n_rows)
        msrp = price * rng.uniform(1.0, 1.5, size=n_rows)
        quarter = rng.integers(1, 5, size=n_rows)
        month = rng.integers(1, 13, size=n_rows)
        year = rng.choice([2003, 2004, 2005], size=n_rows)
        product_line = rng.choice(["Classic Cars", "Motorcycles", "Trucks and Buses"], size=n_rows)
        country = rng.choice(["USA", "France", "UK", "Germany"], size=n_rows)
        deal_size = rng.choice(["Small", "Medium", "Large"], size=n_rows)

        base_sales = quantity * price
        product_weight = np.where(product_line == "Classic Cars", 1.15, 1.0)
        deal_weight = np.where(deal_size == "Large", 1.1, np.where(deal_size == "Small", 0.9, 1.0))
        noise = rng.normal(0, 150, size=n_rows)
        sales = base_sales * product_weight * deal_weight + noise

        frame = pd.DataFrame(
            {
                "QUANTITYORDERED": quantity,
                "PRICEEACH": price,
                "ORDERLINENUMBER": order_line,
                "MSRP": msrp,
                "QTR_ID": quarter,
                "MONTH_ID": month,
                "YEAR_ID": year,
                "PRODUCTLINE": product_line,
                "COUNTRY": country,
                "DEALSIZE": deal_size,
                "SALES": sales,
            }
        )
        return self._prepare(frame)

    def _prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        prepared = frame.copy()

        if SALES_TARGET not in prepared.columns and {
            "QUANTITYORDERED",
            "PRICEEACH",
        }.issubset(prepared.columns):
            prepared[SALES_TARGET] = prepared["QUANTITYORDERED"] * prepared["PRICEEACH"]

        schema = sales_schema()
        schema.validate_frame(prepared)

        for feature in SALES_NUMERIC_FEATURES + [SALES_TARGET]:
            prepared[feature] = pd.to_numeric(prepared[feature], errors="coerce")
            prepared[feature] = prepared[feature].fillna(prepared[feature].median())

        for feature in SALES_CATEGORICAL_FEATURES:
            prepared[feature] = prepared[feature].astype(str).fillna("Unknown")

        prepared = prepared.dropna(subset=[SALES_TARGET]).reset_index(drop=True)
        return prepared
