from pathlib import Path

import pandas as pd
import pytest

from sales_forecasting_regression.data import SalesDatasetLoader


def test_sales_loader_builds_target_from_quantity_and_price(tmp_path: Path) -> None:
    csv_path = tmp_path / "sales.csv"
    frame = pd.DataFrame(
        {
            "QUANTITYORDERED": [2, 3],
            "PRICEEACH": [10.0, 20.0],
            "ORDERLINENUMBER": [1, 2],
            "MSRP": [11.0, 21.0],
            "QTR_ID": [1, 1],
            "MONTH_ID": [1, 2],
            "YEAR_ID": [2003, 2003],
            "PRODUCTLINE": ["Classic Cars", "Motorcycles"],
            "COUNTRY": ["USA", "USA"],
            "DEALSIZE": ["Small", "Medium"],
        }
    )
    frame.to_csv(csv_path, index=False)

    loaded = SalesDatasetLoader(random_state=1).load(csv_path=csv_path, use_synthetic_if_missing=False)
    assert "SALES" in loaded.columns
    assert loaded.loc[0, "SALES"] == pytest.approx(20.0)
    assert loaded.loc[1, "SALES"] == pytest.approx(60.0)


def test_sales_loader_raises_when_csv_missing_and_synthetic_disabled(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        SalesDatasetLoader().load(csv_path=tmp_path / "missing.csv", use_synthetic_if_missing=False)
