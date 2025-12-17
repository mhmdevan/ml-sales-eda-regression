# tests/test_remove_outliers.py

import pandas as pd

from sales_eda import remove_outliers


def test_remove_outliers_reduces_row_count():
    # Tiny DataFrame with obvious outliers
    data = {
        "QUANTITYORDERED": [1, 2, 3, 1000],
        "PRICEEACH": [10.0, 20.0, 30.0, 40.0],
        "SALES": [10.0, 40.0, 90.0, 40000.0],
        "MSRP": [20.0, 30.0, 40.0, 50.0],
    }
    df = pd.DataFrame(data)

    # We explicitly pass which numeric columns to check
    cleaned = remove_outliers(
        df,
        numeric_cols=["QUANTITYORDERED", "SALES"],
        factor=1.5,
    )

    # Expect at least one row to be removed
    assert len(cleaned) < len(df)
    # In this simple toy example, only the extreme row should go
    assert len(cleaned) == 3
    # And max QUANTITYORDERED should be < 1000 now
    assert cleaned["QUANTITYORDERED"].max() < 1000
