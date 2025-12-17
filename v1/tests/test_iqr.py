import pandas as pd

from sales_eda import detect_outliers_iqr


def test_detect_outliers_iqr_simple_case():
    values = [1, 2, 2, 4, 4, 4, 5, 6, 100]
    s = pd.Series(values)

    mask, bounds = detect_outliers_iqr(s, factor=1.5)

    # exactly one outlier
    assert mask.sum() == 1
    # that outlier is the extreme value (100)
    assert s[mask].tolist() == [100]

    # basic sanity on bounds
    assert bounds["q1"] < bounds["q3"]
    assert bounds["lower_bound"] < bounds["upper_bound"]
