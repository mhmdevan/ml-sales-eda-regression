# tests/test_feature_engineering.py

from sales_eda import (
    PROJECT_ROOT,
    load_config,
    load_data,
    add_date_features,
    clean_missing_values,
    remove_outliers,
)
from sales_regression import create_regression_features


def test_create_regression_features_adds_expected_columns():
    config = load_config()
    data_path = PROJECT_ROOT / config["data_path"]
    numeric_cols = config["numeric_columns"]
    cleaning_cfg = config["cleaning"]
    iqr_factor = float(config["outliers"]["iqr_factor"])

    df_raw = load_data(data_path).head(300)
    df_dates = add_date_features(df_raw)
    df_clean_missing = clean_missing_values(df_dates, cleaning_cfg=cleaning_cfg)
    df_clean = remove_outliers(
        df_clean_missing,
        numeric_cols=numeric_cols,
        factor=iqr_factor,
    )

    df_features, numeric_features, categorical_features = create_regression_features(
        df_clean
    )

    for col in ["QUARTER", "SEASON", "PRICE_TO_MSRP_RATIO", "LINE_TOTAL_APPROX"]:
        assert col in df_features.columns

    assert "PRICE_TO_MSRP_RATIO" in numeric_features
    assert "LINE_TOTAL_APPROX" in numeric_features
    assert "SEASON" in categorical_features
