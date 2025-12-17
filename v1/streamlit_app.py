"""
Simple Streamlit dashboard for the Kaggle sales dataset.

This reuses the EDA pipeline from sales_eda.py to:
  - load + clean data
  - show basic stats
  - provide a few interactive views
  - for running python -m streamlit run streamlit_app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from sales_eda import (
    PROJECT_ROOT,
    load_config,
    load_data,
    add_date_features,
    clean_missing_values,
    remove_outliers,
    get_descriptive_stats,
    get_grouped_stats,
)


@st.cache_data(show_spinner=True)
def load_clean_sales_data() -> pd.DataFrame:
    """
    Load and clean the dataset once, then cache it for the app.
    """
    config = load_config()
    data_path = PROJECT_ROOT / config["data_path"]
    numeric_cols = config["numeric_columns"]
    cleaning_cfg = config["cleaning"]
    iqr_factor = float(config["outliers"]["iqr_factor"])

    df_raw = load_data(data_path)
    df_dates = add_date_features(df_raw)
    df_clean_missing = clean_missing_values(df_dates, cleaning_cfg=cleaning_cfg)
    df_clean = remove_outliers(
        df_clean_missing, numeric_cols=numeric_cols, factor=iqr_factor
    )
    return df_clean


def main():
    st.set_page_config(page_title="Sales EDA Dashboard", layout="wide")
    st.title("📊 Kaggle Sales Data – EDA Dashboard")

    with st.sidebar:
        st.header("⚙️ Options")
        show_raw = st.checkbox("Show raw (cleaned) data", value=False)

    df_clean = load_clean_sales_data()

    # Sidebar filter by PRODUCTLINE
    productlines = ["(all)"]
    if "PRODUCTLINE" in df_clean.columns:
        productlines.extend(sorted(df_clean["PRODUCTLINE"].unique()))

    with st.sidebar:
        selected_productline = st.selectbox(
            "Filter by PRODUCTLINE (optional)",
            options=productlines,
            index=0,
        )

    if selected_productline != "(all)" and "PRODUCTLINE" in df_clean.columns:
        df_view = df_clean[df_clean["PRODUCTLINE"] == selected_productline].copy()
    else:
        df_view = df_clean

    st.write(f"Number of rows after filters: **{len(df_view)}**")

    if show_raw:
        st.subheader("Cleaned data preview")
        st.dataframe(df_view.head(200))

    # Descriptive statistics
    st.subheader("Descriptive statistics")
    numeric_cols = load_config()["numeric_columns"]
    desc_stats = get_descriptive_stats(df_view, numeric_cols=numeric_cols)
    st.dataframe(desc_stats)

    # Grouped statistics
    st.subheader("Grouped sales statistics")
    grouped = get_grouped_stats(df_view)
    tabs = st.tabs(["By PRODUCTLINE", "Top countries", "By YEAR"])

    with tabs[0]:
        df_prod = grouped.get("productline")
        if df_prod is not None:
            st.dataframe(df_prod)

    with tabs[1]:
        df_country = grouped.get("country")
        if df_country is not None:
            st.dataframe(df_country)

    with tabs[2]:
        df_year = grouped.get("year")
        if df_year is not None:
            st.dataframe(df_year)

    # Simple charts
    st.subheader("Sales distribution")

    if "SALES" in df_view.columns:
        st.bar_chart(df_view["SALES"])

    if "YEAR" in df_view.columns and "SALES" in df_view.columns:
        sales_by_year = df_view.groupby("YEAR")["SALES"].sum().sort_index()
        st.subheader("Total sales by YEAR")
        st.line_chart(sales_by_year)


if __name__ == "__main__":
    main()
