"""
Small demo script to show how the IQR-based outlier detection works.

Run:

    python iqr_demo.py
"""

import pandas as pd

from sales_eda import detect_outliers_iqr


def main() -> None:
    # Simple toy example with one obvious outlier
    values = [1, 2, 2, 4, 4, 4, 5, 6, 100]
    s = pd.Series(values, name="example_values")

    mask, bounds = detect_outliers_iqr(s, factor=1.5)

    print("Input values:           ", values)
    print("Q1 (25th percentile):   ", bounds["q1"])
    print("Q3 (75th percentile):   ", bounds["q3"])
    print("IQR (Q3 - Q1):          ", bounds["iqr"])
    print("Lower bound (Q1-1.5*IQR):", bounds["lower_bound"])
    print("Upper bound (Q3+1.5*IQR):", bounds["upper_bound"])
    print("Outlier mask:           ", mask.tolist())
    print("Detected outliers:      ", s[mask].tolist())


if __name__ == "__main__":
    main()
