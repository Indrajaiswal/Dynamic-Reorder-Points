import pandas as pd
import numpy as np


def clean_data(df):
    """
    Automatically clean uploaded inventory dataset.
    """

    df = df.copy()

    # -----------------------------
    # BEFORE CLEANING
    # -----------------------------

    before_rows = len(df)
    before_duplicates = df.duplicated().sum()
    before_missing = df.isna().sum().sum()

    # -----------------------------
    # STANDARDIZE COLUMN NAMES
    # -----------------------------

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    # -----------------------------
    # DATE CONVERSION
    # -----------------------------

    date_candidates = [
        "date",
        "transaction_date",
        "order_date",
        "sales_date",
        "sale_date"
    ]

    date_column = None

    for col in date_candidates:
        if col in df.columns:
            date_column = col
            break

    if date_column:
        df[date_column] = pd.to_datetime(
            df[date_column],
            errors="coerce"
        )

        # Remove rows with invalid dates
        df = df.dropna(subset=[date_column])

    # -----------------------------
    # REMOVE DUPLICATES
    # -----------------------------

    df = df.drop_duplicates()

    # -----------------------------
    # IDENTIFY NUMERIC COLUMNS
    # -----------------------------

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    # -----------------------------
    # HANDLE NUMERIC MISSING VALUES
    # -----------------------------

    for col in numeric_columns:

        if df[col].isna().sum() > 0:

            median_value = df[col].median()

            df[col] = df[col].fillna(
                median_value
            )

    # -----------------------------
    # HANDLE CATEGORICAL MISSING VALUES
    # -----------------------------

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    for col in categorical_columns:

        if df[col].isna().sum() > 0:

            mode = df[col].mode()

            if len(mode) > 0:

                df[col] = df[col].fillna(
                    mode[0]
                )

            else:

                df[col] = df[col].fillna(
                    "Unknown"
                )

    # -----------------------------
    # HANDLE NEGATIVE INVENTORY
    # -----------------------------

    stock_columns = [
        "current_stock",
        "stock",
        "inventory",
        "available_stock"
    ]

    for col in stock_columns:

        if col in df.columns:

            df.loc[
                df[col] < 0,
                col
            ] = 0

    # -----------------------------
    # HANDLE NEGATIVE DEMAND
    # -----------------------------

    demand_columns = [
        "units_sold",
        "quantity_sold",
        "qty_sold",
        "sales_quantity",
        "sales_qty",
        "quantity",
        "qty",
        "units",
        "demand",
        "sales"
    ]

    for col in demand_columns:

        if col in df.columns:

            df.loc[
                df[col] < 0,
                col
            ] = 0

    # -----------------------------
    # OUTLIER DETECTION
    # -----------------------------

    outlier_counts = {}

    for col in numeric_columns:

        # Skip binary columns
        unique_values = df[col].nunique()

        if unique_values <= 2:
            continue

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = (
            (df[col] < lower_bound) |
            (df[col] > upper_bound)
        )

        count = outliers.sum()

        outlier_counts[col] = int(count)

        # Winsorization
        df[col] = df[col].clip(
            lower=lower_bound,
            upper=upper_bound
        )

    # -----------------------------
    # FINAL MISSING VALUES
    # -----------------------------

    after_missing = df.isna().sum().sum()

    # -----------------------------
    # CLEANING REPORT
    # -----------------------------

    report = {

        "rows_before": before_rows,

        "rows_after": len(df),

        "duplicates_removed": int(
            before_duplicates
        ),

        "missing_values_before": int(
            before_missing
        ),

        "missing_values_after": int(
            after_missing
        ),

        "outliers_detected": outlier_counts
    }

    return df, report