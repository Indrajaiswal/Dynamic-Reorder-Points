import pandas as pd
import numpy as np


# ============================================================
# COLUMN ALIASES
# ============================================================

COLUMN_ALIASES = {

    "date": [
        "date",
        "transaction_date",
        "order_date",
        "sales_date",
        "sale_date",
        "invoice_date",
        "timestamp",
        "datetime"
    ],

    "product_id": [
        "product_id",
        "productid",
        "item_id",
        "itemid",
        "sku",
        "sku_id",
        "code",
        "product_code"
    ],

    "product": [
        "product",
        "product_name",
        "item",
        "item_name",
        "medicine",
        "medicine_name"
    ],

    "brand": [
        "brand",
        "brand_name",
        "manufacturer"
    ],

    "category": [
        "category",
        "product_category",
        "item_category",
        "department",
        "type"
    ],

    "subcategory": [
        "subcategory",
        "sub_category",
        "product_subcategory",
        "item_subcategory",
        "sub_category_name"
    ],

    "demand": [
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
    ],

    "stock": [
        "current_stock",
        "stock",
        "inventory",
        "available_stock",
        "stock_available",
        "stock_level",
        "inventory_level",
        "on_hand",
        "on_hand_stock",
        "quantity_in_stock",
        "current_inventory"
    ],

    "lead_time": [
        "lead_time",
        "lead_time_days",
        "supplier_lead_time",
        "delivery_time",
        "delivery_days"
    ],

    "price": [
        "unit_price",
        "price",
        "selling_price",
        "sale_price",
        "mrp",
        "cost_price"
    ],

    "discount": [
        "discount",
        "discount_percent",
        "discount_percentage"
    ],

    "promotion": [
        "promotion",
        "promo",
        "promotional",
        "is_promotion"
    ],

    "holiday": [
        "holiday",
        "is_holiday",
        "holiday_flag"
    ]
}


# ============================================================
# NORMALIZE COLUMN
# ============================================================

def normalize_column_name(column):

    column = str(column).strip().lower()

    column = (
        column
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )

    return column


# ============================================================
# FIND COLUMN
# ============================================================

def find_column(df, column_type):

    normalized_columns = {
        normalize_column_name(col): col
        for col in df.columns
    }

    aliases = COLUMN_ALIASES[column_type]

    # Exact match
    for alias in aliases:

        if alias in normalized_columns:

            return (
                normalized_columns[alias],
                1.0
            )

    # Partial match
    for normalized_name, original_name in normalized_columns.items():

        for alias in aliases:

            if (
                alias in normalized_name
                or normalized_name in alias
            ):

                return (
                    original_name,
                    0.75
                )

    return None, 0.0


# ============================================================
# DATE DETECTION
# ============================================================

def detect_date_column(df):

    column, confidence = find_column(
        df,
        "date"
    )

    if column is not None:

        return {
            "column": column,
            "confidence": confidence
        }

    best_column = None
    best_score = 0

    for column in df.columns:

        try:

            converted = pd.to_datetime(
                df[column],
                errors="coerce"
            )

            valid_ratio = converted.notna().mean()

            if valid_ratio > best_score:

                best_score = valid_ratio
                best_column = column

        except Exception:

            continue

    if best_score >= 0.80:

        return {
            "column": best_column,
            "confidence": round(
                best_score,
                2
            )
        }

    return {
        "column": None,
        "confidence": 0
    }


# ============================================================
# DETECT ALL COLUMNS
# ============================================================

def detect_columns(df):

    detected = {}

    detected["date"] = detect_date_column(df)

    for column_type in [
        "product_id",
        "product",
        "brand",
        "category",
        "subcategory",
        "demand",
        "stock",
        "lead_time",
        "price",
        "discount",
        "promotion",
        "holiday"
    ]:

        column, confidence = find_column(
            df,
            column_type
        )

        detected[column_type] = {
            "column": column,
            "confidence": confidence
        }

    return detected


# ============================================================
# DATASET SUMMARY
# ============================================================

def dataset_summary(df):

    detected_columns = detect_columns(df)

    numeric_columns = (
        df
        .select_dtypes(
            include=np.number
        )
        .columns
        .tolist()
    )

    categorical_columns = (
        df
        .select_dtypes(
            include=["object", "category"]
        )
        .columns
        .tolist()
    )

    summary = {

        "rows": df.shape[0],

        "columns": df.shape[1],

        "column_names": df.columns.tolist(),

        "numeric_columns": numeric_columns,

        "categorical_columns": categorical_columns,

        "missing_values": int(
            df.isna()
            .sum()
            .sum()
        ),

        "duplicate_rows": int(
            df.duplicated()
            .sum()
        ),

        "detected_columns": detected_columns
    }

    return summary


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_dataset_summary(df):

    summary = dataset_summary(df)

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(
        f"Rows           : "
        f"{summary['rows']:,}"
    )

    print(
        f"Columns        : "
        f"{summary['columns']:,}"
    )

    print(
        f"Missing Values : "
        f"{summary['missing_values']:,}"
    )

    print(
        f"Duplicates     : "
        f"{summary['duplicate_rows']:,}"
    )

    print("\nColumn Names:")

    for column in summary["column_names"]:

        print(
            f"  - {column}"
        )

    print("\nDetected Columns:")

    for key, value in summary[
        "detected_columns"
    ].items():

        print(
            f"  {key:12} → "
            f"{value['column']} "
            f"(confidence: "
            f"{value['confidence']})"
        )

    print("=" * 60)

    return summary