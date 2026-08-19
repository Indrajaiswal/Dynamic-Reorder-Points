import pandas as pd
import numpy as np


# ============================================================
# HELPER: FIND COLUMN
# ============================================================

def find_column(df, candidates):
    """
    Find a column dynamically using case-insensitive matching.
    """

    column_map = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for candidate in candidates:

        candidate_key = candidate.strip().lower()

        if candidate_key in column_map:
            return column_map[candidate_key]

    return None


# ============================================================
# MAIN FEATURE ENGINEERING FUNCTION
# ============================================================

def create_features(df):
    """
    Create time-series and business features.

    Designed to work with different inventory/sales datasets.

    Examples of supported columns:

    Date / date
    Product_ID / product_id
    SKU / sku
    Product_Name / product_name
    Brand / brand
    Category / category
    Subcategory / subcategory
    Units_Sold / units_sold
    Unit_Price / unit_price
    Current_Stock / stock_available
    Lead_Time_Days / lead_time_days
    Discount_Percent / discount_percent
    Promotion
    Holiday
    """

    data = df.copy()

    # ========================================================
    # NORMALIZE COLUMN NAMES
    # ========================================================

    rename_map = {}

    for column in data.columns:

        normalized = (
            str(column)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
        )

        rename_map[column] = normalized

    data = data.rename(
        columns=rename_map
    )

    # ========================================================
    # DATE
    # ========================================================

    date_column = find_column(
        data,
        [
            "date",
            "transaction_date",
            "order_date",
            "sales_date",
            "sale_date",
            "invoice_date",
            "timestamp",
            "datetime"
        ]
    )

    if date_column is not None:

        data[date_column] = pd.to_datetime(
            data[date_column],
            errors="coerce"
        )

        # Standardize to "date"
        if date_column != "date":

            data["date"] = data[
                date_column
            ]

        # ----------------------------------------------------
        # DATE FEATURES
        # ----------------------------------------------------

        data["year"] = (
            data["date"].dt.year
        )

        data["month"] = (
            data["date"].dt.month
        )

        data["day"] = (
            data["date"].dt.day
        )

        data["day_of_week"] = (
            data["date"].dt.dayofweek
        )

        data["week_of_year"] = (
            data["date"].dt.isocalendar()
            .week
            .astype(float)
        )

        data["is_weekend"] = (
            data["day_of_week"] >= 5
        ).astype(int)

    else:

        # If there is no date column,
        # create safe default values.

        data["year"] = 0
        data["month"] = 0
        data["day"] = 0
        data["day_of_week"] = 0
        data["week_of_year"] = 0
        data["is_weekend"] = 0

    # ========================================================
    # STANDARDIZE DEMAND
    # ========================================================

    demand_column = find_column(
        data,
        [
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
    )

    if demand_column is not None:

        data[demand_column] = pd.to_numeric(
            data[demand_column],
            errors="coerce"
        )

        data[demand_column] = (
            data[demand_column]
            .fillna(0)
        )

        if demand_column != "units_sold":

            data["units_sold"] = data[
                demand_column
            ]

    else:

        data["units_sold"] = 0

    # ========================================================
    # STANDARDIZE STOCK
    # ========================================================

    stock_column = find_column(
        data,
        [
            "current_stock",
            "stock_available",
            "available_stock",
            "stock",
            "inventory",
            "stock_level",
            "inventory_level",
            "on_hand",
            "on_hand_stock"
        ]
    )

    if stock_column is not None:

        data[stock_column] = pd.to_numeric(
            data[stock_column],
            errors="coerce"
        )

        data[stock_column] = (
            data[stock_column]
            .fillna(0)
        )

        if stock_column != "current_stock":

            data["current_stock"] = data[
                stock_column
            ]

    else:

        # Don't fail if stock isn't available.
        # Reorder point module can handle supported
        # stock columns if present.

        data["current_stock"] = 0

    # ========================================================
    # UNIT PRICE
    # ========================================================

    price_column = find_column(
        data,
        [
            "unit_price",
            "price",
            "selling_price",
            "sale_price",
            "mrp",
            "cost_price"
        ]
    )

    if price_column is not None:

        data[price_column] = pd.to_numeric(
            data[price_column],
            errors="coerce"
        )

        data[price_column] = (
            data[price_column]
            .fillna(
                data[price_column].median()
            )
        )

        if price_column != "unit_price":

            data["unit_price"] = data[
                price_column
            ]

    else:

        data["unit_price"] = 0

    # ========================================================
    # DISCOUNT
    # ========================================================

    discount_column = find_column(
        data,
        [
            "discount_percent",
            "discount_percentage",
            "discount"
        ]
    )

    if discount_column is not None:

        data[discount_column] = pd.to_numeric(
            data[discount_column],
            errors="coerce"
        ).fillna(0)

        if discount_column != "discount_percent":

            data["discount_percent"] = data[
                discount_column
            ]

    else:

        data["discount_percent"] = 0

    # ========================================================
    # PROMOTION
    # ========================================================

    promotion_column = find_column(
        data,
        [
            "promotion",
            "promo",
            "promotional",
            "is_promotion"
        ]
    )

    if promotion_column is not None:

        data[promotion_column] = (
            pd.to_numeric(
                data[promotion_column],
                errors="coerce"
            )
            .fillna(0)
        )

        if promotion_column != "promotion":

            data["promotion"] = data[
                promotion_column
            ]

    else:

        data["promotion"] = 0

    # ========================================================
    # HOLIDAY
    # ========================================================

    holiday_column = find_column(
        data,
        [
            "holiday",
            "is_holiday",
            "holiday_flag"
        ]
    )

    if holiday_column is not None:

        data[holiday_column] = (
            pd.to_numeric(
                data[holiday_column],
                errors="coerce"
            )
            .fillna(0)
        )

        if holiday_column != "holiday":

            data["holiday"] = data[
                holiday_column
            ]

    else:

        data["holiday"] = 0

    # ========================================================
    # LEAD TIME
    # ========================================================

    lead_time_column = find_column(
        data,
        [
            "lead_time_days",
            "lead_time",
            "supplier_lead_time",
            "delivery_days",
            "delivery_time"
        ]
    )

    if lead_time_column is not None:

        data[lead_time_column] = pd.to_numeric(
            data[lead_time_column],
            errors="coerce"
        )

        median_lead_time = (
            data[lead_time_column]
            .median()
        )

        if pd.isna(median_lead_time):
            median_lead_time = 7

        data[lead_time_column] = (
            data[lead_time_column]
            .fillna(median_lead_time)
        )

        if lead_time_column != "lead_time_days":

            data["lead_time_days"] = data[
                lead_time_column
            ]

    else:

        # Default lead time
        data["lead_time_days"] = 7

    # ========================================================
    # LAG FEATURES
    # ========================================================

    # Sort by product/SKU and date if available.

    group_column = find_column(
        data,
        [
            "sku",
            "product_id",
            "product_name",
            "product",
            "item_id"
        ]
    )

    if "date" in data.columns:

        if group_column is not None:

            data = data.sort_values(
                [group_column, "date"]
            ).reset_index(drop=True)

        else:

            data = data.sort_values(
                "date"
            ).reset_index(drop=True)

    # --------------------------------------------------------
    # LAG 1
    # --------------------------------------------------------

    if group_column is not None:

        data["lag_1_day"] = (
            data
            .groupby(group_column)["units_sold"]
            .shift(1)
        )

        data["lag_7_days"] = (
            data
            .groupby(group_column)["units_sold"]
            .shift(7)
        )

        data["lag_14_days"] = (
            data
            .groupby(group_column)["units_sold"]
            .shift(14)
        )

        # ----------------------------------------------------
        # ROLLING 7
        # ----------------------------------------------------

        data["rolling_7_day"] = (
            data
            .groupby(group_column)["units_sold"]
            .transform(
                lambda x:
                x.shift(1)
                .rolling(
                    window=7,
                    min_periods=1
                )
                .mean()
            )
        )

        # ----------------------------------------------------
        # ROLLING 30
        # ----------------------------------------------------

        data["rolling_30_day"] = (
            data
            .groupby(group_column)["units_sold"]
            .transform(
                lambda x:
                x.shift(1)
                .rolling(
                    window=30,
                    min_periods=1
                )
                .mean()
            )
        )

        # ----------------------------------------------------
        # ROLLING STD
        # ----------------------------------------------------

        data["rolling_7_day_std"] = (
            data
            .groupby(group_column)["units_sold"]
            .transform(
                lambda x:
                x.shift(1)
                .rolling(
                    window=7,
                    min_periods=2
                )
                .std()
            )
        )

    else:

        data["lag_1_day"] = (
            data["units_sold"].shift(1)
        )

        data["lag_7_days"] = (
            data["units_sold"].shift(7)
        )

        data["lag_14_days"] = (
            data["units_sold"].shift(14)
        )

        data["rolling_7_day"] = (
            data["units_sold"]
            .shift(1)
            .rolling(
                window=7,
                min_periods=1
            )
            .mean()
        )

        data["rolling_30_day"] = (
            data["units_sold"]
            .shift(1)
            .rolling(
                window=30,
                min_periods=1
            )
            .mean()
        )

        data["rolling_7_day_std"] = (
            data["units_sold"]
            .shift(1)
            .rolling(
                window=7,
                min_periods=2
            )
            .std()
        )

    # ========================================================
    # FILL LAG / ROLLING VALUES
    # ========================================================

    rolling_columns = [
        "lag_1_day",
        "lag_7_days",
        "lag_14_days",
        "rolling_7_day",
        "rolling_30_day",
        "rolling_7_day_std"
    ]

    for column in rolling_columns:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

        data[column] = (
            data[column]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
        )

        # Use current demand as fallback
        data[column] = (
            data[column]
            .fillna(
                data["units_sold"]
            )
        )

    # ========================================================
    # DISCOUNT AMOUNT
    # ========================================================

    data["discount_amount"] = (
        data["unit_price"]
        *
        data["discount_percent"]
        /
        100
    )

    # ========================================================
    # FINAL PRICE
    # ========================================================

    data["final_price"] = (
        data["unit_price"]
        -
        data["discount_amount"]
    )

    # ========================================================
    # NUMERIC CLEANUP
    # ========================================================

    numeric_columns = data.select_dtypes(
        include=np.number
    ).columns

    for column in numeric_columns:

        data[column] = (
            pd.to_numeric(
                data[column],
                errors="coerce"
            )
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
        )

    # ========================================================
    # FINAL FILL
    # ========================================================

    for column in data.columns:

        if pd.api.types.is_numeric_dtype(
            data[column]
        ):

            data[column] = (
                data[column]
                .fillna(0)
            )

    return data