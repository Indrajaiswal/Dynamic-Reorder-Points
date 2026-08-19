import pandas as pd
import numpy as np


# ============================================================
# SERVICE LEVEL Z-SCORES
# ============================================================

SERVICE_LEVEL_Z = {
    0.90: 1.28,
    0.95: 1.65,
    0.99: 2.33
}


# ============================================================
# FIND COLUMN
# ============================================================

def find_column(df, candidates):

    column_map = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for candidate in candidates:

        candidate = candidate.lower()

        if candidate in column_map:

            return column_map[candidate]

    return None


# ============================================================
# DYNAMIC REORDER POINT
# ============================================================

def calculate_reorder_point(
    df,
    service_level=0.95
):

    data = df.copy()

    # ========================================================
    # SERVICE LEVEL
    # ========================================================

    closest = min(
        SERVICE_LEVEL_Z.keys(),
        key=lambda x:
        abs(x - service_level)
    )

    z_score = SERVICE_LEVEL_Z[
        closest
    ]

    # ========================================================
    # PRODUCT IDENTIFICATION
    # ========================================================

    product_id_column = find_column(
        data,
        [
            "product_id",
            "sku",
            "item_id",
            "product_code",
            "item_code"
        ]
    )

    product_name_column = find_column(
        data,
        [
            "product_name",
            "product",
            "item_name",
            "item",
            "medicine",
            "medicine_name"
        ]
    )

    brand_column = find_column(
        data,
        [
            "brand",
            "brand_name",
            "manufacturer",
            "company"
        ]
    )

    category_column = find_column(
        data,
        [
            "category",
            "product_category",
            "item_category",
            "department"
        ]
    )

    subcategory_column = find_column(
        data,
        [
            "subcategory",
            "sub_category",
            "product_subcategory",
            "item_subcategory",
            "subtype"
        ]
    )

    # ========================================================
    # STOCK
    # ========================================================

    stock_column = find_column(
        data,
        [
            "current_stock",
            "stock_available",
            "available_stock",
            "inventory",
            "stock",
            "stock_level",
            "inventory_level",
            "on_hand",
            "on_hand_stock",
            "quantity_in_stock",
            "current_inventory"
        ]
    )

    # ========================================================
    # DEMAND
    # ========================================================

    demand_column = find_column(
        data,
        [
            "predicted_demand",
            "forecast_demand",
            "forecast",
            "units_sold",
            "quantity_sold",
            "demand",
            "sales",
            "quantity"
        ]
    )

    # ========================================================
    # LEAD TIME
    # ========================================================

    lead_time_column = find_column(
        data,
        [
            "lead_time_days",
            "lead_time",
            "delivery_days",
            "delivery_time",
            "supplier_lead_time"
        ]
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    if stock_column is None:

        raise ValueError(
            "Current stock column not found. "
            "Supported examples: "
            "current_stock, stock_available, "
            "available_stock, inventory, stock."
        )

    if demand_column is None:

        raise ValueError(
            "Demand column not found."
        )

    # ========================================================
    # STANDARDIZE STOCK
    # ========================================================

    data["current_stock"] = pd.to_numeric(
        data[stock_column],
        errors="coerce"
    ).fillna(0)

    # ========================================================
    # STANDARDIZE DEMAND
    # ========================================================

    data["predicted_demand"] = pd.to_numeric(
        data[demand_column],
        errors="coerce"
    ).fillna(0)

    # ========================================================
    # LEAD TIME
    # ========================================================

    if lead_time_column is None:

        data["lead_time_days"] = 7

    else:

        data["lead_time_days"] = pd.to_numeric(
            data[lead_time_column],
            errors="coerce"
        )

        median_lead_time = (
            data["lead_time_days"]
            .median()
        )

        if pd.isna(
            median_lead_time
        ):

            median_lead_time = 7

        data["lead_time_days"] = (
            data["lead_time_days"]
            .fillna(
                median_lead_time
            )
            .clip(lower=1)
        )

    # ========================================================
    # DEMAND VARIABILITY
    # ========================================================

    if "rolling_7_day_std" in data.columns:

        demand_std = pd.to_numeric(
            data["rolling_7_day_std"],
            errors="coerce"
        )

    elif "units_sold" in data.columns:

        product_group = (
            product_id_column
            or product_name_column
        )

        if product_group is not None:

            demand_std = (
                data
                .groupby(
                    product_group
                )["units_sold"]
                .transform("std")
            )

        else:

            std_value = (
                pd.to_numeric(
                    data["units_sold"],
                    errors="coerce"
                )
                .std()
            )

            demand_std = pd.Series(
                std_value,
                index=data.index
            )

    else:

        demand_std = pd.Series(
            data["predicted_demand"].std(),
            index=data.index
        )

    demand_std = (
        pd.to_numeric(
            demand_std,
            errors="coerce"
        )
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )

    # ========================================================
    # SAFETY STOCK
    # ========================================================

    data["dynamic_safety_stock"] = np.ceil(
        z_score
        * demand_std
        * np.sqrt(
            data["lead_time_days"]
        )
    )

    # ========================================================
    # REORDER POINT
    # ========================================================

    data["dynamic_reorder_point"] = np.ceil(

        data["predicted_demand"]
        * data["lead_time_days"]

        + data["dynamic_safety_stock"]

    )

    # ========================================================
    # REORDER STATUS
    # ========================================================

    data["reorder_status"] = np.where(

        data["current_stock"]
        <= data["dynamic_reorder_point"],

        "REORDER REQUIRED",

        "SUFFICIENT STOCK"

    )

    # ========================================================
    # ORDER QUANTITY
    # ========================================================

    data["recommended_order_quantity"] = np.ceil(

        np.maximum(

            data["dynamic_reorder_point"]
            - data["current_stock"],

            0

        )

    )

    # ========================================================
    # RETURN
    # ========================================================

    return data