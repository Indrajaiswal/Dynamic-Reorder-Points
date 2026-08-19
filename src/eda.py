import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def generate_eda_report(df):
    """
    Generate a general EDA report for an inventory dataset.
    Does not modify the original dataframe.
    """

    data = df.copy()

    report = {}

    # =========================================================
    # 1. BASIC DATASET INFORMATION
    # =========================================================

    report["rows"] = data.shape[0]
    report["columns"] = data.shape[1]

    report["column_names"] = data.columns.tolist()

    report["missing_values"] = (
        data.isna().sum()
    )

    report["duplicate_rows"] = (
        data.duplicated().sum()
    )

    # =========================================================
    # 2. NUMERIC SUMMARY
    # =========================================================

    report["numeric_summary"] = (
        data.select_dtypes(
            include=np.number
        ).describe()
    )

    # =========================================================
    # 3. CATEGORICAL SUMMARY
    # =========================================================

    categorical_columns = (
        data.select_dtypes(
            include=["object", "category"]
        ).columns
    )

    categorical_summary = {}

    for column in categorical_columns:

        categorical_summary[column] = (
            data[column]
            .value_counts()
            .head(10)
        )

    report["categorical_summary"] = (
        categorical_summary
    )

    # =========================================================
    # 4. DETECT IMPORTANT COLUMNS
    # =========================================================

    date_column = find_column(
        data,
        [
            "date",
            "transaction_date",
            "order_date",
            "sales_date",
            "sale_date"
        ]
    )

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

    stock_column = find_column(
        data,
        [
            "current_stock",
            "stock",
            "inventory",
            "available_stock",
            "stock_level",
            "inventory_level"
        ]
    )

    product_column = find_column(
        data,
        [
            "product_id",
            "product_name",
            "product",
            "item_id",
            "item_name",
            "item",
            "sku"
        ]
    )

    # =========================================================
    # 5. DEMAND ANALYSIS
    # =========================================================

    if demand_column:

        report["total_demand"] = (
            data[demand_column].sum()
        )

        report["average_demand"] = (
            data[demand_column].mean()
        )

        report["maximum_demand"] = (
            data[demand_column].max()
        )

        report["minimum_demand"] = (
            data[demand_column].min()
        )

    # =========================================================
    # 6. STOCK ANALYSIS
    # =========================================================

    if stock_column:

        report["average_stock"] = (
            data[stock_column].mean()
        )

        report["minimum_stock"] = (
            data[stock_column].min()
        )

        report["maximum_stock"] = (
            data[stock_column].max()
        )

    # =========================================================
    # 7. PRODUCT DEMAND
    # =========================================================

    if product_column and demand_column:

        product_demand = (
            data.groupby(product_column)[
                demand_column
            ]
            .sum()
            .sort_values(
                ascending=False
            )
        )

        report["product_demand"] = (
            product_demand
        )

    # =========================================================
    # 8. TIME SERIES DEMAND
    # =========================================================

    if date_column and demand_column:

        temp = data.copy()

        temp[date_column] = pd.to_datetime(
            temp[date_column],
            errors="coerce"
        )

        time_demand = (
            temp.groupby(date_column)[
                demand_column
            ]
            .sum()
            .sort_index()
        )

        report["time_demand"] = (
            time_demand
        )

    # =========================================================
    # 9. CATEGORY DEMAND
    # =========================================================

    category_column = find_column(
        data,
        [
            "category",
            "product_category",
            "item_category",
            "type"
        ]
    )

    if category_column and demand_column:

        category_demand = (
            data.groupby(category_column)[
                demand_column
            ]
            .sum()
            .sort_values(
                ascending=False
            )
        )

        report["category_demand"] = (
            category_demand
        )

    return report


# =============================================================
# COLUMN DETECTION HELPER
# =============================================================

def find_column(df, candidates):

    normalized = {
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_"): column
        for column in df.columns
    }

    for candidate in candidates:

        if candidate in normalized:

            return normalized[candidate]

    return None


# =============================================================
# PRINT EDA REPORT
# =============================================================

def print_eda_report(report):

    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    print(
        f"Rows       : {report['rows']:,}"
    )

    print(
        f"Columns    : {report['columns']:,}"
    )

    print(
        f"Duplicates : {report['duplicate_rows']:,}"
    )

    print("\n")

    if "total_demand" in report:

        print(
            f"Total Demand    : "
            f"{report['total_demand']:,.2f}"
        )

        print(
            f"Average Demand  : "
            f"{report['average_demand']:.2f}"
        )

        print(
            f"Minimum Demand  : "
            f"{report['minimum_demand']:.2f}"
        )

        print(
            f"Maximum Demand  : "
            f"{report['maximum_demand']:.2f}"
        )

    if "average_stock" in report:

        print(
            f"\nAverage Stock   : "
            f"{report['average_stock']:.2f}"
        )

        print(
            f"Minimum Stock   : "
            f"{report['minimum_stock']:.2f}"
        )

        print(
            f"Maximum Stock   : "
            f"{report['maximum_stock']:.2f}"
        )

    if "product_demand" in report:

        print("\nTop Products by Demand:")

        print(
            report["product_demand"]
            .head(10)
        )

    if "category_demand" in report:

        print("\nDemand by Category:")

        print(
            report["category_demand"]
        )

    print("=" * 60)


# =============================================================
# PLOT FUNCTIONS
# =============================================================

def plot_demand_distribution(df):

    demand_column = find_column(
        df,
        [
            "units_sold",
            "quantity_sold",
            "qty_sold",
            "quantity",
            "qty",
            "demand",
            "sales"
        ]
    )

    if not demand_column:
        return

    plt.figure(figsize=(10, 5))

    plt.hist(
        df[demand_column],
        bins=30
    )

    plt.title(
        "Demand Distribution"
    )

    plt.xlabel(
        demand_column
    )

    plt.ylabel(
        "Frequency"
    )

    plt.tight_layout()

    plt.show()


def plot_demand_trend(df):

    date_column = find_column(
        df,
        [
            "date",
            "transaction_date",
            "order_date",
            "sales_date",
            "sale_date"
        ]
    )

    demand_column = find_column(
        df,
        [
            "units_sold",
            "quantity_sold",
            "qty_sold",
            "quantity",
            "qty",
            "demand",
            "sales"
        ]
    )

    if not date_column or not demand_column:
        return

    temp = df.copy()

    temp[date_column] = pd.to_datetime(
        temp[date_column],
        errors="coerce"
    )

    daily_demand = (
        temp.groupby(date_column)[
            demand_column
        ]
        .sum()
        .sort_index()
    )

    plt.figure(figsize=(12, 5))

    plt.plot(
        daily_demand.index,
        daily_demand.values
    )

    plt.title(
        "Demand Trend Over Time"
    )

    plt.xlabel("Date")

    plt.ylabel("Demand")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.show()


def plot_product_demand(df):

    product_column = find_column(
        df,
        [
            "product_name",
            "product_id",
            "product",
            "item_name",
            "item",
            "sku"
        ]
    )

    demand_column = find_column(
        df,
        [
            "units_sold",
            "quantity_sold",
            "quantity",
            "qty",
            "demand",
            "sales"
        ]
    )

    if not product_column or not demand_column:
        return

    product_demand = (
        df.groupby(product_column)[
            demand_column
        ]
        .sum()
        .sort_values(
            ascending=False
        )
        .head(10)
    )

    plt.figure(figsize=(10, 5))

    product_demand.plot(
        kind="bar"
    )

    plt.title(
        "Top Products by Demand"
    )

    plt.xlabel(
        "Product"
    )

    plt.ylabel(
        "Total Demand"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    plt.show()