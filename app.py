import streamlit as st
import pandas as pd
import numpy as np

from src.data_detector import detect_columns
from src.data_cleaner import clean_data
from src.feature_engineering import create_features
from src.model_training import train_models
from src.forecasting import generate_predictions
from src.reorder_point import calculate_reorder_point


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Dynamic Reorder Point System",
    page_icon="📦",
    layout="wide"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_numeric(series, default=0):
    """
    Safely convert a pandas Series to numeric.
    """
    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(default)
    )


def find_existing_column(df, candidates):
    """
    Find a column dynamically using case-insensitive matching.
    """
    column_map = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()

        if key in column_map:
            return column_map[key]

    return None


def add_standard_columns(df):
    """
    Create standardized columns where possible.

    This allows the application to work with different
    inventory datasets such as:

        Current_Stock
        stock_available
        available_stock
        inventory
        stock
    """

    data = df.copy()

    # --------------------------------------------------------
    # PRODUCT ID / SKU
    # --------------------------------------------------------

    if "sku" not in data.columns:

        source = find_existing_column(
            data,
            [
                "SKU",
                "sku_id",
                "product_id",
                "product_code",
                "item_id"
            ]
        )

        if source is not None:
            data["sku"] = data[source]

    # --------------------------------------------------------
    # PRODUCT NAME
    # --------------------------------------------------------

    if "product_name" not in data.columns:

        source = find_existing_column(
            data,
            [
                "Product_Name",
                "product",
                "item_name",
                "item",
                "medicine",
                "medicine_name"
            ]
        )

        if source is not None:
            data["product_name"] = data[source]

    # --------------------------------------------------------
    # BRAND
    # --------------------------------------------------------

    if "brand" not in data.columns:

        source = find_existing_column(
            data,
            [
                "Brand",
                "manufacturer",
                "company",
                "maker"
            ]
        )

        if source is not None:
            data["brand"] = data[source]

    # --------------------------------------------------------
    # CATEGORY
    # --------------------------------------------------------

    if "category" not in data.columns:

        source = find_existing_column(
            data,
            [
                "Category",
                "product_category",
                "item_category"
            ]
        )

        if source is not None:
            data["category"] = data[source]

    # --------------------------------------------------------
    # SUBCATEGORY
    # --------------------------------------------------------

    if "subcategory" not in data.columns:

        source = find_existing_column(
            data,
            [
                "Subcategory",
                "Sub_Category",
                "sub_category",
                "product_subcategory",
                "item_subcategory"
            ]
        )

        if source is not None:
            data["subcategory"] = data[source]

    # --------------------------------------------------------
    # STOCK
    # --------------------------------------------------------

    if "current_stock" not in data.columns:

        source = find_existing_column(
            data,
            [
                "Current_Stock",
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

        if source is not None:
            data["current_stock"] = safe_numeric(
                data[source]
            )

    # --------------------------------------------------------
    # LEAD TIME
    # --------------------------------------------------------

    if "lead_time_days" not in data.columns:

        source = find_existing_column(
            data,
            [
                "Lead_Time_Days",
                "lead_time",
                "supplier_lead_time",
                "delivery_days",
                "delivery_time"
            ]
        )

        if source is not None:
            data["lead_time_days"] = safe_numeric(
                data[source],
                default=7
            )

    # --------------------------------------------------------
    # DEMAND
    # --------------------------------------------------------

    if "units_sold" not in data.columns:

        source = find_existing_column(
            data,
            [
                "Units_Sold",
                "quantity_sold",
                "qty_sold",
                "sales_quantity",
                "quantity",
                "qty",
                "units",
                "demand",
                "sales"
            ]
        )

        if source is not None:
            data["units_sold"] = safe_numeric(
                data[source]
            )

    return data


def format_result_columns(df):
    """
    Build the final reorder table columns dynamically.

    Product identification columns are deliberately placed
    FIRST so users can clearly identify what needs ordering.
    """

    preferred_columns = [

        # ----------------------------------------------------
        # PRODUCT IDENTIFICATION
        # ----------------------------------------------------

        "date",
        "sku",
        "product_id",
        "product_name",
        "brand",
        "category",
        "subcategory",

        # ----------------------------------------------------
        # INVENTORY
        # ----------------------------------------------------

        "current_stock",
        "stock_available",

        # ----------------------------------------------------
        # DEMAND
        # ----------------------------------------------------

        "units_sold",
        "predicted_demand",

        # ----------------------------------------------------
        # REORDER
        # ----------------------------------------------------

        "lead_time_days",
        "dynamic_safety_stock",
        "dynamic_reorder_point",

        # ----------------------------------------------------
        # FINAL DECISION
        # ----------------------------------------------------

        "reorder_status",
        "recommended_order_quantity"
    ]

    result_columns = []

    for column in preferred_columns:

        if column in df.columns:

            if column not in result_columns:

                result_columns.append(column)

    return result_columns


# ============================================================
# TITLE
# ============================================================

st.title("📦 Dynamic Reorder Point System")

st.markdown(
    """
    **AI-powered inventory management system**

    Upload any raw inventory/sales CSV. The system automatically
    detects business columns, analyzes data quality, cleans data,
    engineers features, performs EDA, trains ML models, forecasts
    demand and calculates dynamic reorder points.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Settings")

service_level = st.sidebar.selectbox(
    "Service Level",
    options=[
        0.90,
        0.95,
        0.99
    ],
    index=1,
    format_func=lambda x: f"{int(x * 100)}%"
)

st.sidebar.info(
    """
    Higher service levels provide more safety stock.

    90% → Z = 1.28

    95% → Z = 1.65

    99% → Z = 2.33
    """
)


# ============================================================
# FILE UPLOAD
# ============================================================

st.header("📁 Upload Inventory Dataset")

uploaded_file = st.file_uploader(
    "Upload your raw CSV file",
    type=["csv"]
)


# ============================================================
# MAIN PIPELINE
# ============================================================

if uploaded_file is not None:

    try:

        # ====================================================
        # LOAD DATA
        # ====================================================

        df = pd.read_csv(uploaded_file)

        st.success(
            "Dataset uploaded successfully!"
        )

        # ====================================================
        # DATASET OVERVIEW
        # ====================================================

        st.header("📊 Dataset Overview")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Rows",
            f"{df.shape[0]:,}"
        )

        col2.metric(
            "Columns",
            df.shape[1]
        )

        col3.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )

        col4.metric(
            "Duplicate Rows",
            int(df.duplicated().sum())
        )

        # ====================================================
        # COLUMN NAMES
        # ====================================================

        st.subheader("📋 Column Names")

        for i, column in enumerate(df.columns):

            st.write(
                f"**{i}:** `{column}`"
            )

        # ====================================================
        # DATA SAMPLE
        # ====================================================

        with st.expander(
            "🔍 View Dataset Sample"
        ):

            st.dataframe(
                df.head(20),
                use_container_width=True,
                hide_index=True
            )

        # ====================================================
        # AUTOMATIC COLUMN DETECTION
        # ====================================================

        st.header(
            "🔎 Automatic Column Detection"
        )

        detected = detect_columns(df)

        detection_data = []

        for key, value in detected.items():

            if isinstance(value, dict):

                detected_column = value.get(
                    "column"
                )

                confidence = value.get(
                    "confidence"
                )

            else:

                detected_column = value
                confidence = None

            row = {
                "Role": key.replace(
                    "_",
                    " "
                ).title(),

                "Detected Column":
                    detected_column
                    if detected_column is not None
                    else "Not Detected"
            }

            if confidence is not None:

                row["Confidence"] = confidence

            detection_data.append(row)

        detection_df = pd.DataFrame(
            detection_data
        )

        st.dataframe(
            detection_df,
            use_container_width=True,
            hide_index=True
        )

        # ====================================================
        # DATA QUALITY
        # ====================================================

        st.header(
            "🧪 Data Quality Analysis"
        )

        missing_count = int(
            df.isnull().sum().sum()
        )

        duplicate_count = int(
            df.duplicated().sum()
        )

        quality_col1, quality_col2 = st.columns(2)

        quality_col1.metric(
            "Missing Values",
            missing_count
        )

        quality_col2.metric(
            "Duplicate Rows",
            duplicate_count
        )

        # ----------------------------------------------------
        # MISSING VALUES
        # ----------------------------------------------------

        st.subheader(
            "Missing Values by Column"
        )

        missing_df = (
            df.isnull()
            .sum()
            .reset_index()
        )

        missing_df.columns = [
            "Column",
            "Missing Values"
        ]

        missing_df = missing_df[
            missing_df["Missing Values"] > 0
        ]

        if len(missing_df) > 0:

            st.dataframe(
                missing_df,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.success(
                "No missing values detected."
            )

        # ----------------------------------------------------
        # DUPLICATES
        # ----------------------------------------------------

        if duplicate_count > 0:

            st.warning(
                f"{duplicate_count:,} duplicate rows detected."
            )

        else:

            st.success(
                "No duplicate rows detected."
            )

        # ====================================================
        # CATEGORY BALANCE
        # ====================================================

        category_column = find_existing_column(
            df,
            [
                "category",
                "Category",
                "product_category"
            ]
        )

        if category_column is not None:

            st.subheader(
                "⚖️ Data Balance Analysis"
            )

            st.write(
                f"Distribution of `{category_column}`"
            )

            category_counts = (
                df[category_column]
                .value_counts()
            )

            st.bar_chart(
                category_counts
            )

            if len(category_counts) > 1:

                ratio = (
                    category_counts.min()
                    /
                    category_counts.max()
                )

                if ratio >= 0.50:

                    st.success(
                        "Dataset distribution appears relatively balanced."
                    )

                else:

                    st.warning(
                        "Dataset has some category imbalance."
                    )

        # ====================================================
        # DATA CLEANING
        # ====================================================

        st.header(
            "🧹 Data Cleaning"
        )

        cleaned_df, cleaning_report = clean_data(
            df
        )

        st.success(
            f"Cleaning completed. "
            f"Final rows: {len(cleaned_df):,}"
        )

        with st.expander(
            "📄 View Cleaning Report"
        ):

            if isinstance(
                cleaning_report,
                dict
            ):

                st.json(
                    cleaning_report
                )

            else:

                st.write(
                    cleaning_report
                )

        # ====================================================
        # STANDARDIZE BUSINESS COLUMNS
        # ====================================================

        cleaned_df = add_standard_columns(
            cleaned_df
        )

        # ====================================================
        # FEATURE ENGINEERING
        # ====================================================

        st.header(
            "⚙️ Feature Engineering"
        )

        featured_df = create_features(
            cleaned_df
        )

        # ----------------------------------------------------
        # STANDARDIZE AGAIN AFTER FEATURE ENGINEERING
        # ----------------------------------------------------

        featured_df = add_standard_columns(
            featured_df
        )

        st.success(
            f"Feature engineering completed. "
            f"{featured_df.shape[1]} columns available."
        )

        with st.expander(
            "View Engineered Features"
        ):

            st.write(
                list(featured_df.columns)
            )

        # ====================================================
        # EDA
        # ====================================================

        st.header(
            "📊 Exploratory Data Analysis"
        )

        # ----------------------------------------------------
        # DETECT DEMAND
        # ----------------------------------------------------

        demand_column = find_existing_column(
            featured_df,
            [
                "units_sold",
                "quantity_sold",
                "qty_sold",
                "sales_quantity",
                "quantity",
                "qty",
                "units",
                "demand",
                "sales"
            ]
        )

        if demand_column is not None:

            demand_values = safe_numeric(
                featured_df[demand_column]
            )

            e1, e2, e3, e4 = st.columns(4)

            e1.metric(
                "Total Demand",
                f"{demand_values.sum():,.2f}"
            )

            e2.metric(
                "Average Demand",
                f"{demand_values.mean():,.2f}"
            )

            e3.metric(
                "Minimum Demand",
                f"{demand_values.min():,.2f}"
            )

            e4.metric(
                "Maximum Demand",
                f"{demand_values.max():,.2f}"
            )

            # ------------------------------------------------
            # DEMAND DISTRIBUTION
            # ------------------------------------------------

            st.subheader(
                "📈 Demand Distribution"
            )

            st.bar_chart(
                demand_values
                .value_counts()
                .sort_index()
                .head(50)
            )

            # ------------------------------------------------
            # DEMAND BY CATEGORY
            # ------------------------------------------------

            if "category" in featured_df.columns:

                st.subheader(
                    "📊 Demand by Category"
                )

                category_demand = (
                    featured_df
                    .assign(
                        _demand=demand_values
                    )
                    .groupby("category")[
                        "_demand"
                    ]
                    .sum()
                    .sort_values(
                        ascending=False
                    )
                )

                st.bar_chart(
                    category_demand
                )

        # ====================================================
        # MODEL TRAINING
        # ====================================================

        st.header(
            "🤖 Demand Forecasting Model"
        )

        with st.spinner(
            "Training demand forecasting models..."
        ):

            training_result = train_models(
                featured_df
            )

        if not isinstance(
            training_result,
            dict
        ):

            st.error(
                "Unexpected model training output."
            )

            st.stop()

        results = training_result.get(
            "results"
        )

        best_model = training_result.get(
            "best_model"
        )

        best_model_name = training_result.get(
            "best_model_name"
        )

        # ====================================================
        # MODEL COMPARISON
        # ====================================================

        st.subheader(
            "📊 Model Comparison"
        )

        if isinstance(
            results,
            pd.DataFrame
        ):

            st.dataframe(
                results,
                use_container_width=True,
                hide_index=True
            )

        if best_model_name:

            st.success(
                f"🏆 Best Model: {best_model_name}"
            )

        # ====================================================
        # DEMAND FORECASTING
        # ====================================================

        st.header(
            "📈 Demand Forecasting"
        )

        with st.spinner(
            "Predicting future demand..."
        ):

            predicted_df = generate_predictions(
                best_model,
                featured_df,
                feature_columns=training_result.get(
                    "feature_columns"
                )
            )

        st.success(
            "Demand prediction completed successfully!"
        )

        # ----------------------------------------------------
        # PREDICTION STATISTICS
        # ----------------------------------------------------

        if "predicted_demand" in predicted_df.columns:

            prediction_values = safe_numeric(
                predicted_df["predicted_demand"]
            )

            p1, p2, p3, p4 = st.columns(4)

            p1.metric(
                "Average Predicted Demand",
                f"{prediction_values.mean():,.2f}"
            )

            p2.metric(
                "Minimum Prediction",
                f"{prediction_values.min():,.2f}"
            )

            p3.metric(
                "Maximum Prediction",
                f"{prediction_values.max():,.2f}"
            )

            p4.metric(
                "Forecast Records",
                f"{len(predicted_df):,}"
            )

        # ----------------------------------------------------
        # PREDICTION TABLE
        # ----------------------------------------------------

        with st.expander(
            "🔍 View Demand Predictions"
        ):

            prediction_columns = [
                column
                for column in [
                    "date",
                    "sku",
                    "product_id",
                    "product_name",
                    "brand",
                    "category",
                    "subcategory",
                    "units_sold",
                    "predicted_demand"
                ]
                if column in predicted_df.columns
            ]

            st.dataframe(
                predicted_df[
                    prediction_columns
                ].head(100),
                use_container_width=True,
                hide_index=True
            )

        # ====================================================
        # DYNAMIC REORDER POINT
        # ====================================================

        st.header(
            "📐 Dynamic Reorder Point"
        )

        with st.spinner(
            "Calculating dynamic reorder points..."
        ):

            reorder_df = calculate_reorder_point(
                predicted_df,
                service_level=service_level
            )

        st.success(
            "Dynamic reorder point calculation completed!"
        )

        # ====================================================
        # ENSURE PRODUCT INFORMATION EXISTS
        # ====================================================

        reorder_df = add_standard_columns(
            reorder_df
        )

        # ====================================================
        # RESULT COLUMNS
        # ====================================================

        result_columns = format_result_columns(
            reorder_df
        )

        # ====================================================
        # DASHBOARD
        # ====================================================

        st.header(
            "📊 Inventory Dashboard"
        )

        # ----------------------------------------------------
        # PRODUCT COUNT
        # ----------------------------------------------------

        if "product_name" in reorder_df.columns:

            total_products = int(
                reorder_df[
                    "product_name"
                ]
                .nunique()
            )

        elif "sku" in reorder_df.columns:

            total_products = int(
                reorder_df[
                    "sku"
                ]
                .nunique()
            )

        elif "product_id" in reorder_df.columns:

            total_products = int(
                reorder_df[
                    "product_id"
                ]
                .nunique()
            )

        else:

            total_products = len(
                reorder_df
            )

        # ----------------------------------------------------
        # REORDER COUNT
        # ----------------------------------------------------

        if "reorder_status" in reorder_df.columns:

            reorder_count = int(
                (
                    reorder_df[
                        "reorder_status"
                    ]
                    == "REORDER REQUIRED"
                ).sum()
            )

            sufficient_count = int(
                (
                    reorder_df[
                        "reorder_status"
                    ]
                    == "SUFFICIENT STOCK"
                ).sum()
            )

        else:

            reorder_count = 0
            sufficient_count = 0

        # ----------------------------------------------------
        # TOTAL ORDER
        # ----------------------------------------------------

        if "recommended_order_quantity" in reorder_df.columns:

            total_order = int(
                safe_numeric(
                    reorder_df[
                        "recommended_order_quantity"
                    ]
                ).sum()
            )

        else:

            total_order = 0

        dashboard_col1, dashboard_col2, dashboard_col3, dashboard_col4 = st.columns(4)

        dashboard_col1.metric(
            "Products",
            f"{total_products:,}"
        )

        dashboard_col2.metric(
            "🚨 Reorder Required",
            f"{reorder_count:,}"
        )

        dashboard_col3.metric(
            "✅ Sufficient Stock",
            f"{sufficient_count:,}"
        )

        dashboard_col4.metric(
            "📦 Total Units to Order",
            f"{total_order:,}"
        )

        # ====================================================
        # REORDER ALERTS
        # ====================================================

        st.header(
            "🚨 Reorder Alerts"
        )

        if "reorder_status" in reorder_df.columns:

            reorder_items = reorder_df[
                reorder_df[
                    "reorder_status"
                ]
                == "REORDER REQUIRED"
            ].copy()

        else:

            reorder_items = pd.DataFrame()

        if len(reorder_items) > 0:

            st.warning(
                f"{len(reorder_items):,} "
                "inventory records require reordering."
            )

            # -----------------------------------------------
            # IMPORTANT:
            # SHOW PRODUCT + BRAND + CATEGORY + SUBCATEGORY
            # -----------------------------------------------

            alert_columns = format_result_columns(
                reorder_items
            )

            st.dataframe(
                reorder_items[
                    alert_columns
                ],
                use_container_width=True,
                hide_index=True
            )

        else:

            st.success(
                "🎉 No products currently require reordering."
            )

        # ====================================================
        # FULL INVENTORY STATUS
        # ====================================================

        st.header(
            "📋 Product Inventory Status"
        )

        st.dataframe(
            reorder_df[
                result_columns
            ],
            use_container_width=True,
            hide_index=True
        )

        # ====================================================
        # CATEGORY SUMMARY
        # ====================================================

        if "category" in reorder_df.columns:

            st.header(
                "📊 Category Summary"
            )

            aggregation = {}

            # ------------------------------------------------
            # PRODUCT COUNT
            # ------------------------------------------------

            if "product_name" in reorder_df.columns:

                aggregation[
                    "Products"
                ] = (
                    "product_name",
                    "nunique"
                )

            elif "sku" in reorder_df.columns:

                aggregation[
                    "Products"
                ] = (
                    "sku",
                    "nunique"
                )

            elif "product_id" in reorder_df.columns:

                aggregation[
                    "Products"
                ] = (
                    "product_id",
                    "nunique"
                )

            # ------------------------------------------------
            # STOCK
            # ------------------------------------------------

            if "current_stock" in reorder_df.columns:

                aggregation[
                    "Current_Stock"
                ] = (
                    "current_stock",
                    "sum"
                )

            elif "stock_available" in reorder_df.columns:

                aggregation[
                    "Current_Stock"
                ] = (
                    "stock_available",
                    "sum"
                )

            # ------------------------------------------------
            # DEMAND
            # ------------------------------------------------

            if "predicted_demand" in reorder_df.columns:

                aggregation[
                    "Predicted_Demand"
                ] = (
                    "predicted_demand",
                    "sum"
                )

            # ------------------------------------------------
            # REORDER REQUIRED
            # ------------------------------------------------

            if "reorder_status" in reorder_df.columns:

                aggregation[
                    "Reorder_Required"
                ] = (
                    "reorder_status",
                    lambda x:
                    (
                        x == "REORDER REQUIRED"
                    ).sum()
                )

            if aggregation:

                category_summary = (
                    reorder_df
                    .groupby("category")
                    .agg(**aggregation)
                    .reset_index()
                )

                st.dataframe(
                    category_summary,
                    use_container_width=True,
                    hide_index=True
                )

        # ====================================================
        # DOWNLOAD RESULTS
        # ====================================================

        st.header(
            "⬇️ Download Results"
        )

        csv_data = reorder_df.to_csv(
            index=False
        ).encode(
            "utf-8"
        )

        st.download_button(
            label="📥 Download Reorder Results",
            data=csv_data,
            file_name="dynamic_reorder_results.csv",
            mime="text/csv"
        )

    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        st.error(
            "❌ An error occurred while processing the dataset."
        )

        st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Dynamic Reorder Point System | "
    "ML-based Demand Forecasting & Inventory Management"
)