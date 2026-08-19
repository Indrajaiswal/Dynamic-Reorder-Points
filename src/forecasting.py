import pandas as pd
import numpy as np


def generate_predictions(
    model,
    df,
    feature_columns=None
):
    """
    Generate demand predictions using the exact
    feature columns used during model training.

    This prevents ColumnTransformer errors caused by
    missing or mismatched columns during prediction.
    """

    data = df.copy()

    # ============================================================
    # VALIDATION
    # ============================================================

    if model is None:
        raise ValueError(
            "No trained model was provided."
        )

    if feature_columns is None:
        raise ValueError(
            "Feature columns from model training were not provided."
        )

    feature_columns = list(feature_columns)

    # ============================================================
    # CHECK MISSING FEATURES
    # ============================================================

    missing_features = [
        col
        for col in feature_columns
        if col not in data.columns
    ]

    # ------------------------------------------------------------
    # Create missing columns safely
    # ------------------------------------------------------------

    for column in missing_features:

        # Date-derived columns
        if column in [
            "date_year",
            "date_month",
            "date_day",
            "date_dayofweek",
            "date_week",
            "year",
            "month",
            "day",
            "day_of_week",
            "week_of_year",
            "is_weekend"
        ]:

            data[column] = 0

        # Numeric columns
        elif column in [
            "current_stock",
            "stock_available",
            "units_sold",
            "unit_price",
            "sales_revenue",
            "discount_percent",
            "promotion",
            "holiday",
            "lead_time_days",
            "safety_stock",
            "lag_1_day",
            "lag_7_days",
            "lag_14_days",
            "rolling_7_day",
            "rolling_30_day",
            "rolling_7_day_std",
            "discount_amount",
            "final_price"
        ]:

            data[column] = 0

        # Other columns are assumed categorical
        else:

            data[column] = "Unknown"

    # ============================================================
    # BUILD EXACT MODEL INPUT
    # ============================================================

    X = data[
        feature_columns
    ].copy()

    # ============================================================
    # DATETIME HANDLING
    # ============================================================

    for column in X.columns:

        if pd.api.types.is_datetime64_any_dtype(
            X[column]
        ):

            X[column] = X[column].astype(str)

    # ============================================================
    # NUMERIC CLEANING
    # ============================================================

    numeric_columns = X.select_dtypes(
        include=np.number
    ).columns

    for column in numeric_columns:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    # ============================================================
    # INFINITE VALUES
    # ============================================================

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # ============================================================
    # PREDICTION
    # ============================================================

    predictions = model.predict(X)

    predictions = np.asarray(
        predictions,
        dtype=float
    )

    predictions = np.maximum(
        predictions,
        0
    )

    # ============================================================
    # STORE PREDICTION
    # ============================================================

    data["predicted_demand"] = predictions

    # ============================================================
    # RETURN ALL ORIGINAL + ENGINEERED COLUMNS
    # ============================================================

    return data