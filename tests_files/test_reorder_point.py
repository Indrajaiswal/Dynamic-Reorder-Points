import os

os.environ["LOKY_MAX_CPU_COUNT"] = "12"

import pandas as pd

from src.data_cleaner import clean_data
from src.feature_engineering import create_features
from src.model_training import train_models
from src.forecasting import generate_predictions
from src.reorder_point import calculate_reorder_point


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(
    "data/processed/cleaned_inventory.csv"
)


# ============================================================
# 2. CLEAN DATA
# ============================================================

cleaned_df, _ = clean_data(df)


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

featured_df = create_features(
    cleaned_df
)


# ============================================================
# 4. TRAIN MODELS
# ============================================================

training_result = train_models(
    featured_df
)


# ============================================================
# 5. GET BEST MODEL
# ============================================================

best_model = training_result[
    "best_model"
]

best_model_name = training_result[
    "best_model_name"
]

print(
    f"\nSelected Model: {best_model_name}"
)


# ============================================================
# 6. PREDICT DEMAND
# ============================================================

predicted_df = generate_predictions(
    best_model,
    featured_df
)


# ============================================================
# 7. CALCULATE DYNAMIC REORDER POINT
# ============================================================

result = calculate_reorder_point(
    predicted_df,
    service_level=0.95
)


# ============================================================
# 8. DISPLAY RESULTS
# ============================================================

columns_to_show = [
    "product_name",
    "current_stock",
    "predicted_demand",
    "lead_time_days",
    "dynamic_safety_stock",
    "dynamic_reorder_point",
    "reorder_status",
    "recommended_order_quantity"
]

print(
    "\n" + "=" * 100
)

print(
    "DYNAMIC REORDER POINT RESULTS"
)

print(
    "=" * 100
)

print(
    result[
        columns_to_show
    ].head(20).to_string(
        index=False
    )
)


# ============================================================
# 9. SUMMARY
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "INVENTORY SUMMARY"
)

print(
    "=" * 60
)

print(
    result[
        "reorder_status"
    ].value_counts()
)

print(
    "\nTotal units recommended "
    "for reorder:",
    int(
        result[
            "recommended_order_quantity"
        ].sum()
    )
)

print(
    "=" * 60
)