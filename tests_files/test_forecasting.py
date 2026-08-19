import os

os.environ["LOKY_MAX_CPU_COUNT"] = "12"

import pandas as pd

from src.data_cleaner import clean_data
from src.feature_engineering import create_features
from src.model_training import train_models
from src.forecasting import generate_predictions


# Load data
df = pd.read_csv(
    "data/processed/cleaned_inventory.csv"
)


# Cleaning
cleaned_df, _ = clean_data(df)


# Feature engineering
featured_df = create_features(
    cleaned_df
)


# Train models
training_result = train_models(
    featured_df
)


# Get best model
best_model = training_result[
    "best_model"
]


# Generate predictions
predicted_df = generate_predictions(
    best_model,
    featured_df
)


print("\n" + "=" * 60)
print("DEMAND FORECASTING")
print("=" * 60)

print(
    predicted_df[
        [
            "product_name",
            "units_sold",
            "predicted_demand"
        ]
    ].head(20)
)

print("\nPrediction Statistics:")

print(
    predicted_df[
        "predicted_demand"
    ].describe()
)

print("=" * 60)