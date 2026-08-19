import os

os.environ["LOKY_MAX_CPU_COUNT"] = "12"

import pandas as pd

from src.data_cleaner import clean_data
from src.feature_engineering import create_features
from src.model_training import train_models

# Load dataset
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
result = train_models(
    featured_df
)

print("\nFinal Model Results:")
print(
    result["results"]
)

print(
    "\nSelected Best Model:",
    result["best_model_name"]
)