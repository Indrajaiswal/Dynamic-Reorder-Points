import pandas as pd

from src.data_cleaner import clean_data
from src.feature_engineering import create_features


# Load raw/cleaned dataset
df = pd.read_csv(
    "data/processed/cleaned_inventory.csv"
)

# Clean
cleaned_df, cleaning_report = clean_data(df)

# Feature engineering
featured_df = create_features(
    cleaned_df
)


print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

print(
    "Original columns:",
    len(df.columns)
)

print(
    "New columns:",
    len(featured_df.columns)
)

print("\nColumns:")

print(featured_df.columns.tolist())

print("\nMissing values:")

print(
    featured_df.isna().sum()
)

print("\nSample data:")

print(
    featured_df.head()
)

print("=" * 60)