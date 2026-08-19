import pandas as pd

from src.data_cleaner import clean_data


df = pd.read_csv(
    "data/processed/cleaned_inventory.csv"
)

cleaned_df, report = clean_data(df)


print("\n" + "=" * 60)
print("DATA CLEANING REPORT")
print("=" * 60)

print(
    "Rows before:",
    report["rows_before"]
)

print(
    "Rows after:",
    report["rows_after"]
)

print(
    "Duplicates removed:",
    report["duplicates_removed"]
)

print(
    "Missing values before:",
    report["missing_values_before"]
)

print(
    "Missing values after:",
    report["missing_values_after"]
)

print("\nOutliers detected:")

for column, count in report[
    "outliers_detected"
].items():

    print(
        f"{column}: {count}"
    )

print("=" * 60)