import pandas as pd

from src.data_cleaner import clean_data
from src.feature_engineering import create_features
from src.eda import (
    generate_eda_report,
    print_eda_report,
    plot_demand_distribution,
    plot_demand_trend,
    plot_product_demand
)


df = pd.read_csv(
    "data/processed/cleaned_inventory.csv"
)

# Cleaning
cleaned_df, _ = clean_data(df)

# Feature engineering
featured_df = create_features(
    cleaned_df
)

# EDA
report = generate_eda_report(
    featured_df
)

print_eda_report(report)

# Visualizations
plot_demand_distribution(
    featured_df
)

plot_demand_trend(
    featured_df
)

plot_product_demand(
    featured_df
)