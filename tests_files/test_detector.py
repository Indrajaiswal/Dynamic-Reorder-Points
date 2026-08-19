import pandas as pd

from src.data_detector import print_dataset_summary


# Change this to your actual cleaned CSV path
file_path = "data/processed/cleaned_inventory.csv"


df = pd.read_csv(file_path)

print_dataset_summary(df)