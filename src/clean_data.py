import pandas as pd
import os

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data/training/concatenated_dataset.csv")

print("Original Shape:", df.shape)

# -----------------------------
# Convert datetime
# -----------------------------
df["datetime"] = pd.to_datetime(df["datetime"])

# -----------------------------
# Remove duplicate rows
# -----------------------------
duplicates = df.duplicated().sum()
print("Duplicate Rows:", duplicates)

df = df.drop_duplicates()

# -----------------------------
# Create Time Features
# -----------------------------
df["hour"] = df["datetime"].dt.hour
df["day"] = df["datetime"].dt.day
df["month"] = df["datetime"].dt.month
df["year"] = df["datetime"].dt.year
df["day_of_week"] = df["datetime"].dt.dayofweek

# Monday=0 ... Sunday=6
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

# -----------------------------
# Create processed folder
# -----------------------------
os.makedirs("data/processed", exist_ok=True)

# -----------------------------
# Save cleaned dataset
# -----------------------------
output_path = "data/processed/processed_dataset.csv"
df.to_csv(output_path, index=False)

print("\nProcessed Shape:", df.shape)
print(f"\nDataset saved to: {output_path}")

print("\nFirst 5 rows:")
print(df.head())