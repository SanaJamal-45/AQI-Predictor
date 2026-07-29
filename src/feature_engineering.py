import pandas as pd
import numpy as np

# -----------------------------
# Load processed dataset
# -----------------------------
df = pd.read_csv("data/processed/processed_dataset.csv")

print("Original Shape:", df.shape)

# -----------------------------
# Convert datetime
# -----------------------------
df["datetime"] = pd.to_datetime(df["datetime"])

# -----------------------------
# Cyclical Features
# -----------------------------
# Hour (0–23)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

# Month (1–12)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# -----------------------------
# AQI Labels
# -----------------------------
aqi_labels = {
    1: "Good",
    2: "Fair",
    3: "Moderate",
    4: "Poor",
    5: "Very Poor"
}

df["aqi_label"] = df["main_aqi"].map(aqi_labels)

# -----------------------------
# Hazard Alert
# -----------------------------
df["hazard_alert"] = (df["main_aqi"] >= 4).astype(int)

# -----------------------------
# Save Feature Dataset
# -----------------------------
output_path = "data/processed/feature_dataset.csv"
df.to_csv(output_path, index=False)

print("\nNew Shape:", df.shape)
print(f"\nFeature dataset saved to: {output_path}")

print("\nFirst 5 rows:")
print(df.head())