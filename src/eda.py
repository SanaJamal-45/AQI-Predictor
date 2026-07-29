import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data/processed/feature_dataset.csv")

# Create output folder
os.makedirs("reports/figures", exist_ok=True)

# Better looking plots
plt.style.use("ggplot")

# -----------------------------
# Dataset Information
# -----------------------------
print("=" * 60)
print("Dataset Shape")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

# -----------------------------
# AQI Distribution
# -----------------------------
plt.figure(figsize=(8,5))
sns.countplot(data=df, x="aqi_label", order=["Good","Fair","Moderate","Poor","Very Poor"])
plt.title("AQI Category Distribution")
plt.xlabel("AQI Category")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("reports/figures/aqi_distribution.png")
plt.close()

# -----------------------------
# Correlation Heatmap
# -----------------------------
plt.figure(figsize=(14,10))

numeric_df = df.select_dtypes(include=["number"])

sns.heatmap(
    numeric_df.corr(),
    cmap="coolwarm",
    center=0
)

plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("reports/figures/correlation_heatmap.png")
plt.close()

# -----------------------------
# AQI by Month
# -----------------------------
plt.figure(figsize=(8,5))

monthly_avg = df.groupby("month")["main_aqi"].mean().reset_index()

plt.figure(figsize=(8,5))

sns.barplot(
    data=monthly_avg,
    x="month",
    y="main_aqi"
)

plt.title("Average AQI by Month")
plt.xlabel("Month")
plt.ylabel("Average AQI")

plt.tight_layout()
plt.savefig("reports/figures/aqi_by_month.png")
plt.close()
# -----------------------------
# AQI by Hour
# -----------------------------
plt.figure(figsize=(10,5))

hourly_avg = df.groupby("hour")["main_aqi"].mean().reset_index()

plt.figure(figsize=(10,5))

sns.lineplot(
    data=hourly_avg,
    x="hour",
    y="main_aqi",
    marker="o"
)

plt.title("Average AQI by Hour")
plt.xlabel("Hour")
plt.ylabel("Average AQI")

plt.tight_layout()
plt.savefig("reports/figures/aqi_by_hour.png")
plt.close()

# -----------------------------
# PM2.5 vs AQI
# -----------------------------
plt.figure(figsize=(8,5))

sns.scatterplot(
    data=df,
    x="components_pm2_5",
    y="main_aqi",
    alpha=0.3
)

plt.title("PM2.5 vs AQI")
plt.tight_layout()
plt.savefig("reports/figures/pm25_vs_aqi.png")
plt.close()

# -----------------------------
# Temperature vs AQI
# -----------------------------
plt.figure(figsize=(8,5))

sns.scatterplot(
    data=df,
    x="temperature_2m",
    y="main_aqi",
    alpha=0.3
)

plt.title("Temperature vs AQI")
plt.tight_layout()
plt.savefig("reports/figures/temperature_vs_aqi.png")
plt.close()

# -----------------------------
# Humidity vs AQI
# -----------------------------
plt.figure(figsize=(8,5))

sns.scatterplot(
    data=df,
    x="relative_humidity_2m",
    y="main_aqi",
    alpha=0.3
)

plt.title("Humidity vs AQI")
plt.tight_layout()
plt.savefig("reports/figures/humidity_vs_aqi.png")
plt.close()

# -----------------------------
# Hazard Alert Distribution
# -----------------------------
plt.figure(figsize=(6,5))

sns.countplot(
    data=df,
    x="hazard_alert"
)

plt.title("Hazard Alert Distribution")
plt.tight_layout()
plt.savefig("reports/figures/hazard_alert_distribution.png")
plt.close()

print("\nEDA Complete!")
print("Graphs saved in reports/figures/")