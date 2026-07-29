import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data/processed/feature_dataset.csv")

# -----------------------------
# Remove non-numeric columns
# -----------------------------
df = df.drop(columns=["datetime", "aqi_label"])

# -----------------------------
# Features & Target
# -----------------------------
X = df.drop(columns=["main_aqi", "hazard_alert","year"])

y = df["main_aqi"]

print("Number of Features:", X.shape[1])
print("Number of Samples:", X.shape[0])

# -----------------------------
# Train-Test Split
# -----------------------------
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

print("\nTraining Features:")
print(list(X.columns))
# -----------------------------
# Train Model
# -----------------------------
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest...")
model.fit(X_train, y_train)

# -----------------------------
# Predictions
# -----------------------------
predictions = model.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------
rmse = mean_squared_error(y_test, predictions) ** 0.5
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Performance")
print("-" * 30)
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R²   : {r2:.4f}")

# -----------------------------
# Save Model
# -----------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/random_forest_model.pkl")

results = pd.DataFrame({
    "Model": ["Random Forest"],
    "RMSE": [rmse],
    "MAE": [mae],
    "R2": [r2]
})

os.makedirs("results", exist_ok=True)

results.to_csv("results/model_results.csv", index=False)

print("\nResults saved!")
print(results)

print("\nModel saved successfully!")

# -----------------------------
# Feature Importance
# -----------------------------
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features")
print(importance.head(10))

importance.to_csv(
    "results/feature_importance.csv",
    index=False
)