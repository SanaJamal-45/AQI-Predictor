import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
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
# Feature Scaling
# -----------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
# -----------------------------
# Train Model
# -----------------------------
model = Ridge(alpha=1.0)

print("\nTraining Ridge Regression...")
model.fit(X_train_scaled, y_train)

# -----------------------------
# Predictions
# -----------------------------
predictions = model.predict(X_test_scaled)

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

joblib.dump(model, "models/ridge_model.pkl")

joblib.dump(scaler, "models/ridge_scaler.pkl")

results = pd.DataFrame({
    "Model": ["Ridge Regression"],
    "RMSE": [rmse],
    "MAE": [mae],
    "R2": [r2]
})

results_file = "results/model_results.csv"

if os.path.exists(results_file):
    old_results = pd.read_csv(results_file)
    updated_results = pd.concat([old_results, results], ignore_index=True)
    updated_results.to_csv(results_file, index=False)
else:
    results.to_csv(results_file, index=False)
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
    "Coefficient": model.coef_
})

importance["Absolute"] = importance["Coefficient"].abs()

importance = importance.sort_values(
    by="Absolute",
    ascending=False
)

importance.to_csv(
    "results/ridge_coefficients.csv",
    index=False
)

print("\nTop 10 Coefficients")
print(importance.head(10))
