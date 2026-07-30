import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

import torch
import torch.nn as nn
import torch.optim as optim

df = pd.read_csv("data/processed/feature_dataset.csv")

print("Dataset Shape:", df.shape)

print("\nColumns:")

print(df.columns.tolist())

X = df.drop(
    columns=[
        "datetime",
        "main_aqi",
        "aqi_label",
        "hazard_alert",
        "year"
    ]
)

y = df["main_aqi"]

split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# Convert NumPy arrays to PyTorch tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

# Convert target values to tensors
y_train = torch.FloatTensor(y_train.to_numpy(copy=True)).view(-1, 1)
y_test = torch.FloatTensor(y_test.to_numpy(copy=True)).view(-1, 1)


print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

#Create the Neural Network

class AQIPredictor(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(25, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1)

        )

    def forward(self, x):
        return self.network(x)

#Create the Model

model = AQIPredictor()
#Define Loss Function
criterion = nn.MSELoss()
#Define Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0005)
#Train the Model
epochs = 300

print("\nTraining PyTorch Model...\n")
best_loss = float("inf")
for epoch in range(epochs):
    
    predictions = model(X_train)

    loss = criterion(predictions, y_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if loss.item() < best_loss:
        best_loss = loss.item()
    torch.save(model.state_dict(), "models/pytorch_model.pth")

    if (epoch + 1) % 25 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.4f}")

#Make Predictions
model.eval()

with torch.no_grad():
    y_pred = model(X_test)
#Convert Predictions to NumPy
y_pred = y_pred.numpy()
y_true = y_test.numpy()

#Calculate Metrics

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

#Display Results
print("\nModel Performance")
print("-" * 30)

print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R²   : {r2:.4f}")

#Save the Model
import os

os.makedirs("models", exist_ok=True)


torch.save(model.state_dict(), "models/pytorch_model.pth")

joblib.dump(scaler, "models/pytorch_scaler.pkl")

print("\nPyTorch model saved successfully!")