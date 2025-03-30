import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math

NUM_PARTITIONS = 10
BATCH_SIZE = 32

class NeuralNetworkModel(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNetworkModel, self).__init__()
        self.lat_model = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        self.lon_model = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, X):
        lat_pred = self.lat_model(X)
        lon_pred = self.lon_model(X)
        return lat_pred, lon_pred

def train_model(model, X, y_lat, y_lon, epochs=100, lr=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_lat_tensor = torch.tensor(y_lat, dtype=torch.float32).view(-1, 1)
        y_lon_tensor = torch.tensor(y_lon, dtype=torch.float32).view(-1, 1)

        lat_pred, lon_pred = model(X_tensor)
        loss_lat = criterion(lat_pred, y_lat_tensor)
        loss_lon = criterion(lon_pred, y_lon_tensor)

        loss = (loss_lat + loss_lon) / 2
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

def predict(model, X):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        predicted_lat, predicted_lon = model(X_tensor)
        return predicted_lat.numpy().flatten(), predicted_lon.numpy().flatten()

def compute_loss_and_accuracy(actual, predicted_lat, predicted_lon):
    actual_lat = [elem.x for elem in actual]
    actual_lon = [elem.y for elem in actual]

    rmse_lat = math.sqrt(np.mean((np.array(actual_lat) - predicted_lat) ** 2))
    rmse_lon = math.sqrt(np.mean((np.array(actual_lon) - predicted_lon) ** 2))

    loss = (rmse_lat + rmse_lon) / 2

    threshold = 0.08
    accuracy_lat = np.mean(np.abs(predicted_lat - actual_lat) < threshold) * 100
    accuracy_lon = np.mean(np.abs(predicted_lon - actual_lon) < threshold) * 100

    accuracy = (accuracy_lat + accuracy_lon) / 2
    return loss, accuracy
