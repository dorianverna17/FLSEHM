import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math

NUM_PARTITIONS = 10
BATCH_SIZE = 32

class NeuralNetworkModel(nn.Module):
	def __init__(self):
		super(NeuralNetworkModel, self).__init__()
		self.lat_model = nn.Sequential(
			nn.Linear(2, 16),
			nn.ReLU(),
			nn.Linear(16, 1)
		)
		self.lon_model = nn.Sequential(
			nn.Linear(2, 16),
			nn.ReLU(),
			nn.Linear(16, 1)
		)
		self.name = "Neural Network Model"

	def get_parameters(self):
		"""Returns model parameters as a list of NumPy arrays."""
		params = []
		for param in self.parameters():
			params.append(param.detach().cpu().numpy())
		return params

	def set_parameters(self, parameters):
		"""Sets model parameters from a list of NumPy arrays."""
		with torch.no_grad():
			for param, new_value in zip(self.parameters(), parameters):
				param.copy_(torch.tensor(new_value, dtype=torch.float32))

	def forward(self, X):
		lat_pred = self.lat_model(X).view(-1, 1)
		lon_pred = self.lon_model(X).view(-1, 1)
		return lat_pred, lon_pred
	
	def train(self, X, y_lat, y_lon, epochs=100, lr=0.01):
		criterion_lat = nn.MSELoss()
		optimizer_lat = optim.Adam(self.parameters(), lr=lr)

		criterion_lon = nn.MSELoss()
		optimizer_lon = optim.Adam(self.parameters(), lr=lr)
		for epoch in range(epochs):
			self.lat_model.train()
			optimizer_lat.zero_grad()

			self.lon_model.train()
			optimizer_lon.zero_grad()

			X_tensor = torch.tensor(X, dtype=torch.float32).view(-1, 2)
			y_lat_tensor = torch.tensor(y_lat, dtype=torch.float32).view(-1, 1)
			y_lon_tensor = torch.tensor(y_lon, dtype=torch.float32).view(-1, 1)
			
			lat_pred, lon_pred = self.lat_model(X_tensor)
			lat_pred = lat_pred.view(-1, 1)
			lon_pred = lon_pred.view(-1, 1)

			loss_lat = criterion_lat(lat_pred, y_lat_tensor)
			loss_lon = criterion_lon(lon_pred, y_lon_tensor)
			
			loss = (loss_lat + loss_lon) / 2
			loss.backward()
			optimizer_lat.step()
			optimizer_lon.step()

	def predict(self, X):
		self.lat_model.eval()
		with torch.no_grad():
			X_tensor = torch.tensor(X, dtype=torch.float32)
			predicted_lat, predicted_lon = self.forward(X_tensor)
			return predicted_lat.numpy().flatten(), predicted_lon.numpy().flatten()

	def compute_loss_and_accuracy(self, actual, predicted_lat, predicted_lon):
		actual_lat = [elem.x for elem in actual]
		actual_lon = [elem.y for elem in actual]

		rmse_lat = math.sqrt(np.mean((np.array(actual_lat) - predicted_lat) ** 2))
		rmse_lon = math.sqrt(np.mean((np.array(actual_lon) - predicted_lon) ** 2))

		loss = (rmse_lat + rmse_lon) / 2

		# accuracy will be a mean between the accuracy for
		# latitude and accuracy for longitute
		threshold = 0.08 # (4 * 11 km / 5)
		accuracy_lat = 0
		for i in range(len(actual_lat)):
			if abs(predicted_lat[i] - actual_lat[i]) < threshold:
				accuracy_lat += 1
		accuracy_lat = (100 * accuracy_lat) / len(actual_lat)

		accuracy_lon = 0
		for i in range(len(actual_lon)):
			if abs(predicted_lon[i] - actual_lon[i]) < threshold:
				accuracy_lon += 1
		accuracy_lon = (100 * accuracy_lon) / len(actual_lon)

		accuracy = (accuracy_lat + accuracy_lon) / 2

		return loss, accuracy
