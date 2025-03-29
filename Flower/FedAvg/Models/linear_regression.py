import numpy as np
import math
from sklearn.linear_model import LinearRegression

NUM_PARTITIONS = 10
BATCH_SIZE = 32

class LinearRegressionModel:
	def __init__(self):
		self.model_lat = LinearRegression()
		self.model_lon = LinearRegression()

	def get_parameters(self):
		return [
			self.model_lat.coef_, np.array([self.model_lat.intercept_]),
			self.model_lon.coef_, np.array([self.model_lon.intercept_]),
		]

	def set_parameters(self, parameters):
		self.model_lat.coef_ = parameters[0]
		self.model_lat.intercept_ = parameters[1][0]
		self.model_lon.coef_ = parameters[2]
		self.model_lon.intercept_ = parameters[3][0]


	def train(self, X, y_lat, y_lon):
		self.model_lat.fit(X, y_lat)
		self.model_lon.fit(X, y_lon)

	def predict(self, X):
		predicted_final_lat = self.model_lat.predict(X)
		predicted_final_lon = self.model_lon.predict(X)
		return predicted_final_lat, predicted_final_lon

	def compute_loss_and_accuracy(self, actual, predicted_lat, predicted_lon):
		actual_lat = [elem.x for elem in actual]
		actual_lon = [elem.y for elem in actual]

		# loss will be a mean between the RMSE for latitude
		# and RMSE for longitute
		sum_rmse_lat = 0
		for i in range(len(accuracy_lat)):
			sum_rmse_lat += (actual_lat[i] - predicted_lat[i])**2
		sum_rmse_lat /= len(actual_lat)
		rmse_lat = math.sqrt(sum_rmse_lat)

		sum_rmse_lon = 0
		for i in range(len(actual_lon)):
			sum_rmse_lon += (actual_lon[i] - predicted_lon[i])**2
		sum_rmse_lon /= len(actual_lon)
		rmse_lon = math.sqrt(sum_rmse_lon)

		loss = (rmse_lat + rmse_lon) / 2

		# accuracy will be a mean between the accuracy for
		# latitude and accuracy for longitute
		threshold = 0.02 # (11 km / 5)
		accuracy_lat = 0
		for i in range(len(accuracy_lat)):
			if math.abs(predicted_lat[i] - accuracy_lat[i]) < threshold:
				accuracy_lat += 1
		accuracy_lat = (100 * accuracy_lat) / len(actual_lat)

		accuracy_lon = 0
		for i in range(len(accuracy_lon)):
			if math.abs(predicted_lon[i] - accuracy_lon[i]) < threshold:
				accuracy_lon += 1
		accuracy_lon = (100 * accuracy_lon) / len(actual_lon)

		accuracy = (accuracy_lat + accuracy_lon) / 2

		return loss, accuracy