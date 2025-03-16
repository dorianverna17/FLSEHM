import numpy as np
from sklearn.linear_model import LinearRegression

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