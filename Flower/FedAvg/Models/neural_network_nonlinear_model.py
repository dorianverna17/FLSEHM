import numpy as np
import tensorflow as tf
from tensorflow import keras

class NonlinearNeuralNetworkModel:
	def __init__(self):
		self.model = self.build_model()
		self.name = "Neural Network Non-linear Model"
	
	def build_model(self):
		inputs = keras.Input(shape=(2,))
		outputs = keras.layers.Dense(2, activation='relu', name="output_layer")(inputs)
		model = keras.Model(inputs=inputs, outputs=outputs)
		model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='mse')
		return model
	
	def get_parameters(self):
		weights = self.model.get_weights()
		return weights
	
	def set_parameters(self, parameters):
		self.model.set_weights(parameters)
	
	def train(self, X, y_lat, y_lon, epochs=100, batch_size=32):
		y = np.column_stack((y_lat, y_lon))
		X = np.array(X)
		self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
	
	def predict(self, X):
		X = np.array(X)
		predictions = self.model.predict(X)
		return predictions[:, 0], predictions[:, 1]
	
	def compute_loss_and_accuracy(self, actual, predicted_lat, predicted_lon):
		actual_lat = [elem.x for elem in actual]
		actual_lon = [elem.y for elem in actual]
		
		rmse_lat = np.sqrt(np.mean((np.array(actual_lat) - np.array(predicted_lat))**2))
		rmse_lon = np.sqrt(np.mean((np.array(actual_lon) - np.array(predicted_lon))**2))
		
		loss = (rmse_lat + rmse_lon) / 2
		
		threshold = 0.08  # (4 * 11 km / 5)
		accuracy_lat = np.mean(np.abs(np.array(predicted_lat) - np.array(actual_lat)) < threshold) * 100
		accuracy_lon = np.mean(np.abs(np.array(predicted_lon) - np.array(actual_lon)) < threshold) * 100
		
		accuracy = (accuracy_lat + accuracy_lon) / 2
		
		return loss, accuracy