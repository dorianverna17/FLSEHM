import numpy as np
import tensorflow as tf

from tensorflow import keras
from keras.callbacks import EarlyStopping

# Early stopping is disabled by default
early_stopping = 0

def construct_layer(layer):
	if layer['type'] == 'dense':
		return keras.layers.Dense(layer['neurons'],
						activation=layer['activation'],
						name=layer['name'])
	elif layer['type'] == 'dropout':
		return keras.layers.Dropout(layer['neurons'],
						activation=layer['activation'],
						name=layer['name'])
	elif layer['type'] == 'batch_normalization':
		return keras.layers.BatchNormalization(layer['neurons'],
						activation=layer['activation'],
						name=layer['name'])
	return None


def construct_optimizer(optimizer, learning_rate):
	if optimizer == 'adam':
		return keras.optimizers.Adam(learning_rate=learning_rate)
	elif optimizer == 'SGD':
		return keras.optimizers.SGD(learning_rate=learning_rate)
	elif optimizer == 'RMSprop':
		return keras.optimizers.RMSprop(learning_rate=learning_rate)
	elif optimizer == 'Adagrad':
		return keras.optimizers.Adagrad(learning_rate=learning_rate)
	elif optimizer == 'Adadelta':
		return keras.optimizers.Adadelta(learning_rate=learning_rate)
	elif optimizer == 'Adamax':
		return keras.optimizers.Adamax(learning_rate=learning_rate)
	elif optimizer == 'Nadam':
		return keras.optimizers.Nadam(learning_rate=learning_rate)
	return None


class EnhancedModel:
	def __init__(self, config):
		self.model = self.build_model(config)
		self.name = "Enhanced Model"
	
	def build_model(self, config):
		inputs = keras.Input(shape=(config.shape,))

		# define whether we can should batch normalization
		global early_stopping
		early_stopping = config.early_stopping

		# define first layer
		x = construct_layer(config.layers[0])(inputs)

		# loop through all the other hidden layers
		for i in range(1, len(config.layers) - 1):
			x = keras.layers.Dense(config.layers[i]['neurons'],
						 activation=config.layers[i]['activation'],
						 name=config.layers[i]['name'])(x)
		
		# attach output layer
		index_out = len(config.layers) - 1
		outputs = keras.layers.Dense(config.layers[index_out]['neurons'],
						 activation=config.layers[index_out]['activation'],
						 name=config.layers[index_out]['name'])(x)

		model = keras.Model(inputs=inputs, outputs=outputs)
		model.compile(
			optimizer=construct_optimizer(config.optimizer, config.learning_rate), loss='mse')
		
		return model

	def get_parameters(self):
		weights = self.model.get_weights()
		return weights
	
	def set_parameters(self, parameters):
		self.model.set_weights(parameters)
	
	def train(self, X, y_lat, y_lon, epochs=100, batch_size=32):
		y = np.column_stack((y_lat, y_lon))
		X = np.array(X)

		callbacks = []
		if early_stopping == 1:
			callbacks.append(EarlyStopping(monitor='loss', patience=10))

		self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=callbacks)
	
	def predict(self, X):
		X = np.array(X)
		predictions = self.model.predict(X)
		return predictions[:, 0], predictions[:, 1]
	
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
