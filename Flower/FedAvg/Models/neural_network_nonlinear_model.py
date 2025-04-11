import numpy as np
import tensorflow as tf
from tensorflow import keras

def create_initial_parameters(model):
	"""Generate initial parameters for all layers of the model"""
	# Get the shapes of all weight tensors in the model
	weight_shapes = [w.shape for w in model.get_weights()]
	
	# Initialize all parameters (you would customize this logic)
	initial_weights = []
	for shape in weight_shapes:
		if len(shape) == 2:  # Dense layer weights
			initial_weights.append(np.random.normal(0, 0.05, shape))
		elif len(shape) == 1:  # Biases or BN parameters
			initial_weights.append(np.zeros(shape))
		else:
			initial_weights.append(np.ones(shape))
			
	return initial_weights

class NonlinearNeuralNetworkModel:
	def __init__(self):
		self.model = self.build_model()
		self.name = "Neural Network Non-linear Model"
	
	def build_model(self):
		inputs = keras.Input(shape=(2,))
		
		# Normalization layer
		x = keras.layers.Normalization()(inputs)
		
		# Hidden layers with batch normalization and dropout
		x = keras.layers.Dense(32)(x)
		x = keras.layers.BatchNormalization()(x)
		x = keras.layers.Activation('relu')(x)
		x = keras.layers.Dropout(0.2)(x)
		x = keras.layers.Dense(16)(x)
		x = keras.layers.BatchNormalization()(x)
		x = keras.layers.Activation('relu')(x)
		x = keras.layers.Dropout(0.2)(x)
		
		# Output layer
		outputs = keras.layers.Dense(2, name="output_layer")(x)
		
		model = keras.Model(inputs=inputs, outputs=outputs)
		
		optimizer = keras.optimizers.Adam(learning_rate=0.01)
		model.compile(optimizer=optimizer, loss='mse')
		
		return model
	
	def get_parameters(self):
		weights = self.model.get_weights()
		return weights
	
	def get_weights(self):
		# For a model, it collects weights from all its layers
		weights = []
		for layer in self.model.layers:
			if hasattr(layer, 'weights') and layer.weights:
				weights.extend(layer.weights)
		
		# Convert symbolic tensors to NumPy arrays
		return [weight.numpy() for weight in weights]

	def set_parameters(self, parameters):
		self.model.set_weights(parameters)
	
	def train(self, X, y_lat, y_lon, epochs=100, batch_size=32):
		y = np.column_stack((y_lat, y_lon))
		X = np.array(X)

		# Create callbacks
		early_stopping = keras.callbacks.EarlyStopping(
			monitor='loss', patience=10, restore_best_weights=True
		)
		lr_scheduler = keras.callbacks.ReduceLROnPlateau(
			monitor='loss', factor=0.5, patience=5, min_lr=0.0001
		)

		# Train the model
		history = self.model.fit(
			X, y, 
			epochs=epochs, 
			batch_size=batch_size, 
			verbose=0,
			callbacks=[early_stopping, lr_scheduler]
		)
	
		return history
	
	def predict(self, X):
		X = np.array(X)
		predictions = self.model.predict(X)
		return predictions[:, 0], predictions[:, 1], predictions
	
	def compute_loss_and_accuracy(self, actual, predicted_lat, predicted_lon):
		# Extract actual coordinates
		actual_lat = np.array([elem.x for elem in actual])
		actual_lon = np.array([elem.y for elem in actual])
		
		# Convert predictions to numpy arrays
		pred_lat = np.array(predicted_lat)
		pred_lon = np.array(predicted_lon)
		
		# Calculate Root Mean Square Error for each coordinate
		rmse_lat = np.sqrt(np.mean((actual_lat - pred_lat)**2))
		rmse_lon = np.sqrt(np.mean((actual_lon - pred_lon)**2))
		
		# Average loss across both coordinates
		loss = (rmse_lat + rmse_lon) / 2
		
		# Calculate distance errors (in coordinate units)
		distance_errors = np.sqrt((actual_lat - pred_lat)**2 + (actual_lon - pred_lon)**2)
		
		threshold = 0.1
		
		# Calculate accuracy as percentage of predictions within threshold
		accuracy = 100 * np.mean(distance_errors < threshold)
		
		return loss, accuracy
