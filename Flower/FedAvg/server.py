import flwr as fl
import pandas as pd
import numpy as np

from client import generate_client_resources
from client import PositionPredictionClient
from Models.linear_regression import LinearRegressionModel

if __name__ == "__main__":
	data = {
		'device_hash': ['device1', 'device2', 'device3', 'device4', 'device5'],
		'initial_lat': [37.7749, 34.0522, 40.7128, 51.5074, 48.8566],
		'initial_lon': [-122.4194, -118.2437, -74.0060, -0.1278, 2.3522],
		'final_lat': [37.7800, 34.0600, 40.7200, 51.5150, 48.8650],
		'final_lon': [-122.4100, -118.2300, -73.9900, -0.1200, 2.3600],
	}
	df = pd.DataFrame(data)

	client_resources = generate_client_resources(df)

	print(type(client_resources))
	print(client_resources)
	print()

	for key in client_resources:
		c = client_resources[key]()
		print(str(c.device_data))

	def fit_config(server_round: int):
		config = {
			"server_round": server_round,
		}
		return config
	
	strategy = fl.server.strategy.FedAvg(
		fraction_fit=0.8,
		min_fit_clients=4,
		min_available_clients=len(df['device_hash'].unique()),
		on_fit_config_fn=fit_config,
	)

	def get_client_fn(client_id):
		print(client_id.node_id)
		return client_resources[client_id.node_id % len(df['device_hash'].unique())]()

	history = fl.simulation.start_simulation(
		client_fn=get_client_fn,
		client_resources = {'num_cpus': 1, 'num_gpus': 0.0},
		num_clients=len(df['device_hash'].unique()),
		config=fl.server.ServerConfig(num_rounds=10),
		strategy=strategy,
	)

	# if history.losses_distributed:
	# 	global_model_parameters = history.losses_distributed[-1][1]
	# 	dummy_client = PositionPredictionClient(df.head(1))
	# 	dummy_client.set_parameters(global_model_parameters)
	# 	global_model = dummy_client.model

	# 	new_initial_positions = pd.DataFrame({
	# 		'initial_lat': [38.0, 35.0],
	# 		'initial_lon': [-122.0, -119.0],
	# 	})

	# 	predicted_changes_lat, predicted_changes_lon = global_model.predict(new_initial_positions)

	# 	predicted_final_lat = new_initial_positions['initial_lat'] + predicted_changes_lat
	# 	predicted_final_lon = new_initial_positions['initial_lon'] + predicted_changes_lon

	# 	print('Predicted final latitude: ', predicted_final_lat)
	# 	print('Predicted final longitude: ', predicted_final_lon)
