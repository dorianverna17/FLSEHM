from Models.linear_regression import NUM_PARTITIONS

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import os
import logging

from torch import tensor, float32

import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg, FedAdagrad
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset
from flwr.common import Metrics
from flwr.common import ndarrays_to_parameters, NDArrays, Scalar, Context

from client import client_fn

from constants import NO_CLIENTS

from Models.neural_network_nonlinear_model import NonlinearNeuralNetworkModel
from Models.enhanced_model import EnhancedModel
from Models.models_config.model_config import ModelConfig

# Create an instance of the model and get the parameters
model_to_use = os.environ['FD_MODEL']
model_config_file = os.environ['FD_MODEL_CONFIG']

params_linear_regression = [
	np.array([0.9, 0.1]),
	np.array([1.0]),
	np.array([0.2, 0.8]),
	np.array([-0.5]),
]

# Initialize model with given parameters
params_neural_network = [
    np.array([[0.9, 0.1], [0.2, 0.8]]),  # Weights for input
    np.array([1.0, -0.5])  # Biases
]

params_enhanced_neural_network = [
	np.random.uniform(-1, 1, (2, 8)),     # hidden_layer_1 weights
	np.zeros(8),                         # hidden_layer_1 biases
	np.random.uniform(-1, 1, (8, 8)),     # hidden_layer_2 weights
	np.zeros(8),                         # hidden_layer_2 biases
	np.random.uniform(-1, 1, (8, 2)),     # output_layer weights
	np.zeros(2)                          # output_layer biases
]

def build_params(config):
	params = []

	for i, p in enumerate(model_config.initial_parameters):
		weights = p['layer_weights']
		biases = p['layer_biases']
		params.append(np.random.uniform(weights[0], weights[1], (weights[2][0], weights[2][1])))
		params.append(np.zeros(biases))
	return params

params = None
if model_to_use == "linear_regression":
	params = params_linear_regression
elif model_to_use == "nn_model":
	params = params_neural_network
elif model_to_use == "nonlinear_nn_model":
	params = params_neural_network
elif model_to_use == "enhanced_model":
	model_config = ModelConfig(model_config_file)
	params = build_params(model_config)
else: # default to linear regression
	params = params_linear_regression

round = 0

# copied shamelessly from http://flower.ai/docs/framework/how-to-use-strategies.html
def get_on_fit_config_fn() -> Callable[[int], Dict[str, str]]:
	"""Return a function which returns training configurations."""

	def fit_config(server_round: int) -> Dict[str, str]:
		"""Return a configuration with static batch size and (local) epochs."""
		global proxy_positions

		config = {
			"learning_rate": str(0.001),
			"batch_size": str(32),
			"current_round": server_round,
		}
		return config

	return fit_config

def get_on_evaluate_config_fn() -> Callable[[int], Dict[str, str]]:
	"""Return a function which returns training configurations."""

	def evaluate_config(server_round: int) -> Dict[str, str]:
		"""Return a configuration with static batch size and (local) epochs."""
		global proxy_positions

		config = {
			"current_round": server_round,
		}
		return config

	return evaluate_config

def evaluate_metrics_aggregation_fn():
	def evaluate_metrics_aggregation(metrics):
		print("Server obtained the following metrics:" + str(metrics))		

		accuracies = [m[1]["accuracy"] for m in metrics]
		return {"accuracy": int(sum(accuracies)) / len(accuracies)}

	return evaluate_metrics_aggregation


def server_fn(context: Context) -> ServerAppComponents:
	# Create FedAvg strategy
	strategy = FedAvg(
		fraction_fit=0.3,
		fraction_evaluate=0.3,
		min_fit_clients=NO_CLIENTS,
		min_evaluate_clients=NO_CLIENTS,
		min_available_clients=NUM_PARTITIONS,
		initial_parameters=ndarrays_to_parameters(
			params
		),  # Pass initial model parameters
		on_fit_config_fn=get_on_fit_config_fn(),
		on_evaluate_config_fn=get_on_evaluate_config_fn(),
		evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn(),
	)

	# Configure the server for 7 rounds of training
	config = ServerConfig(num_rounds=7)
	return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
server = ServerApp(server_fn=server_fn)

# Specify the resources each of your clients need
# If set to none, by default, each client will be allocated 2x CPU and 0x GPUs
backend_config = {"client_resources": None}

# Create the ClientApp
clientApp = ClientApp(client_fn=client_fn)

# Run simulation
run_simulation(
	server_app=server,
	client_app=clientApp,
	num_supernodes=NUM_PARTITIONS,
	backend_config=backend_config,
)
