from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Callable

import importlib.util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg, FedAdagrad
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset
from flwr.common import Metrics
from flwr.common import ndarrays_to_parameters, NDArrays, Scalar, Context

from torch.utils.data import DataLoader
from Models.custom_tutorial_training import DEVICE, NUM_PARTITIONS, get_parameters
from Models.custom_tutorial_training import Net
from client import client_fn

from constants import NO_CLIENTS

print(f"Training on {DEVICE}")
print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")

# Create an instance of the model and get the parameters
params = get_parameters(Net())

round = 0

# duplicated code, make sure to delete it at some point
def module_from_file(module_name, file_path):
	spec = importlib.util.spec_from_file_location(module_name, file_path)
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module

sim = module_from_file("bar", "Data_construction/simulate_GNSS_data_v2.py")

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
			"proxy_position": str(sim.generate_random_point())
		}
		return config

	return fit_config


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
	)

	# Configure the server for 3 rounds of training
	config = ServerConfig(num_rounds=3)
	return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
server = ServerApp(server_fn=server_fn)

# Specify the resources each of your clients need
# If set to none, by default, each client will be allocated 2x CPU and 0x GPUs
backend_config = {"client_resources": None}
if DEVICE.type == "cuda":
	backend_config = {"client_resources": {"num_gpus": 1}}

# Create the ClientApp
clientApp = ClientApp(client_fn=client_fn)

# Run simulation
run_simulation(
	server_app=server,
	client_app=clientApp,
	num_supernodes=NUM_PARTITIONS,
	backend_config=backend_config,
)
