from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg, FedAdagrad
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset
from flwr.common import Metrics
from flwr.common import ndarrays_to_parameters, NDArrays, Scalar, Context

from Models.custom_tutorial_training import DEVICE, NUM_PARTITIONS, get_parameters
from Models.custom_tutorial_training import Net
from client import client_fn

print(f"Training on {DEVICE}")
print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")

# Create an instance of the model and get the parameters
params = get_parameters(Net())

round = 0

def fit_config(server_round: int):
	"""Generate training configuration for each round."""
	# Create the configuration dictionary
	config = {
		"current_round": server_round,
	}
	return config

class CustomStrategy(flwr.server.strategy.FedAvg):
	def aggregate_fit(
		self,
		server_round: int,
		results,
		failures,
	) -> Optional[Metrics]:
		global round
		
		# Aggregate results using default FedAvg
		aggregated_metrics = super().aggregate_fit(server_round, results, failures)

		super.on_fit_config_fn=fit_config

		return aggregated_metrics


def server_fn(context: Context) -> ServerAppComponents:
	# Create FedAvg strategy
	strategy = CustomStrategy(
		fraction_fit=0.3,
		fraction_evaluate=0.3,
		min_fit_clients=3,
		min_evaluate_clients=3,
		min_available_clients=NUM_PARTITIONS,
		initial_parameters=ndarrays_to_parameters(
			params
		),  # Pass initial model parameters
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
