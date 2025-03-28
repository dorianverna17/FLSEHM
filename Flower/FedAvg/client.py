from Models.linear_regression import LinearRegressionModel
from utils import load_datasets

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
from flwr.common import ndarrays_to_parameters, NDArrays, Scalar, Context

from utils import load_dataset_GNSS

DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU

NUM_PARTITIONS = 10
BATCH_SIZE = 32

proxy_pos = None

class FlowerClient(NumPyClient):
	def __init__(self, partition_id, model, points):
		self.partition_id = partition_id
		self.model = model
		self.points = points

	def get_parameters(self, config):
		print(f"[Client {self.partition_id}] get_parameters")
		return get_parameters(self.net)

	def fit(self, parameters, config):
		print(f"[Client {self.partition_id}] fit, config: {config}")
		print("Round printed by client is " + str(config["current_round"]))
		print("This client has proxy position: " + str(config["proxy_position"]))

		# now the client has to figure out which are the points it consideres
		# based on the round, the list of points, and the proxy positions

		# TODO - modify this function
		return None

	def evaluate(self, parameters, config):
		print(f"[Client {self.partition_id}] evaluate, config: {config}")

		# TODO - modify this
		return 0, 0, None


def client_fn(context: Context) -> Client:
	model = LinearRegressionModel()

	# Read the node_config to fetch data partition associated to this node
	partition_id = context.node_config["partition-id"]
	num_partitions = context.node_config["num-partitions"]

	# Each client loads all the data generated
	points = load_dataset_GNSS()

	print("Client loaded dataset")

	return FlowerClient(partition_id, net, points).to_client()
