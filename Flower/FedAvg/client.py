from Models.custom_tutorial_training import DEVICE, NUM_PARTITIONS, get_parameters
from Models.custom_tutorial_training import Net, set_parameters, test, train
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

# from server import round

DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU

NUM_PARTITIONS = 10
BATCH_SIZE = 32

class FlowerClient(NumPyClient):
	def __init__(self, partition_id, net, trainloader, valloader):
		self.partition_id = partition_id
		self.net = net
		self.trainloader = trainloader
		self.valloader = valloader

	def get_parameters(self, config):
		print(f"[Client {self.partition_id}] get_parameters")
		return get_parameters(self.net)

	def fit(self, parameters, config):
		print(f"[Client {self.partition_id}] fit, config: {config}")
		print("Round printed by client is " + str(config["current_round"]))
		set_parameters(self.net, parameters)
		train(self.net, self.trainloader, epochs=1)
		return get_parameters(self.net), len(self.trainloader), {}

	def evaluate(self, parameters, config):
		print(f"[Client {self.partition_id}] evaluate, config: {config}")
		set_parameters(self.net, parameters)
		loss, accuracy = test(self.net, self.valloader)
		return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def client_fn(context: Context) -> Client:
	net = Net().to(DEVICE)

	# Read the node_config to fetch data partition associated to this node
	partition_id = context.node_config["partition-id"]
	num_partitions = context.node_config["num-partitions"]
	
	trainloader, valloader, _ = load_datasets(partition_id, num_partitions)
	return FlowerClient(partition_id, net, trainloader, valloader).to_client()
