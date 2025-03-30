from Models.linear_regression import LinearRegressionModel
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import importlib.util

import random
import numpy as np
import os
import logging

import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg, FedAdagrad
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset
from flwr.common import ndarrays_to_parameters, NDArrays, Scalar, Context

from utils import load_dataset_GNSS, get_closest_point

NUM_PARTITIONS = 10
BATCH_SIZE = 32

def split_data(x):
    random.shuffle(x)  # Shuffles in place, does not return anything
    split_idx = int(3 * len(x) / 4)  # Ensuring integer index
    return x[:split_idx], x[split_idx:]

# duplicated code, make sure to delete it at some point
def module_from_file(module_name, file_path):
	spec = importlib.util.spec_from_file_location(module_name, file_path)
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module

sim = module_from_file("bar", "Data_construction/simulate_GNSS_data_v2.py")

# Configure logging
logging.basicConfig(
	filename = f'Flower/FedAvg/output/app_client_{os.getpid()}.log',
	level=logging.INFO,  # Set the logging level
	format='%(asctime)s - %(levelname)s - %(message)s',  # Define the log format
)

class FlowerClient(NumPyClient):
	def __init__(self, partition_id, model, points):
		self.partition_id = partition_id
		self.model = model
		self.points = points
		self.proxy_position = sim.generate_random_point()

		# get the closest points used in training for the current round
		self.closest_points = get_closest_point(self.proxy_position, self.partition_id,
									 self.points[0])

		# split the closest points between train loader and valloader
		# use train test split random approach (80% - 20%)
		self.trainloader, self.valloader = split_data(self.closest_points)


	def get_parameters(self, config):
		logging.info(f"[Client {self.partition_id}] get_parameters")
		return self.model.get_parameters()

	def fit(self, parameters, config):
		logging.info(f"[Client {self.partition_id}] fit, config: {config}")
		
		# get the closest points used in training for the current round
		self.closest_points = get_closest_point(self.proxy_position, self.partition_id,
									 self.points[config["current_round"]])

		# split the closest points between train loader and valloader
		# use train test split random approach (80% - 20%)
		self.trainloader, self.valloader = split_data(self.closest_points)

		self.model.set_parameters(parameters)

		self.model.train([[elem.x, elem.y] for elem in self.trainloader[0]],
				[elem.x for elem in self.trainloader[1]],
				[elem.y for elem in self.trainloader[1]])

		return self.model.get_parameters(), len(self.trainloader), {}

	def evaluate(self, parameters, config):
		logging.info(f"[Client {self.partition_id}] evaluate, config: {config}")
		
		self.model.set_parameters(parameters)
		final_lat, final_lon = self.model.predict(
			[[elem.x, elem.y] for  elem in self.valloader[0]])

		loss, accuracy = self.model.compute_loss_and_accuracy(self.valloader[0], final_lat, final_lon)

		logging.info(f"[Client {self.partition_id}] model accuracy at round {config["current_round"]}: {accuracy}")

		return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def client_fn(context: Context) -> Client:
	model = LinearRegressionModel()

	# Read the node_config to fetch data partition associated to this node
	partition_id = context.node_config["partition-id"]
	num_partitions = context.node_config["num-partitions"]

	# Each client loads all the data generated
	points = load_dataset_GNSS()

	logging.info(f"[Client {partition_id}] loaded dataset")

	return FlowerClient(partition_id, model, points).to_client()
