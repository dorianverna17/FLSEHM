from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import os
import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from constants import BASESTATIONS, FILES_TO_GENERATE
from helpers import parse_generated_points, get_basestation
from helpers import create_matrix_with_points
from logging import INFO
from start_data_generation import output_dir
from shapely.geometry import Point

from torch.utils.data import DataLoader

import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg, FedAdagrad
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset
from flwr.common import ndarrays_to_parameters, NDArrays, Scalar, Context

DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
print(f"Training on {DEVICE}")
print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")

NUM_PARTITIONS = 10
BATCH_SIZE = 32

def load_datasets(partition_id: int, num_partitions: int):
	fds = FederatedDataset(dataset="cifar10", partitioners={"train": num_partitions})

	partition = fds.load_partition(partition_id)
	# Divide data on each node: 80% train, 20% test
	partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
	pytorch_transforms = transforms.Compose(
		[transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
	)

	def apply_transforms(batch):
		# Instead of passing transforms to CIFAR10(..., transform=transform)
		# we will use this function to dataset.with_transform(apply_transforms)
		# The transforms object is exactly the same
		batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
		return batch

	partition_train_test = partition_train_test.with_transform(apply_transforms)
	trainloader = DataLoader(
		partition_train_test["train"], batch_size=BATCH_SIZE, shuffle=True
	)
	valloader = DataLoader(partition_train_test["test"], batch_size=BATCH_SIZE)
	testset = fds.load_split("test").with_transform(apply_transforms)
	testloader = DataLoader(testset, batch_size=BATCH_SIZE)
	return trainloader, valloader, testloader


# counter of processed generated points
counter_processed = 0

# each client needs to know where the centroids are
# Read from file
centroids = []
with open("Data_Construction/centroids.log", "r", os.O_NONBLOCK) as f:
    for line in f:
        centroids.append(np.fromstring(line.strip()[1:-1], sep=' '))  # Convert string to NumPy array


# get the closest points to this proxy instance (at least 10 km for now).
# Some of the points might have been chosed by other proxies as well,
# which is not a problem.
def get_closest_point(proxy_position: Point, node_id: int, server_round: int) -> list[Point]:
	# List of devices positions that the proxy sees as being
	# around it. The proxy will use this positions to compute
	# a new Markov transition matrix
	close_points = []

	# Get the latest processed file
	latest_file = os.path.join(output_dir, f"generated_points_{server_round}.txt")

	logging.info("Client %d reads from file %s", node_id, latest_file)

	parsed_data = None
	with open(latest_file, "r") as file:
		lines = file.readlines()
		parsed_data = [parse_generated_points(l) for l in lines]
		
	# Iterate over the points
	for i, lp in enumerate(parsed_data):
		x, y = lp[0].x, lp[0].y
		# compute euclidian distance
		distance = math.dist([x, y], [proxy_position[0], proxy_position[1]])
		# 0.1 degrees is ~11 km, this is the threshold by which we
		# can consider a point as being close to the proxy
		if distance < 0.1:
			close_points.append((Point(x, y), lp[1]))
	
	counter_processed += 1
	return close_points


# This function returns the list of rounds, along with the points
# read from each one of them
def load_dataset_GNSS():
	points = []
	# get a list of generated files
	for file in range(FILES_TO_GENERATE):
		current_file = os.path.join(output_dir, f"generated_points_{file}.txt")

		parsed_data = None
		with open(file, "r") as file:
			lines = file.readlines()
			parsed_data = [parse_generated_points(l) for l in lines]

		points.append([file, parsed_data])

	return points
		
