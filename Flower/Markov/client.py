import warnings
import logging
import numpy as np
import math
import os
import constants as ct
import differential_privacy as dp
import fcntl
import time

from constants import BASESTATIONS
from helpers import parse_generated_points, get_basestation
from helpers import create_matrix_with_points
from logging import INFO
from start_data_generation import output_dir
from shapely.geometry import Point

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from flwr.common import array_from_numpy
from flwr.common.logger import log
from flwr.client import ClientApp
from flwr.common import Context, Message, ParametersRecord, RecordSet
from flwr.client.mod import LocalDpMod

# Associate this proxy with a position on the map
# In the future, also consider this point to be mobile
proxy_position = None

# Configure logging
logging.basicConfig(
    filename = f'Flower/Markov/output/app_client_{os.getpid() % 4}.log',
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(levelname)s - %(message)s',  # Define the log format
)

fds = None  # Cache FederatedDataset

warnings.filterwarnings("ignore", category=UserWarning)

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
def get_closest_point() -> list[Point]:
	global counter_processed, proxy_position
	# List of devices positions that the proxy sees as being
	# around it. The proxy will use this positions to compute
	# a new Markov transition matrix
	close_points = []

	# Get the latest processed file
	latest_file = os.path.join(output_dir, f"generated_points_{counter_processed}.txt")

	logging.info("Client %s reads from file %s", app, latest_file)

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


def compute_new_matrix(points: list[list[Point]], matrix: np.ndarray) -> np.ndarray:
	# map points to basestations - don't use the IDs (PII data)
	mapped_points = []
	for i, p in enumerate(points):
		before = p[0]
		after = p[1]

		bs_before = get_basestation(centroids, before)
		bs_after = get_basestation(centroids, after)

		mapped_points.append([bs_before, bs_after])

	logging.info("Client %s gathered near points: %s", app, mapped_points)

	# compute new matrix based only on gathered points
	transition_matrix = create_matrix_with_points(mapped_points, BASESTATIONS)

	logging.info("Client %s created new transition matrix %s", app, transition_matrix)	

	return transition_matrix


# Create Client App
app = ClientApp()

@app.query()
def query(msg: Message, context: Context):
	global proxy_position
	# handle received messages
	record = msg.content.parameters_records
	if "proxy_positions" in record:
		proxy_position = record["proxy_positions"]["proxy_position"].numpy()
		logging.info("Client %s received proxy position: %s", app, str(proxy_position))
		return msg.create_reply(RecordSet())
	else:
		# get server transition matrix
		server_params = msg.content.parameters_records["markov_parameters"]
		server_matrix = server_params['markov_matrix'].numpy()

		logging.info("Client %s received transition matrix: %s", app, server_matrix)

		# generate new matrix based on observations
		# Instead of generating a random matrix, iterate over the current counter (counter_processed + 1)
		# entry of the generated_points list (import the start_data_generation package here).
		# See which devices from the list are the closest to it. Only process these entries then.
		close_points = get_closest_point()
		new_matrix = compute_new_matrix(close_points, server_matrix)
		
		# aggregate received matrix with new one
		aggregated_matrix = np.add(new_matrix, server_matrix)
		aggregated_matrix = np.divide(aggregated_matrix, 2)

		logging.info("Client %s aggregated matrix not noisy: %s", app, aggregated_matrix)

		# Apply local differential privacy
		noised_aggregated_matrix = dp.add_noise("local", "gaussian", aggregated_matrix)

		logging.info("Client %s built aggregated matrix: %s", app, noised_aggregated_matrix)

		matrix_to_send = aggregated_matrix

		# create response
		recordset = RecordSet()
		markov_matrix = array_from_numpy(np.array(matrix_to_send))
		parametes_records = ParametersRecord({'markov_matrix': markov_matrix})
		recordset.parameters_records["markov_parameters"] = parametes_records

		# Reply to the server
		return msg.create_reply(recordset)
