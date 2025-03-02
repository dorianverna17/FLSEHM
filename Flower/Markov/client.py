import warnings
import logging
import numpy as np
import math
import os
import constants as ct
import differential_privacy as dp

from constants import BASESTATIONS
from helpers import parse_point
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
    filename='Flower/Markov/output/app_client.log',  # Specify the file name
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(levelname)s - %(message)s'  # Define the log format
)

fds = None  # Cache FederatedDataset

warnings.filterwarnings("ignore", category=UserWarning)

# counter of processed generated points
counter_processed = 0

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

	with open(latest_file, "r") as file:
		latest_points = [line.strip().split(', ') for line in file.readlines()]

	print("Points proxy position: " + str(proxy_position))

	# Iterate over the points
	for lp in latest_points:
		lp = parse_point(lp[0])
		x, y = lp.x, lp.y
		# compute euclidian distance
		distance = math.dist([x, y], [proxy_position[0], proxy_position[1]])
		# 0.1 degrees is ~11 km, this is the threshold by which we
		# can consider a point as being close to the proxy
		if distance < 0.1:
			close_points.append((x, y))

	counter_processed += 1
	return close_points


def compute_new_matrix(points: list[Point], matrix: np.ndarray) -> np.ndarray:
	print("Computing new matrix based on local proxy measurements")

	# TODO - add logic to compute new type of matrix

	# TODO - figure out which point changed basestations
	# in order to do that, we have to retain in the start_data_generation.py the last point where
	# the close device was before generating the new position

	return matrix


# Create Client App
app = ClientApp()

@app.query()
def query(msg: Message, context: Context):
	global proxy_position
	# handle received messages
	record = msg.content.parameters_records
	if "proxy_positions" in record:
		proxy_position = record["proxy_positions"]["proxy_position"].numpy()
		print("Points proxy position: " + str(proxy_position))
		return msg.create_reply(RecordSet())
	else:
		# get server transition matrix
		server_params = msg.content.parameters_records["markov_parameters"]
		server_matrix = server_params['markov_matrix'].numpy()

		print(server_matrix)

		logging.info("Client %s received transition matrix: %s", app, server_matrix)

		# generate new matrix based on observations
		# Instead of generating a random matrix, iterate over the current counter (counter_processed + 1)
		# entry of the generated_points list (import the start_data_generation package here).
		# See which devices from the list are the closest to it. Only process these entries then.
		close_points = get_closest_point()
		new_matrix = compute_new_matrix(close_points, server_matrix)

		# aggregate received matrix with new one
		aggregated_matrix = np.add(new_matrix, server_matrix)
		aggregated_matrix = np.divide(aggregated_matrix, BASESTATIONS)

		logging.info("Client %s aggregated matrix not noisy: %s", app, aggregated_matrix)

		# Apply local differential privacy
		noised_aggregated_matrix = dp.add_noise("local", "gaussian", aggregated_matrix)

		logging.info("Client %s built aggregated matrix: %s", app, noised_aggregated_matrix)

		# create response
		recordset = RecordSet()
		markov_matrix = array_from_numpy(np.array(noised_aggregated_matrix))
		parametes_records = ParametersRecord({'markov_matrix': markov_matrix})
		recordset.parameters_records["markov_parameters"] = parametes_records

		# Reply to the server
		return msg.create_reply(recordset)
