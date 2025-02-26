import warnings
import logging
import numpy as np
import constants as ct
import differential_privacy as dp

from constants import BASESTATIONS
from helpers import generate_random_markov_matrix
from logging import INFO

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from flwr.common import array_from_numpy
from flwr.common.logger import log
from flwr.client import ClientApp
from flwr.common import Context, Message, ParametersRecord, RecordSet
from flwr.client.mod import LocalDpMod

# TODO - associate this proxy with a position on the map
# In the future, also consider this point to be mobile
# One idea would be to make use of the function generate_random_point
# as we already have the allowlists put there
# we should retain the positions of these proxies and make sure that
# their positions are as diversified as they could
# Consider adding logic in the server to designate the proxies
# (the server may be the right entity to generate these locations
# for the proxies at first)

# Configure logging
logging.basicConfig(
    filename='Flower/Markov/output/app_client.log',  # Specify the file name
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(levelname)s - %(message)s'  # Define the log format
)

fds = None  # Cache FederatedDataset

warnings.filterwarnings("ignore", category=UserWarning)

# Create Client App
app = ClientApp()

@app.query()
def query(msg: Message, context: Context):
	# handle received messages
	record = msg.content.parameters_records
	if "proxy_positions" in record:
		print(record)
		return msg.create_reply(RecordSet())
	else:
		# get server transition matrix
		server_params = msg.content.parameters_records["markov_parameters"]
		server_matrix = server_params['markov_matrix'].numpy()

		logging.info("Client %s received transition matrix: %s", app, server_matrix)

		# generate new matrix based on observations
		# TODO - instead of generating a random matrix, iterate over the current counter (counter_processed + 1)
		# entry of the generated_points list (import the start_data_generation package here).
		# See which devices from the list are the closest to it. Only process these entries then.
		new_matrix = generate_random_markov_matrix()

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
