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


# Configure logging
logging.basicConfig(
    filename='output/app_client.log',  # Specify the file name
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(levelname)s - %(message)s'  # Define the log format
)

fds = None  # Cache FederatedDataset

warnings.filterwarnings("ignore", category=UserWarning)

# Create Client App
app = ClientApp()

@app.query()
def query(msg: Message, context: Context):
    # get server transition matrix
    server_params = msg.content.parameters_records["markov_parameters"]
    server_matrix = server_params['markov_matrix'].numpy()

    logging.info("Client received transition matrix %s", server_matrix)

    # generate new matrix based on observations
    new_matrix = generate_random_markov_matrix()

    # aggregate received matrix with new one
    aggregated_matrix = np.add(new_matrix, server_matrix)
    aggregated_matrix = np.divide(aggregated_matrix, BASESTATIONS)

    # Apply local differential privacy
    noised_aggregated_matrix = dp.add_noise("local", aggregated_matrix)

    logging.info("Client %s built aggregated matrix: %s", app, noised_aggregated_matrix)

    # create response
    recordset = RecordSet()
    markov_matrix = array_from_numpy(np.array(noised_aggregated_matrix))
    parametes_records = ParametersRecord({'markov_matrix': markov_matrix})
    recordset.parameters_records["markov_parameters"] = parametes_records

    # Reply to the server
    return msg.create_reply(recordset)
