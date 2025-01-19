# Add explanation comment

import random
import time
import logging
import numpy as np

from constants import BASESTATIONS
from helpers import generate_random_markov_matrix
from logging import INFO

from flwr.common import Context, MessageType, RecordSet, Message
from flwr.common import ParametersRecord
from flwr.common.logger import log
from flwr.server import Driver, ServerApp
from flwr.common import array_from_numpy

# Configure logging
logging.basicConfig(
    filename='output/app_server.log',  # Specify the file name
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(levelname)s - %(message)s'  # Define the log format
)

app = ServerApp()

@app.main()
def main(driver: Driver, context: Context) -> None:
    # Configure the basic server settings:
    #  - number of nodes
    #  - training rounds
    #  - fraction to be sampled
    num_rounds = context.run_config["num-server-rounds"]
    min_nodes = 2
    fraction_sample = context.run_config["fraction-sample"]

    for server_round in range(num_rounds):
        log(INFO, "")
        log(INFO, "Starting round %s/%s", server_round + 1, num_rounds)

        # Loop and wait until enough nodes are available.
        all_node_ids = []
        while len(all_node_ids) < min_nodes:
            all_node_ids = driver.get_node_ids()
            if len(all_node_ids) >= min_nodes:
                # Sample nodes
                num_to_sample = int(len(all_node_ids) * fraction_sample)
                node_ids = random.sample(all_node_ids, num_to_sample)
                break
            log(INFO, "Waiting for nodes to connect...")
            time.sleep(2)

        log(INFO, "Sampled %s nodes (out of %s)", len(node_ids), len(all_node_ids))

        # Create messages
        recordset = RecordSet()
        messages = []

        # Create a random markov matrix to take care of coldstart problem
        markov_matrix = array_from_numpy(np.array(generate_random_markov_matrix()))
        parametes_records = ParametersRecord({'markov_matrix': markov_matrix})
        recordset.parameters_records["markov_parameters"] = parametes_records

        for node_id in node_ids:  # one message for each node
            message = driver.create_message(
                content=recordset,
                message_type=MessageType.QUERY,  # target `query` method in ClientApp
                dst_node_id=node_id,
                group_id=str(server_round),
            )
            messages.append(message)

        # Send messages and wait for all results
        replies = driver.send_and_receive(messages)
        logging.info("Received %s/%s results", len(replies), len(messages))

        # Aggregate markov matrices
        aggregated_matrix = aggregate_client_responses(replies)

        # Display aggregated markov matrix
        logging.info("Aggregated matrices: %s", aggregated_matrix)


# This function aims to aggregate the responses received from clients
# within the Federated framework
def aggregate_client_responses(messages: Message):
    log(INFO, "Aggregating partial responses from clients...")

    # aggregated markov matrix
    aggregated_matrix = [[0 for bs in range(BASESTATIONS)] for bs in range(BASESTATIONS)]

    for rep in messages:
        if rep.has_error():
            continue
        query_results = rep.content.parameters_records["markov_parameters"]

        if 'markov_matrix' not in query_results:
            continue
        
        response_matrix = query_results['markov_matrix'].numpy()

        logging.info("Client built aggregated matrix: %s", response_matrix)

        # sum up markov matrices
        aggregated_matrix = np.add(aggregated_matrix, response_matrix)

    # perform mean of the summed up aggregated matrix
    aggregated_matrix = np.divide(aggregated_matrix, len(messages))

    logging.info("Aggregated Markov matrix after round: %s", response_matrix)

    return aggregated_matrix
