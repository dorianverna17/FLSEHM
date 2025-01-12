# Add explanation comment

import random
import time
from logging import INFO

import numpy as np

from flwr.common import Context, MessageType, RecordSet, Message
from flwr.common.logger import log
from flwr.server import Driver, ServerApp

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
        log(INFO, "Received %s/%s results", len(replies), len(messages))


# This function aims to aggregate the responses received from clients
# within the Federated framework
def aggregate_client_responses(messages: Message):
    log(INFO, "Aggregating partial responses from clients...")

    pass

