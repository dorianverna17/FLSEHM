import warnings

import numpy as np
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner

from flwr.client import ClientApp
from flwr.common import Context, Message, MetricsRecord, RecordSet

fds = None  # Cache FederatedDataset

warnings.filterwarnings("ignore", category=UserWarning)

# Flower ClientApp
app = ClientApp()

@app.query()
def query(msg: Message, context: Context):
    metrics = {}
    
    metrics['client_log'] = 1

    # Reply to the server
    reply_content = RecordSet(metrics_records={"query_results": MetricsRecord(metrics)})

    return msg.create_reply(reply_content)
