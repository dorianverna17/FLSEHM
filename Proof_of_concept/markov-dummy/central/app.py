from flask import Flask, request, jsonify
import requests
import threading
import time
import numpy as np

app = Flask(__name__)

aggregated_matrix = None
worker_nodes = [
    'http://worker-node-0:5001',
    'http://worker-node-1:5002',
    'http://worker-node-2:5003',
    'http://worker-node-3:5004'
]

def aggregate_matrices(matrices):
    """
    Aggregates a list of matrices (example: sum them).

    Args:
        matrices (list): A list of matrices to aggregate.

    Returns:
        list: The aggregated matrix as a list of lists.
    """
    # Aggregate the matrices by summing them
    aggregated_matrix = np.sum(matrices, axis=0)
    
    # Normalize each row of the aggregated matrix
    row_sums = aggregated_matrix.sum(axis=1, keepdims=True)
    
    # Handle division by zero if any row sums are zero
    row_sums[row_sums == 0] = 1
    
    normalized_matrix = aggregated_matrix / row_sums
    
    return normalized_matrix.tolist()



def send_aggregated_periodically(interval):
    """
    Send the aggregated matrix to worker nodes at regular intervals.

    Args:
        interval (int): The interval in seconds at which the aggregated matrix is sent.
    """
    while True:
        if aggregated_matrix is not None:
            for worker_node in worker_nodes:
                try:
                    response = requests.post(f'{worker_node}/receive_aggregated', json={'aggregated_matrix': aggregated_matrix})
                    print(f"Sent to {worker_node}: {response.status_code}")
                except requests.exceptions.RequestException as e:
                    print(f"Failed to send to {worker_node}: {e}")
        time.sleep(interval)


@app.route('/aggregate', methods=['POST'])
def aggregate():
    """
    Endpoint to aggregate matrices received from worker nodes.

    Returns:
        Response: JSON response confirming the aggregation status and the aggregated matrix.
    """
    global aggregated_matrix
    matrices = request.json['matrices']
    if aggregated_matrix is None:
        aggregated_matrix = matrices[0]
    else:
        aggregated_matrix = aggregate_matrices([aggregated_matrix] + matrices)
    return jsonify({'status': 'aggregated', 'aggregated_matrix': aggregated_matrix})


@app.route('/get_aggregated', methods=['GET'])
def get_aggregated():
    """
    Endpoint to get the current aggregated matrix.

    Returns:
        Response: JSON response containing the aggregated matrix.
    """
    return jsonify({'aggregated_matrix': aggregated_matrix})


@app.route('/send_aggregated', methods=['POST'])
def send_aggregated():
    """
    Endpoint to send the current aggregated matrix to all worker nodes.

    Returns:
        Response: JSON response confirming the sending status.
    """
    global aggregated_matrix
    for worker_node in worker_nodes:
        try:
            response = requests.post(f'{worker_node}/receive_aggregated', json={'aggregated_matrix': aggregated_matrix})
            print(f"Sent to {worker_node}: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to send to {worker_node}: {e}")
    return jsonify({'status': 'sent'})


if __name__ == '__main__':
    """
    Start the background thread that sends aggregated data periodically and run the Flask app.
    """
    # Start the background thread that sends aggregated data periodically
    interval = 30  # Interval in seconds
    threading.Thread(target=send_aggregated_periodically, args=(interval,), daemon=True).start()
    app.run(host='0.0.0.0', port=5005)
