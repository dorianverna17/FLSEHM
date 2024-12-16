import requests
import random
import threading
import time
from flask import Flask, request, jsonify

app = Flask(__name__)
central_node_url = 'http://central-node:5005'

local_aggregated_matrix = None


def generate_markov_matrix(size):
    """
    Generates a Markov matrix from the sample data.

    Args:
        data (list): The sample data to use for generating the Markov matrix.

    Returns:
        list: A Markov matrix as a list of lists.
    """
    matrix = []
    for i in range(size):
        row = [round(random.random(), 3) for _ in range(size)]
        row_sum = sum(row)
        normalized_row = [round(value / row_sum, 3) for value in row]  # Normalize to sum to 1
        matrix.append(normalized_row)

    print(f'Sent matrix: {matrix}')

    return matrix


def send_data_periodically(interval, size):
    """
    Send data to the central node at regular intervals.

    Args:
        interval (int): The interval in seconds at which data is sent.
        size (int): The size of the sample data list to generate.
    """
    while True:
        matrix = generate_markov_matrix(size)
        response = requests.post(f'{central_node_url}/aggregate', json={'matrices': [matrix]})
        print(f'Sent matrix: {matrix}, Response: {response.json()}')
        time.sleep(interval)


@app.route('/receive_aggregated', methods=['POST'])
def receive_aggregated():
    """
    Endpoint to receive aggregated matrix from the central node.

    Returns:
        Response: JSON response confirming receipt of the aggregated matrix.
    """
    global local_aggregated_matrix
    local_aggregated_matrix = request.json['aggregated_matrix']
    print(f'Received matrix: {local_aggregated_matrix}')
    return jsonify({'status': 'received', 'local_aggregated_matrix': local_aggregated_matrix})


@app.route('/get_aggregated', methods=['GET'])
def get_aggregated():
    """
    Endpoint to get the aggregated matrix from the central node.

    Returns:
        Response: JSON response containing the aggregated matrix from the central node.
    """
    response = requests.get(f'{central_node_url}/get_aggregated')
    return jsonify(response.json())


if __name__ == '__main__':
    """
    Start the background thread that sends data periodically and run the Flask app.
    """
    # Start the background thread that sends data periodically
    interval = 10  # Interval in seconds
    size = 2  # Size of the sample data
    threading.Thread(target=send_data_periodically, args=(interval, size), daemon=True).start()
    app.run(host='0.0.0.0', port=5000)
