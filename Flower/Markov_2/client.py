import flwr as fl
import numpy as np

class MarkovClient(fl.client.NumPyClient):
    def __init__(self, num_states=3):
        self.num_states = num_states
        self.markov_matrix = self._initialize_markov_matrix()

    def _initialize_markov_matrix(self):
        """Generate a random Markov matrix."""
        matrix = np.random.rand(self.num_states, self.num_states)
        return matrix / matrix.sum(axis=1, keepdims=True)

    def get_parameters(self):
        return [self.markov_matrix]

    def fit(self, parameters, config):
        """Receive global Markov matrix and update local matrix."""
        if parameters:
            self.markov_matrix = np.array(parameters[0])
        # Simulate training (e.g., refine the matrix based on data)
        self.markov_matrix += np.random.normal(scale=0.01, size=self.markov_matrix.shape)
        self.markov_matrix = np.clip(self.markov_matrix, 0, 1)
        self.markov_matrix /= self.markov_matrix.sum(axis=1, keepdims=True)
        return self.get_parameters(), len(self.markov_matrix), {}

    def evaluate(self, parameters, config):
        """Evaluate the current model (optional)."""
        return 0.0, len(self.markov_matrix), {}

# Start the client
if __name__ == "__main__":
    client = MarkovClient()
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)
