import flwr as fl
import numpy as np

def aggregate_markov_matrices(weights):
    """Aggregate Markov matrices (averaging)."""
    aggregated = np.mean([w[0] for w in weights], axis=0)
    return [aggregated]

# Define the server strategy
class MarkovStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        # Extract Markov matrices
        weights = [res.parameters for res in results]
        aggregated_weights = aggregate_markov_matrices(weights)
        return aggregated_weights, {}

# Start the Flower server
if __name__ == "__main__":
    strategy = MarkovStrategy()
    fl.server.start_server(strategy=strategy)
