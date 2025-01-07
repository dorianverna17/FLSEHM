import flwr as fl
import numpy as np

num_clients = 10

transition_matrix = np.random.rand(num_clients, num_clients)
transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)

state_distribution = np.ones(num_clients) / num_clients

class MyStrategy(fl.server.strategy.FedAvg):
    def __init__(self, transition_matrix, state_distribution, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transition_matrix = transition_matrix
        self.state_distribution = state_distribution

    def aggregate_fit(
        self,
        rnd: int,
        results: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: list[BaseException],
    ) -> tuple[Optional[fl.common.Parameters], dictp[str, fl.common.Scalar]]:
        if not results:
            return None, {}

        if not self.accept_failures and failures:
            return None, {}
        
        weights_results = [
            (parameters_to_weights(fit_res.parameters), fit_res.num_examples)
            for client, fit_res in results
        ]
        return weights_to_parameters(aggregate(weights_results)), {}


    def configure_fit(self, rnd, parameters, client_manager):
        selected_clients = np.random.choice(
            client_manager.all().ids,
            size=int(len(client_manager.all().ids * self.fraction_fit)),
            replace=False,
            p=self.state_distribution
        )

        self.state_distribution = self.state_distribution @ self.transition_matrix

        return super().configure_fit(rnd, parameters, client_manager)


strategy = MyStrategy(
    fraction_fit=0.1,
    transition_matrix=transition_matrix,
    state_distribution=state_distribution
)

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy
)
