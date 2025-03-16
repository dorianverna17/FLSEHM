import flwr as fl
import pandas as pd
import numpy as np

from flwr.common import FitRes, EvaluateRes, Parameters, Scalar
from typing import Dict, Tuple, List
from Models.linear_regression import LinearRegressionModel

class PositionPredictionClient(fl.client.NumPyClient):
	def __init__(self, device_data: pd.DataFrame):
		self.device_data = device_data
		self.model = LinearRegressionModel()

	def get_parameters(self, config: Dict[str, Scalar]) -> List[np.ndarray]:
		return self.model.get_parameters()

	def set_parameters(self, parameters: List[np.ndarray]) -> None:
		self.model.set_parameters(parameters)

	def fit(self, parameters: List[np.ndarray], config: Dict):
		self.set_parameters(parameters)

		X = self.device_data[['initial_lat', 'initial_lon']]
		y_lat = self.device_data['final_lat']
		y_lon = self.device_data['final_lon']

		self.model.train(X, y_lat, y_lon)

		return FitRes(
			status=fl.common.Status.OK,
			parameters=self.get_parameters(config),
			num_examples=len(self.device_data),
			metrics={},
		)

	def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Scalar]):
		self.set_parameters(parameters)
		
		return EvaluateRes(
			status=fl.common.Status.OK,
			loss=0.0,
			num_examples=len(self.device_data),
			metrics={},
		)

def client_fn(device_hash: str, df: pd.DataFrame) -> fl.client.NumPyClient:
	device_data = df[df['device_hash'] == device_hash]
	return PositionPredictionClient(device_data)

def generate_client_resources(df: pd.DataFrame) -> Dict[str, fl.client.ClientFn]:
	device_hashes = df['device_hash'].unique()
	client_resources = {int(device_hash[len(device_hash) - 1]) - 1:lambda : client_fn(device_hash, df) for device_hash in device_hashes}
	return client_resources
