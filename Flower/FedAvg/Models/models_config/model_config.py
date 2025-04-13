import json

class ModelConfig():
	def __init__(self, json_config):
		with open(json_config) as json_data:
			config = json.load(json_data)

			self.model_name = config['model_name']
			self.shape = config['shape']
			self.layers = config['layers']
			self.optimizer = config['optimizer']
			self.learning_rate = config['learning_rate']
			self.initial_parameters = config['initial_parameters']
