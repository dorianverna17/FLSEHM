#! /bin/bash

export PATH="/opt/homebrew/anaconda3/bin:$PATH"

# preliminary cleanup
rm -r Flower/FedAvg/output
rm -r Flower/FedAvg/generated_points/*

# recreate output directory
mkdir Flower/FedAvg/output
touch Flower/FedAvg/output/app_server.log

python Flower/FedAvg/start_data_generation.py

if [[ $# -eq 0 ]]; then
	echo "No arguments supplied, running linear regression_model"
	export FD_MODEL="linear_regression"
else
	if [[ $1 = "linear_regression" ]]; then
		echo "running with linear regression"
		export FD_MODEL="linear_regression"
	elif [[ $1 = "nn_model" ]]; then
		echo "running with a neural network model"
		export FD_MODEL="nn_model"
	elif [[ $1 = "nonlinear_nn_model" ]]; then
		echo "running with a nonlinear neural network model"
		export FD_MODEL="nonlinear_nn_model"
	elif [[ $1 = "enhanced_model" ]]; then
		echo "running with an enhanced neural network model"
		export FD_MODEL="enhanced_model"
		if [[ $# -eq 2 ]]; then
			echo "configuration used for enhanced model is $2"
			export FD_MODEL_CONFIG=$2 # TODO - to check whether the config really exists
		else
			echo "no configuration specified, running with default config1.json"
			export FD_MODEL_CONFIG="Flower/FedAvg/Models/models_config/config1.json"
		fi
	else
		echo "unimplemented model"
		exit 1
	fi
fi

python Flower/FedAvg/server.py
