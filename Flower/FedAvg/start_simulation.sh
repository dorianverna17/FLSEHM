#! /bin/bash

export PATH="/opt/homebrew/anaconda3/bin:$PATH"

# preliminary cleanup
rm -r Flower/FedAvg/output
rm -r Flower/FedAvg/generated_points/*

# recreate output directory
mkdir Flower/FedAvg/output
touch Flower/FedAvg/output/app_server.log

python Flower/FedAvg/start_data_generation.py

python Flower/FedAvg/server.py