#! /bin/bash

export PATH="/opt/homebrew/anaconda3/bin:$PATH"

# preliminary cleanup
rm -r Flower/Markov/output
rm -r Flower/Markov/generated_points/*

# recreate output directory
mkdir Flower/Markov/output
touch Flower/Markov/output/app_server.log

python Flower/Markov/start_data_generation.py &

sleep 5

flwr run Flower/Markov/