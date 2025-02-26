#! /bin/bash

export PATH="/opt/homebrew/anaconda3/bin:$PATH"

# preliminary cleanup
rm -r Flower/Markov/output

# recreate output directory
mkdir Flower/Markov/output
touch Flower/Markov/output/app_client.log
touch Flower/Markov/output/app_server.log

python Flower/Markov/start_data_generation.py &
echo "$!"
flwr run Flower/Markov/