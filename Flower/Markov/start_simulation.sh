#! /bin/bash

export PATH="/opt/homebrew/anaconda3/bin:$PATH"

python start_data_generation.py
flwr run .