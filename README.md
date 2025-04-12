# FLSEHM
Thesis Project - Federated Learning in Scenarios of External Human Mobility 

# Contents

## Dataset Construction

An important part of the implementation is represented by the construction of the set of data used in
federated learning.

We used GNSS data, see Datasets/ and Documentation/

To run the dataset construction and cold start Markov matrix computation:
```
python Dataset_construction/GNSS_data.py
```

The above Python script will also save the cold-start stochastic Markov matrix in the cold_start_matrix.log file.
The script also builds a log file with the starting and ending position of each point in the GNSS dataset, the file is simulation_starting_data.log.

To run simulation data generation v1:
```
python Dataset_construction/simulate_GNSS_data_v1.py
```

To run simulation data generation v2:
```
python Dataset_construction/simulate_GNSS_data_v2.py
```

To run simulation data generation v3:
```
python Dataset_construction/simulate_GNSS_data_v3.py
```

## Proof of Concept

The Proof of Concept solution contains a basic Docker configuration followed by a client app and a server app.

Running the Proof of Concept app:
```
cd Proof_of_concept/markov-dummy
sudo docker-compose up --build
```

## Flower Markov

Slightly  different approached as compared to the Proof of concept one. \
It is basically a reimplementation of it, but using Flower framework.
```
cd Flower/Markov
flwr run .
```

Logs of the client and server apps can be checked out at:
```
Flower/Markov/output/app_client.log
Flower/Markov/output/app_server.log
```

To validate differential privacy results:
```
cd Flower/Markov/dp_validation
python validate.py
```

Alternatively, in order to make use of the simulated data, based on the cold-start problem
```
Flower/Markov/start_simulation.sh
```

## Flower FedAvg

How to run

Running with linear regression model option
```
source Flower/FedAvg/start_simulation.sh linear_regression
```

Running with neural network model option
```
source Flower/FedAvg/start_simulation.sh nn_model
```

Running with nonlinear neural network model option
```
source Flower/FedAvg/start_simulation.sh nonlinear_nn_model
```

Running with enhanced neural network model option
```
source Flower/FedAvg/start_simulation.sh enhanced_model
```

# Running Conda

Prerequisited: it is necessary to use the dependencies installed with Conda:
```
export PATH="/opt/homebrew/anaconda3/bin:$PATH"
```