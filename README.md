# FLSEHM
Thesis Project - Federated Learning in Scenarios of External Human Mobility 

# Contents
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

# Running Conda

Prerequisited: it is necessary to use the dependencies installed with Conda:
```
export PATH="/opt/homebrew/anaconda3/bin:$PATH" \
```