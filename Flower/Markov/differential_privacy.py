import logging
import constants
import numpy as np

# Function that adds noise for local differential privacy approach
def local_dp_noise(matrix):
    sensitivity = constants.sensitivity
    epsilon = constants.epsilon
    noise = np.random.laplace(loc=0, scale=sensitivity / epsilon, size = matrix.shape)

    noisy_matrix = matrix + noise
    noisy_matrix = noisy_matrix / noisy_matrix.sum(axis=1, keepdims=True)

    noisy_matrix = np.clip(noisy_matrix, 0, 1)

    return noisy_matrix


# Function that adds noise for central differential privacy approach
def central_dp_noise():
    pass


def add_noise(dp_type, matrix):
    if dp_type == "local":
        return local_dp_noise(matrix)
    elif dp_type == "central":
        return central_dp_noise(matrix)
    else:
        logging.info("Unknown/Unimplemented differential privacy type")