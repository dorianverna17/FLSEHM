import logging
import constants
import numpy as np

# Function that adds noise for local differential privacy approach
def local_dp_laplace(matrix):
    sensitivity = constants.sensitivity
    epsilon = constants.epsilon
    noise = np.random.laplace(loc=0, scale=sensitivity / epsilon, size = matrix.shape)
    noisy_matrix = matrix + noise

    # Assuming noisy_matrix is already generated
    noisy_matrix = np.maximum(noisy_matrix, 0)  # Ensure non-negative entries

    # Normalize each row to ensure the row sums to 1
    row_sums = noisy_matrix.sum(axis=1, keepdims=True)  # Sum of each row
    normalized_matrix = noisy_matrix / row_sums  # Divide each row by its sum
    
    # If any row sums to zero (which can happen due to noise), set it to a uniform distribution (or another suitable handling)
    normalized_matrix[np.isnan(normalized_matrix)] = 0
    normalized_matrix[normalized_matrix.sum(axis=1) == 0] = 1 / noisy_matrix.shape[1]

    return normalized_matrix


def local_dp_gaussian(matrix):
    delta = constants.delta
    sensitivity = constants.sensitivity
    epsilon = constants.epsilon

    sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
    noise = np.random.normal(loc=0, scale=sigma, size=matrix.shape)
    noisy_matrix = matrix + noise

    # Assuming noisy_matrix is already generated
    noisy_matrix = np.maximum(noisy_matrix, 0)  # Ensure non-negative entries

    # Normalize each row to ensure the row sums to 1
    row_sums = noisy_matrix.sum(axis=1, keepdims=True)  # Sum of each row
    normalized_matrix = noisy_matrix / row_sums  # Divide each row by its sum

    # If any row sums to zero (which can happen due to noise), set it to a uniform distribution (or another suitable handling)
    normalized_matrix[np.isnan(normalized_matrix)] = 0
    normalized_matrix[normalized_matrix.sum(axis=1) == 0] = 1 / noisy_matrix.shape[1]
    
    return normalized_matrix


# TODO(dorianverna)
def local_dp_discrete_laplace(matrix):
    pass

# TODO(dorianverna)
def local_dp_exponential(matrix):
    pass

# Function that adds noise for central differential privacy approach
def central_dp_noise():
    pass


def add_noise(dp_type, noise_type, matrix):
    if dp_type == "local":
        if noise_type == "laplace":
            return local_dp_laplace(matrix)
        elif noise_type == "gaussian":
            return local_dp_gaussian(matrix)
        elif noise_type == "discrete_laplace":
            return local_dp_discrete_laplace(matrix)
        elif noise_type == "exponential":
            return local_dp_exponential(matrix)
        # return default value
        return local_dp_gaussian(matrix)
    elif dp_type == "central":
        return central_dp_noise(matrix)
    else:
        logging.info("Unknown/Unimplemented differential privacy type")
        return None
