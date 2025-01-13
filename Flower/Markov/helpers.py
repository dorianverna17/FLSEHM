import numpy as np

from constants import BASESTATIONS

# This function generates a random markov matrix
# based on the number of basestations
def generate_random_markov_matrix():
    transition_matrix = []
    
    for _ in range(BASESTATIONS):
        transition_probabilities = []
        sum_probabilities = 0
        for _ in range(BASESTATIONS):
            transition_probabilities += [np.random.rand() % 10]
            sum_probabilities += transition_probabilities[-1]
        transition_probabilities = [x / sum_probabilities
                                    for x in transition_probabilities]
        
        transition_matrix += [transition_probabilities]

    return transition_matrix
        