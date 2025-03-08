import numpy as np
import math

from shapely.geometry import Point
from constants import BASESTATIONS
from typing import Tuple, List

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


# This function parses a string, returning a Point object
def parse_point(point: str) -> Point:
	x_index = point.find('(')
	y_index = x_index + point[x_index:].find(' ')

	point_x = point[x_index + 1:y_index]
	point_y = point[y_index + 1:point.find(')')]

	return Point(float(point_x), float(point_y))


def parse_generated_points(line:str) -> Tuple[Point, Point, str]:
	# get the index of the second point (ending one)
	index_end_p = line.find(')')

	index_end_p += 1

	start_p = parse_point(line[:index_end_p])
	end_p = parse_point(line[index_end_p:index_end_p + line[index_end_p:].find(')')])

	hash_id = line[index_end_p + line[index_end_p:].find(')') + 1:]

	return (start_p, end_p, hash_id)


# This function returns the index of the centroid to which
# the point belongs
def get_basestation(centroids, point):
	min_distance = float('inf')
	basestation = -1

	for i, (cx, cy) in enumerate(centroids):
		distance = math.sqrt((point.x - cx) ** 2 + (point.y - cy) ** 2)
		if distance < min_distance:
			min_distance = distance
			basestation = i

	return basestation


# This function creates a transition matrix, based on the
# movement of the given point between basestations
def create_matrix_with_points(points: List[Point], num_basestations: int) -> np.ndarray:
	transition_matrix = []
	for i in range(num_basestations):
		transition_matrix.append([0 for x in range(num_basestations)])

	for p in points:
		transition_matrix[p[0]][p[1]] += 1

	for i in range(len(transition_matrix)):
		count_p = 0
		for j in range(len(transition_matrix[i])):
			count_p += transition_matrix[i][j]
		if count_p != 0:
			for j in range(len(transition_matrix[i])):
				transition_matrix[i][j] /= count_p

	return transition_matrix
