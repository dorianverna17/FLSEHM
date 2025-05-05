#########################################################
# From app_server.log, take the last aggregated matrix. #
# Generate random starting and ending points.           #
# predict the ending value of the movement.             #
#########################################################

import numpy as np
import random
import os
import importlib.util
import math

def module_from_file(module_name, file_path):
	spec = importlib.util.spec_from_file_location(module_name, file_path)
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module

sim = module_from_file("bar", "Data_construction/simulate_GNSS_data_v2.py")

def get_basestation(centroids, point):
	min_distance = float('inf')
	basestation = -1

	for i, (cx, cy) in enumerate(centroids):
		distance = math.sqrt((point[0] - cx) ** 2 + (point[1] - cy) ** 2)
		if distance < min_distance:
			min_distance = distance
			basestation = i

	return basestation

# Step 1 - get latest aggregated matrix
matrix = [[], []]
with open("Flower/Markov/output/app_server.log", "r") as f:
	l = f.readline()
	while len(l) != 0:
		if "Aggregated matrices" in l:
			l1 = l
			l2 = f.readline()
		l = f.readline()

	# take l1 and l2 and get the matrix
	matrix[0].append(float(l1[l1.find("[[") + 2:l1.find("0.", l1.find("[[") + 3) - 1]))
	matrix[0].append(float(l1[l1.find("0.", l1.find("[[") + 3):-2]))

	matrix[1].append(float(l2[l2.find("[") + 1:l2.find("0.", l2.find("[") + 2) - 1]))
	matrix[1].append(float(l2[l2.find("0.", l2.find("[") + 2):-3]))

	print(matrix)

# Step 2 - generate random points for testing
test_data = []
for i in range(1, 100):
	test_data.append([sim.generate_random_point(), sim.generate_random_point()])

# Step 3 - translate points to basestations
# each client needs to know where the centroids are
# Read from file
centroids = []
with open("Data_Construction/centroids.log", "r", os.O_NONBLOCK) as f:
    for line in f:
        centroids.append(np.fromstring(line.strip()[1:-1], sep=' '))

for i in range(len(test_data)):
	test_data[i][0] = get_basestation(centroids, test_data[i][0])
	test_data[i][1] = get_basestation(centroids, test_data[i][1])

# Step 4 - feed generated points to transition matrix
predicted_data = []
for i in range(len(test_data)):
	if test_data[i][0] == 0:
		if matrix[0][0] > matrix[0][1]:
			predicted_data.append(0)
		else:
			predicted_data.append(1)
	else:
		if matrix[1][0] > matrix[1][1]:
			predicted_data.append(0)
		else:
			predicted_data.append(1)

# Step 5 - compute accuracy of results
count = 0
for i in range(len(test_data)):
	if predicted_data[i] == test_data[i][1]:
		count += 1
	
accuracy = (100 * count) / len(test_data)
print(accuracy)
