###################################################
# This file generates a list of random positions  #
# for the devices used in training the FD network #
###################################################

import sys
import time
import os
import importlib.util

def module_from_file(module_name, file_path):
	spec = importlib.util.spec_from_file_location(module_name, file_path)
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module

sim = module_from_file("bar", "Data_construction/simulate_GNSS_data_v2.py")

count = 0

# Create output directory
output_dir = "Flower/Markov/generated_points"
os.makedirs(output_dir, exist_ok=True)

# get the cold data points
data = sim.data
hash_ids, start_p_list, end_p_list = zip(*data)

while True:
	time.sleep(10)

	# Generate points
	new_points = sim.generate_next_points()

	# TODO - we also have to retain the last point where the device was
	# Write to an intermediate file
	filename = os.path.join(output_dir, f"generated_points_{count}.txt")
	with open(filename, "w") as file:
		for i, point in enumerate(new_points):
			file.write(f"{point} {end_p_list[i]} {hash_ids[i]}\n")

	end_p_list = new_points
	count += 1

	# Timeout block - exit after 10 iterations
	if count == 10:
		break
