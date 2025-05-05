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

def start_generation():
	global count, hash_ids, start_p_list, end_p_list

	while True:
		# Generate points
		new_points = sim.generate_next_points(len(hash_ids))

		filename = os.path.join(output_dir, f"generated_points_{count}.txt")
		with open(filename, "w+") as file:
			input = ""
			for i, point in enumerate(new_points):
				if i == len(new_points) - 1:
					break
				input += str(point) + " " + str(end_p_list[i]) + str({hash_ids[i]}) + "\n"
			input += str(point) + " " + str(end_p_list[len(new_points) - 1]) + str({hash_ids[len(new_points) - 1]})
			file.seek(0)
			file.write(input)

		end_p_list = new_points
		count += 1

		# Timeout block - exit after 10 iterations
		if count == 10:
			break


if __name__ == "__main__":
	start_generation()
