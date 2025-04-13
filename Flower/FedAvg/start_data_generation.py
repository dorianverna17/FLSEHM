###################################################
# This file generates a list of random positions  #
# for the devices used in training the FD network #
###################################################

import sys
import time
import os
import random
import importlib.util

from string import hexdigits
from hashlib import sha256
from constants import RANDOM_SAMPLED_DEVICES, HASH_LENGTH
from shapely.geometry import Point

os.environ["RAY_DEDUP_LOGS"]="0"

def module_from_file(module_name, file_path):
	spec = importlib.util.spec_from_file_location(module_name, file_path)
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module

sim = module_from_file("bar", "Data_construction/simulate_GNSS_data_v2.py")

count = 0

# Create output directory
output_dir = "Flower/FedAvg/generated_points"
os.makedirs(output_dir, exist_ok=True)

# get the cold data points
data = sim.data
hash_ids, start_p_list, end_p_list = zip(*data)

# This function aims to create new instances of devices, separate for the real ones
# which are reflected by the GNSS Dataset. We will create different files with the
# nonlinear data in order to keep sampled data separated, even though all the entries
# will be mixed in the simulation.
def generate_new_hashes():
	"""
	Device hashes are 64 bit long strings. We need to check whether the hases are unique.
	"""
	hashes = set()
	for i in range(RANDOM_SAMPLED_DEVICES):
		input = random.choice(hexdigits)
		hash = sha256(input.encode('utf-8'))
		while hash in hashes or hash in hash_ids:
			input = random.choice(hexdigits)
			hash = sha256(input.encode('utf-8'))
		hashes.add(hash)

	hashes = list(map(lambda x : x.hexdigest(), hashes))
	return hashes


# This function generates a test entry for each random device included in the simulation.
# It strongly relates to the generate_new_hashes function above.
def generate_sampled_devices_entry(hashes):
	random_data = []

	for hash in random_hashes:
		random_data.append([hash, sim.generate_random_point(), sim.generate_random_point()])

	return tuple(x[0] for x in random_data), tuple(Point(x[1][0], x[1][1]) for x in random_data), \
		tuple(Point(x[2][0], x[2][1]) for x in random_data)


# This function represents the main entrypoint of this script, taking care of the
# generation of sampled data for all the devices considered within the simulation,
# irrespective whether we are reffering to randomly generated devices or real ones
def start_generation(random_hashes):
	global count, hash_ids, start_p_list, end_p_list

	random_h_ids, r_s_p_list, r_e_p_list = generate_sampled_devices_entry(random_hashes)

	hash_ids += random_h_ids
	start_p_list += r_s_p_list
	end_p_list += r_e_p_list

	while True:
		# Generate points
		new_points = sim.generate_next_points(len(hash_ids))

		filename = os.path.join(output_dir, f"generated_points_{count}.txt")
		with open(filename, "w+") as file:
			input = ""
			for i, h in enumerate(hash_ids):
				input += str(start_p_list[i]) + " " + str(end_p_list[i]) + str({hash_ids[i]}) + "\n"
			file.write(input)

		start_p_list = end_p_list
		end_p_list = new_points
		count += 1

		# Timeout block - exit after 10 iterations
		if count == 20:
			break


if __name__ == "__main__":
	random_hashes = generate_new_hashes()
	start_generation(random_hashes)
