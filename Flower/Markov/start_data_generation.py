###################################################
# This file generates a list of random positions  #
# for the devices used in training the FD network #
###################################################

import sys
import time

import importlib.util

def module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

sim = module_from_file("bar", "Data_construction/simulate_GNSS_data_v2.py")

generated_points = []
count = 0
counter_processed = 0

while True:
	time.sleep(10)

	# generate points
	generated_points.append(sim.generate_next_points())
	count += 1

	# this is a timeout block (this script shouldn't run forever in our simulation)
	if count == 10:
		break
