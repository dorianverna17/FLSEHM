###################################################
# This file generates a list of random positions  #
# for the devices used in training the FD network #
###################################################

import sys
import time

sys.path.append('../Data_construction/')

from Data_construction.simulate_GNSS_data_v2 import generate_next_points

generated_points = []
count = 0
counter_processed = 0

while True:
	time.sleep(10)

	# generate points
	generated_points.append(generate_next_points())
	count += 1
