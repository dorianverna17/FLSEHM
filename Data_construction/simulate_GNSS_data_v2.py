###################################################
# This file aims to provide training data for the #
#    federated learning solutions proposed, by    #
# analyzing the log files created by GNSS_data.py #
###################################################

from shapely.geometry import Point
import random

# As opposed to v1, we now just generate a starting point
# and an ending point for a certain timeframe, we are no
# longer interested in generating the while trajectory

# Define the future points' areas. We will have random points
# generated within these areas randomly.
allowlist_map = [
    [[37.512986, -122.482981], [37.793298, -122.409622]],
    [[37.371691, -122.398847], [37.553132, -122.274443]],
    [[37.277468, -122.269079], [37.478106, -122.141880]],
    [[37.226661, -122.138627], [37.418525, -121.850868]],
]


def generate_random_point():
    """
    This function generates a random point within the
    allowlist map's coordinates.
    
    Ret:
        A new random point within the allowlist map coordinates
    """

    allowlist_entry = random.randint(0, 3)

    # Generate a point within the allowlist entry selected
    lat = round(random.uniform(allowlist_map[allowlist_entry][0][0],
                               allowlist_map[allowlist_entry][1][0]), 6)

    lon = round(random.uniform(allowlist_map[allowlist_entry][0][1],
                               allowlist_map[allowlist_entry][1][1]), 6)

    return [lat, lon]


# Step 1. Read file data in a list
# TODO - This code is duplicated - consider adding a helper package
f = open("simulation_starting_data.log", "rt")
data = []
line = f.readline()
while line:
    hash_id = line[line.find('\'') + 1:line.find('\'', 3)]
    
    sp_index = line.find('POINT')
    starting_point = line[sp_index + 6:line.find('>', sp_index)]
    starting_point_x = starting_point[1:starting_point.find(' ')]
    starting_point_y = starting_point[starting_point.find(' '):len(starting_point) - 1]

    sp_index = line.find('POINT', sp_index + 1)
    ending_point = line[sp_index + 6:line.find('>', sp_index)]
    ending_point_x = ending_point[1:ending_point.find(' ')]
    ending_point_y = ending_point[ending_point.find(' '):len(ending_point) - 1]
    
    data.append([hash_id, Point(starting_point_x, starting_point_y),
                 Point(ending_point_x, ending_point_y)])

    line = f.readline()

f.close()


def generate_next_points():
    """
    This function generates the next positions of all the devices
    that were part of the GNSS datasets.
    
    Ret:
		A list of ending points for the devices from the GNSS dataset
    """
    ending_points_list = []
    for d in data:
        next_point = generate_random_point()
        ending_points_list.append(Point(next_point[0], next_point[1]))
    return ending_points_list
	