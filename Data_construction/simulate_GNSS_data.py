###################################################
# This file aims to provide training data for the #
#    federated learning solutions proposed, by    #
# analyzing the log files created by GNSS_data.py #
###################################################

from shapely.geometry import Point

# log file taken:
# - simulation_starting_data.log

# Step 1. Read file data in a list
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


def get_random_next_point(p, direction='N', threshold=100, counter=0):
    """
    This function computes the position of a nearby point next to
    a given point with a difference of 0.001 degrees in one direction.

    Note: the function optionally receives a parameter stating the
    direction tendency and a threshold and incremented counter which,
    in case it exceeds the threshold, the function randomly chooses to
    change the direction where the relative position of the point will
    change.

    Args:
        p - The point relative to which the returned
        point is determined.
        direction - Direction relative to p, where the computed point
        will be.
        threshold - The amount of times the point is allowed to be
        in the direction specified relative to p.
        counter - The amount of times the point is calculated to be
        in the direction specified relative to p.
    
    Ret:
        The point next to p, computed by adding, decreasing
        latitude/longitude.
    """
    pass


def build_trajectory(hash_id, start):
    """
    This function computes a trajectory for each point in the read
    dataset

    Args:
        hash_id - the device for which we compute the trajectory
        start - The starting point of the device
        
    Ret:
        trajectory - a list of points for the device
    """
    pass


# Step 2. Build random trajectories for each point
# TODO.

# Step 3. Write trajectories to collection files
# TODO.

f.close()
