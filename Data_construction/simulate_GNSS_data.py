###################################################
# This file aims to provide training data for the #
#    federated learning solutions proposed, by    #
# analyzing the log files created by GNSS_data.py #
###################################################

from shapely.geometry import Point
import os
import glob
import pandas as pd
import plotly.express as px
import random

# log file taken:
# - simulation_starting_data.log

# Step 0. Declare auxiliary variables and functions
excluded_points = [
    [Point(37.510, -122.802), Point(37.510, -122.490), Point(37.778, -122.490), Point(37.778, -122.802)],
    [Point(37.804, -122.545), Point(37.804, -122.088), Point(37.850, -122.088), Point(37.850, -122.545)],
    [Point(37.605720, -122.371855), Point(37.767330, -122.156121), Point(37.745981, -122.156121), Point(37.745981, -122.371855)],
    [Point(37.294, -122.448), Point(37.294, -121.740), Point(37.200, -121.740), Point(37.200, -122.448)],
    # [Point(0, 0), Point(0, 0), Point(0, 0), Point(0, 0)], # add a new point if you need it
]
no_points_to_create = 2000

def validate_point(p):
    """
    This function checks whether an existing point is within the
    area covered by the points specified in the exclusion list

    Args:
        p - The point that is checked
    
    Ret:
        boolean - True if the point is valid (not within the
        exclusion list), False otherwise
    """
    for ex_p in excluded_points:
        if p.x > ex_p[0].x and p.x < ex_p[2].x \
            and p.y > ex_p[0].y and p.y < ex_p[1].y:
            return False
    return True


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

f.close()


def get_new_direction(d):
    """
    TODO add commentary
    """
    random_dir = random.randint(0, 3)

    if random_dir == 0:
        if d != 'S':
            direction = 'N'
        else:
            direction = 'S'
    elif random_dir == 1:
        if d != 'N':
            direction = 'S'
        else:
            direction = 'N'
    elif random_dir == 2:
        if d != 'E':
            direction = 'W'
        else:
            direction = 'S'
    elif random_dir == 3:
        if d != 'W':
            direction = 'E'
        else:
            direction = 'W'

    return direction


def get_next_point(p, direction='N', threshold=100, counter=0):
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
        The direction where the point heads to
    """
    if counter > threshold:
        direction = get_new_direction(direction)

    if direction == 'N':
        return Point(p.x + 0.001, p.y), direction
    elif direction == 'S':
        return Point(p.x - 0.001, p.y), direction
    elif direction == 'W':
        return Point(p.x, p.y + 0.001), direction
    elif direction == 'E':
        return Point(p.x, p.y - 0.001), direction

    return None, direction


def build_trajectory(start, no_points):
    """
    This function computes a trajectory for each point in the read
    dataset

    Args:
        start - The starting point of the device
        no_points - The number of points to calculate

    Ret:
        trajectory - a list of points for the device
    """
    direction = get_new_direction('N')
    
    print("Build trajectory for point " + str(start))

    trajectory = []
    current = start
    counter = 0

    # prioritze going north or south
    no_south = 0
    no_north = 0
    no_west = 0
    no_east = 0

    for _ in range(no_points):
        prev = Point(current.x, current.y)
        if no_south + no_north < 2 * (no_west + no_east):
            direction = random.randint(0, 1)
            if direction == 0:
                direction = 'N'
            else:
                direction = 'S'

        current, direction = get_next_point(prev, direction, 30, counter)
        while validate_point(current) is False:
            current, direction = get_next_point(prev, get_new_direction(direction), 30, counter)
        trajectory.append(current)

        if counter > 30:
            counter = 0

        counter += 1

    print("Finished building trajectory for point " + str(start))

    return trajectory


# Step 2. Build random trajectories for each point
trajectories = []
for point in data:
    trajectories.append([point[0], build_trajectory(point[2], no_points_to_create)])
    print("Appended trajectory for point " + str(point[2]))

# Step 3. Write trajectories to collection files
if not os.path.isdir("./Trajectories"):
    os.makedirs("./Trajectories")
else:
    files = glob.glob("./Trajectories/*")
    for f in files:
        os.remove(f)
for trajectory in trajectories:
    trajectory_filename = "./Trajectories/future_trajectories_" + trajectory[0] + '.txt'
    g = open(trajectory_filename, 'x')
    for t in trajectory[1]:
        print(str(t), file=g)
    g.close()

# Step 4. Print a sample of a trajectory - Use Open Street Map
# Function used to visualize traffic obtained in the ground truth
def visualize_trafic(df, center, label="phoneName", zoom=9):
    fig = px.scatter_mapbox(df,  
                            # Here, plotly gets, (x,y) coordinates
                            lat="latDeg",
                            lon="lngDeg",
                            #Here, plotly detects color of series
                            color=label,
                            labels=label,
                            zoom=zoom,
                            center=center,
                            height=900,
                            width=1200)
    fig.update_layout(mapbox_style='open-street-map')
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update_layout(title_text="GPS trafic")
    fig.show()

df_single_device = pd.DataFrame(
    {
        'phoneName': no_points_to_create * [trajectories[0][0]],
        'latDeg': [p.x for p in trajectories[0][1]],
        'lngDeg': [p.y for p in trajectories[0][1]]
    }
)
center = {"lat":37.423576, "lon":-122.094132}
visualize_trafic(df_single_device, center)

# Step 5. Print more trajectories - Use Open Street Map
counter = 10
devices_dict = {
    'phoneName': [],
    'latDeg': [],
    'lngDeg': []
}
for t in trajectories:
    counter -= 1
    for i in range(no_points_to_create):
        devices_dict['phoneName'].append(t[0])
        trj_p = t[1][i]
        devices_dict['latDeg'].append(trj_p.x)
        devices_dict['lngDeg'].append(trj_p.y)
    if counter == 0:
        break

df_multiple_devices = pd.DataFrame(devices_dict)
center = {"lat":37.423576, "lon":-122.094132}
visualize_trafic(df_multiple_devices, center)

