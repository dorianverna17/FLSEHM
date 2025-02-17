###################################################
# This file aims to provide training data for the #
#    federated learning solutions proposed, by    #
# analyzing the log files created by GNSS_data.py #
###################################################

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


print(generate_random_point())

