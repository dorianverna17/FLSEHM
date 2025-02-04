###################################################
# This file aims to analyze and perform modelling #
#  of the GNSS ground truth dataset in order to   #
#  obtain a cold start Markov transition matrix   #
###################################################

import glob
import warnings
import hashlib

from geopandas import GeoDataFrame
import math
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import plotly.express as px
from shapely.geometry import Point
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')

DATA_PATH = "../Datasets/"

df_sample_trail = pd.read_csv(DATA_PATH + "2020-05-14-US-MTV-1/Pixel4/Pixel4_derived.csv")
df_sample_trail_gt = pd.read_csv(DATA_PATH + "2020-05-14-US-MTV-1/Pixel4/ground_truth.csv")

print("Brief look at the head of data:")
print(df_sample_trail.head())

print("A brief look at the columns:")
print(df_sample_trail.columns)

print("Brief look at the head of ground truth data:")
print(df_sample_trail_gt.head())

print("A brief look at the columns of ground truth data:")
print(df_sample_trail_gt.columns)

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

# Function used to perform hashing on the PII data
def hash_data(data):
    return hashlib.sha256(data.encode()).hexdigest()

##########################################################################
## Sample Code just to test if we can display data with Open Street Map ##
##########################################################################

# Reload dataframe so that it is easy to look at other data.
df_sample_trail_gt = pd.read_csv(DATA_PATH + "2020-05-14-US-MTV-1/Pixel4/ground_truth.csv")
df_sample_trail_gt2 = pd.read_csv(DATA_PATH + "2020-05-14-US-MTV-1/Pixel4XLModded/ground_truth.csv")

# Since plotly looks at the phoneName of the dataframe,
# you can visualize multiple series of data by simply concatting dataframes.
df_sample_trail_gt3 = pd.concat([df_sample_trail_gt, df_sample_trail_gt2])

center = {"lat":37.423576, "lon":-122.094132}
visualize_trafic(df_sample_trail_gt3, center)

##########################
## Print all train data ##
##########################
collectionNames = [item.split("/")[-1] for item in glob.glob(DATA_PATH + "*")]
print(collectionNames)

gdfs = []
for collectionName in collectionNames:
    gdfs_each_collectionName = []
    csv_paths = glob.glob(DATA_PATH + f"{collectionName}/*/ground_truth.csv")
    for csv_path in csv_paths:
        df_gt = pd.read_csv(csv_path)
        df_gt["geometry"] = [Point(latDeg, lngDeg) for latDeg, lngDeg in zip(df_gt["latDeg"], df_gt["lngDeg"])]
        gdfs_each_collectionName.append(GeoDataFrame(df_gt))
    gdfs.append(gdfs_each_collectionName)

# At this point, gdfs is a list of dataframes

all_tracks = pd.DataFrame()

for collectionName, gdfs_each_collectionName in zip(collectionNames, gdfs):   
    print(collectionName)
    for i, gdf in enumerate(gdfs_each_collectionName):
        all_tracks = pd.concat([all_tracks, gdf])

##########################################################
##   Take care of the privacy concerns in the dataset   ##
## Specifically work on hashing the name of every phone ##
##            and reconstruct the dataset               ##
##########################################################
print(all_tracks)
all_tracks.to_csv('custom_gnss.csv', index=False)

# create new column (collectionName and phoneName):
all_tracks['measurementID'] = all_tracks['collectionName'] + all_tracks["phoneName"]

# hash the phone name
all_tracks['phoneName'] = all_tracks['phoneName'].apply(hash_data)
all_tracks.to_csv('custom_gnss.csv', index=False)

# Print openstreet data map
center={"lat":37.423576, "lon":-122.094132}
visualize_trafic(all_tracks, center=center, label="measurementID")


######################################
# Determine some of the basestations #
#    Perform a K-means clustering    #
######################################
data_x = [p.x for p in all_tracks['geometry']]
data_y = [p.y for p in all_tracks['geometry']]
data = list(zip(data_x, data_y))

inertias = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

figure3 = plt.figure(3)
plt.plot(range(1,11), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
figure3.show()

##
## We've tried figuring out how many basestations we can
## divide our dataset in.
## By using the elbow method, we found out that we can
## have 2 basestations (the above graph tends to go
## horizontal when the number of clusters is 2).
##

kmeans = KMeans(n_clusters=2)
kmeans.fit(data)

# Let's also display the centroids
centroids  = kmeans.cluster_centers_
print(centroids)

figure4 = plt.figure(4)
y_centr = data_x + [c[0] for c in centroids]
x_centr = data_y + [c[1] for c in centroids]
labels = kmeans.labels_
labels = np.append(labels, [[10, 10]])
plt.scatter(x_centr, y_centr, c=labels)
figure4.show()

###################################################
# Compute how many devices changed their position #
# from their start of the movement up to the end  #
###################################################

def compute_distance_to_centroids(p):
    return [math.dist([p.x, p.y], c) for c in centroids]

# Get the starting and end point for each device
# and day this means that we should get the point
# for each first row in every ground_truth file.

# Hash the collection name along with the phone
# name so that we distinguish between measurements.
hashed_positions = []
for collectionName in collectionNames:
    csv_paths = glob.glob(DATA_PATH + f"{collectionName}/*/ground_truth.csv")
    for csv_path in csv_paths:
        df_gt = pd.read_csv(csv_path)

        hashed_id = hash_data(df_gt['collectionName'][0] +
                              '-' + df_gt['phoneName'][0])
        start_lat = df_gt['latDeg'][0]
        start_lon = df_gt['lngDeg'][0]

        end_lat = df_gt['latDeg'][len(df_gt) - 1]
        end_lon = df_gt['lngDeg'][len(df_gt) - 1]

        hashed_positions.append([hashed_id,
                                 Point(start_lat, start_lon), Point(end_lat, end_lon)])

# Now compute the length to the centroids for each
# of these hashed positions and determine whether
# the point has moved.
p_truth_movement = []
for h in hashed_positions:
    start = -1
    end = -1

    distances = compute_distance_to_centroids(h[1])
    min_d = min(distances)
    for i in range(len(distances)):
        if min_d == distances[i]:
            start = i
            break

    distances = compute_distance_to_centroids(h[2])
    min_d = min(distances)
    for i in range(len(distances)):
        if min_d == distances[i]:
            end = i
            break

    p_truth_movement.append([h[0], start, end])

# create a truth matrix with the data from the truth movement
#      [started_BS0_ended_BS0  started_BS0_ended_BS1]
#      [started_BS1_ended_BS0  started_BS1_ended_BS1]
truth_matrix = [[0, 0], [0, 0]]
for entry in p_truth_movement:
    if entry[1] == 0 and entry[2] == 0:
        truth_matrix[0][0] += 1
    elif entry[1] == 0 and entry[2] == 1:
        truth_matrix[0][1] += 1
    elif entry[1] == 1 and entry[2] == 0:
        truth_matrix[1][0] += 1
    elif entry[1] == 1 and entry[1] == 1:
        truth_matrix[1][1] += 1

print(truth_matrix)

# now compute the Markov transition matrix to serve as
# cold start matrix
stochastic_matrix = [[0, 0], [0, 0]]
for i in range(len(truth_matrix)):
    sum_row = sum(truth_matrix[i])

    for j in range(len(truth_matrix)):
        stochastic_matrix[i][j] = truth_matrix[i][j] / sum_row

print(stochastic_matrix)

#################################################
# Let's print all the useful data to a log file #
#################################################
# print data used for building simulation data
f = open("simulation_starting_data.log", 'w')
for h in hashed_positions:
    print(str(h), file=f)
f.close()
# print cold start data
g = open("cold_start_matrix.log", 'w')
print(str(stochastic_matrix), file=g)
g.close()

# this line ensures that the plots are still displayed
input()