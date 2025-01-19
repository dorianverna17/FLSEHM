import glob
import itertools
import json
import os
import warnings
import hashlib

import geopandas as gpd
from geopandas import GeoDataFrame
import geoplot as gplt
from IPython.display import Video
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import plotly.express as px
import pynmea2
import requests
import seaborn
from shapely.geometry import Point, shape
import shapely.wkt

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
def visualize_trafic(df, center, zoom=9):
    fig = px.scatter_mapbox(df,  
                            # Here, plotly gets, (x,y) coordinates
                            lat="latDeg",
                            lon="lngDeg",
                            #Here, plotly detects color of series
                            color="phoneName",
                            labels="phoneName",
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
        df_gt["geometry"] = [Point(lngDeg, latDeg) for lngDeg, latDeg in zip(df_gt["lngDeg"], df_gt["latDeg"])]
        gdfs_each_collectionName.append(GeoDataFrame(df_gt))
    gdfs.append(gdfs_each_collectionName)

# At this point, gdfs is a list of dataframes

all_tracks = pd.DataFrame()

for collectionName, gdfs_each_collectionName in zip(collectionNames, gdfs):   
    for i, gdf in enumerate(gdfs_each_collectionName):
        all_tracks = pd.concat([all_tracks, gdf])
        # Tracks they have same collectionName is also same
        break

##########################################################
##   Take care of the privacy concerns in the dataset   ##
## Specifically work on hashing the name of every phone ##
##            and reconstruct the dataset               ##
##########################################################
print(all_tracks)
all_tracks.to_csv('custom_gnss.csv', index=False)

# hash the phone name
all_tracks['phoneName'] = all_tracks['phoneName'].apply(hash_data)
all_tracks.to_csv('custom_gnss.csv', index=False)

# Print openstreet data map
center={"lat":37.423576, "lon":-122.094132}
visualize_trafic(all_tracks, center=center)
