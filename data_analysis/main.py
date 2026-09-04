import matplotlib.pyplot as plt
from functions.read_csv import read_csv
from functions.plot import *
from functions.map_points import *

# loading the first file data
csv_file = "../data/F-30_001.csv"
json_file = "../data/marker_maps/F-30map.json"
df = read_csv(csv_file)
# renaming each marker to the correct name
df = rename_markers(df, json_file)
# finding the kinematic points given the markers
df = marker_to_kinematic_points(df)

# Plotting the markers in a wanted frame
frame = 2400
#plot_markers(df, frame)
plot_links(df, frame)