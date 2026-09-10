import matplotlib.pyplot as plt
from functions.read_csv import read_csv
from functions.plot import *
from functions.map_points import *

file_select = "F90"

match file_select:
    case "F-30":
        csv_file = "../data/F-30_001.csv"
        json_file = "../data/marker_maps/F-30map.json"

    case "F-60":
        csv_file = "../data/F-60_001.csv"
        json_file = "../data/marker_maps/F-60map.json"

    case "F-90":
        csv_file = "../data/F-90_001.csv"
        json_file = "../data/marker_maps/F-90map.json"

    case "F-110":
        csv_file = "../data/F-110_001.csv"
        json_file = "../data/marker_maps/F-110map.json"

    case "F0":
        csv_file = "../data/F0_001.csv"
        json_file = "../data/marker_maps/F0map.json"

    case "F30":
        csv_file = "../data/F30_001.csv"
        json_file = "../data/marker_maps/F30map.json"

    case "F60":
        csv_file = "../data/F60_001.csv"
        json_file = "../data/marker_maps/F60map.json"

    case "F90":
        csv_file = "../data/F90_002.csv"
        json_file = "../data/marker_maps/F90map.json"

    case _:
        print("Error")

df = read_csv(csv_file)
# renaming each marker to the correct name
df = rename_markers(df, json_file)
# finding the kinematic points given the markers
df = marker_to_kinematic_points(df)

# Plotting the markers in a wanted frame
#plot_markers(df, 0)
#plot_links(df, frame)
plot_markers_slider(df, step=100)

print(df)
plt.plot(df["GPS_F"])