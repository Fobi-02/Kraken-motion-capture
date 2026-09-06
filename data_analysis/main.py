import matplotlib.pyplot as plt
from functions.read_csv import read_csv
from functions.plot import *
from functions.map_points import *

file_select = "F-60"

if file_select == "F-30":
    csv_file = "../data/F-30_001.csv"
    json_file = "../data/marker_maps/F-30map.json"
    df = read_csv(csv_file)
    # renaming each marker to the correct name
    df = rename_markers(df, json_file)
    # finding the kinematic points given the markers
    #df = marker_to_kinematic_points(df)

    # Plotting the markers in a wanted frame
    frame = 2400
    plot_markers(df, frame)
    #plot_links(df, frame)
    #plot_markers_slider(df)

elif file_select == "F-60":
    csv_file = "../data/F-60_001.csv"
    json_file = "../data/marker_maps/F-60map.json"
    df = read_csv(csv_file)
    df = rename_markers(df, json_file)
    #df = marker_to_kinematic_points(df)
    plot_markers(df, 0)