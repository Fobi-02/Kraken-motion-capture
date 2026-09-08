import matplotlib.pyplot as plt
from functions.read_csv import read_csv
from functions.plot import *
from functions.map_points import *

file_select = "F-90"

if file_select == "F-30":
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
    #plot_links(df, frame)
    plot_markers_slider(df)

elif file_select == "F-60":
    csv_file = "../data/F-60_001.csv"
    json_file = "../data/marker_maps/F-60map.json"
    df = read_csv(csv_file)
    df = rename_markers(df, json_file)
    df = marker_to_kinematic_points(df)
    plot_markers(df, 0)

elif file_select == "F-90":
    csv_file = "../data/F-90_001.csv"
    json_file = "../data/marker_maps/F-90map.json"
    df = read_csv(csv_file)
    df = rename_markers(df, json_file)
    print("P4: ", df.loc[0, "p04_Z"], " P5: ", df.loc[0, "p05_Z"], " P6: ", df.loc[0, "p06_Z"])
    df = marker_to_kinematic_points(df)
    plot_markers_slider(df, 200)
    
    

elif file_select == "SW":
    csv_file = "../data/SW_002.csv"
    df = read_csv(csv_file)
    plot_markers_slider(df, 200)

# test of steering angle function
P4 = np.array([94.1511, 46.9846, 183.93])
P5 = np.array([89.0077, -38.3022, 169.799])
P6 = np.array([116.841, -8.68241, 246.271])
delta = steering_angle(P4,P5,P6)
#print(delta)