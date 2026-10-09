import json

from data_analysis.data_processing.experiment_list import get_experiment
from data_analysis.data_processing.map_points import marker_to_kinematic_points, rename_markers, report_point_quality, report_marker_availability, report_problematic_points
from data_analysis.data_processing.read_csv import read_csv
from data_analysis.data_processing.reference_frame import find_reference_frame
from data_analysis.visualization.plot import plot_markers, plot_markers_slider, plot_links


# Problem with kinematic points without all marker visible during the optimization

def main():
    file_select = "F-30"  # Change this to the desired experiment name
    csv_file, json_file = get_experiment(file_select)
    df_raw = read_csv(csv_file)

    # Rename markers and convert to kinematic points
    df_renamed = rename_markers(df_raw, json_file)

    # Convert to kinematic points, without steering wheel. 
    # We keep also the kinematic points that are mapped with only one marker, just for plotting purposes
    # but we add a flag to understand if they are cyomplete or not
    df = marker_to_kinematic_points(df_renamed)

    # find the car reference frame, only with the kinematic points with both m arkers visible
    T = find_reference_frame(df)

    # Plot raw markers
    #plot_markers(df_renamed, 0)

    with open("data/marker_maps/marker_to_kinematic_points.json", "r", encoding="utf-8") as f:
        marker_groups = json.load(f)

    # Some statistics and reports on marker availability and point quality
    report_marker_availability(df_renamed, marker_groups)
    report_point_quality(df)
    #plot_markers_slider(df, step=100)

    # Plot links for a specific frame
    #plot_links(df, 0)

    ## WIP: Transform points to car RF and save to new CSV
    # get the car RF and transform the points to the car RF, then save it to a new csv file
    # H_marker_to_car = determine_marker_to_car_transformation(df_renamed, marker_groups)
    # df_correctedRF = transform_points_to_car_RF(df, H_marker_to_car)
    # df_correctedRF.to_csv("data/processed/F0_correctedRF.csv", index=False)

if __name__ == "__main__":
    main()