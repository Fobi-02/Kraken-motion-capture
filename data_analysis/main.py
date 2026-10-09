import json

from data_analysis.data_processing.experiment_list import EXPERIMENTS, get_experiment
from data_analysis.data_processing.map_points import marker_to_kinematic_points, rename_markers, report_point_quality, report_marker_availability
from data_analysis.data_processing.read_csv import read_csv
from data_analysis.data_processing.reference_frame import find_reference_frame, transform_points_to_car_RF
from data_analysis.data_processing.steering import extract_steering_angle


def main():
    file_select = "F-30"
    csv_file, json_file = get_experiment(file_select)
    df_raw = read_csv(csv_file)

    # Rename markers and convert to kinematic points
    df_renamed = rename_markers(df_raw, json_file)

    # Convert to kinematic points, keeping the 3 steering wheel markers
    # We keep also the kinematic points that are mapped with only one marker, just for plotting purposes
    # but we add a flag to understand if they are complete or not
    df = marker_to_kinematic_points(df_renamed)
    print(df)
    # find the car reference frame, only with the fixed kinematic points and those with both markers visible (mean over time)
    #T = find_reference_frame(df)

    # Transform kinematic points to car RF and save to new CSV. We transform all the kin points, also those with only one marker just for plotting
    # aand we keep the flag to understand if they are complete or not
    # we transform also the steering wheel points
    #df_carRF = transform_points_to_car_RF(df, T)

    # Determine steering angle and save to new CSV
    #df_carRF = extract_steering_angle(df_carRF, T)
    #df_carRF.to_csv(f"data/processed/{file_select}_correctedRF.csv", index=False)


    with open("data/marker_maps/marker_to_kinematic_points.json", "r", encoding="utf-8") as f:
        marker_groups = json.load(f)

    # Some statistics and reports on marker availability and point quality
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