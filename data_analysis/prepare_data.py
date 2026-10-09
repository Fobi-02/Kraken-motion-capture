import json

from data_analysis.data_processing.experiment_list import EXPERIMENTS, get_experiment
from data_analysis.data_processing.map_points import marker_to_kinematic_points, rename_markers
from data_analysis.data_processing.read_csv import read_csv
from data_analysis.data_processing.reference_frame import find_reference_frame, transform_points_to_car_RF
from data_analysis.data_processing.steering import extract_steering_angle


def main():
    for file_select in EXPERIMENTS.keys():
        csv_file, json_file = get_experiment(file_select)
        df_raw = read_csv(csv_file)

        # Rename markers and convert to kinematic points
        df_renamed = rename_markers(df_raw, json_file)

        # Convert to kinematic points, keeping the 3 steering wheel markers
        # We keep also the kinematic points that are mapped with only one marker, just for plotting purposes
        # but we add a flag to understand if they are complete or not
        df = marker_to_kinematic_points(df_renamed)

        # find the car reference frame, only with the fixed kinematic points and those with both markers visible (mean over time)
        T = find_reference_frame(df)

        # Transform kinematic points to car RF and save to new CSV. We transform all the kin points, also those with only one marker just for plotting
        # aand we keep the flag to understand if they are complete or not
        # we transform also the steering wheel points
        df_carRF = transform_points_to_car_RF(df, T)

        # Determine steering angle and save to new CSV
        df_carRF = extract_steering_angle(df_carRF, T)
        df_carRF.to_csv(f"data/processed/{file_select}_correctedRF.csv", index=False)

if __name__ == "__main__":
    main()