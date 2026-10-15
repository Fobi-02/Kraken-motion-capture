from data_analysis.data_processing.experiment_list import EXPERIMENTS, get_experiment
from data_analysis.data_processing.map_points import marker_to_kinematic_points, rename_markers
from data_analysis.data_processing.read_csv import read_csv
from data_analysis.visualization.plot import plot_markers_slider, plot_markers
from data_analysis.data_processing.reference_frame import find_reference_frame, transform_points_to_car_RF
from data_analysis.data_processing.steering import extract_steering_angle
import matplotlib.pyplot as plt
import numpy as np

def main():
    file_selected = "SW"
    csv_file, json_file = get_experiment(file_selected)
    df_raw = read_csv(csv_file)
    df_renamed = rename_markers(df_raw, json_file)
    df_kin = marker_to_kinematic_points(df_renamed)
    T, error = find_reference_frame(df_kin, plot=False)
    df_carRF = transform_points_to_car_RF(df_kin, T)
    df= extract_steering_angle(df_carRF)

    plt.figure(figsize=(10, 4))
    plt.plot(df["steering_angle"]*180/np.pi)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return


if __name__ == "__main__":
    main()