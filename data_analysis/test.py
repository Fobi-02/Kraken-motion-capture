from data_analysis.data_processing.experiment_list import EXPERIMENTS, get_experiment
from data_analysis.data_processing.map_points import marker_to_kinematic_points, rename_markers
from data_analysis.data_processing.read_csv import read_csv
from data_analysis.visualization.plot import plot_markers_slider, plot_markers
from data_analysis.data_processing.reference_frame import find_reference_frame, transform_points_to_car_RF
from data_analysis.data_processing.steering import extract_steering_angle
from data_analysis.data_processing.filtering import filter_markers
import matplotlib.pyplot as plt
import numpy as np

def main():
    file_selected = "F0"
    csv_file, json_file = get_experiment(file_selected)
    df_raw = read_csv(csv_file)
    # df_renamed = rename_markers(df_raw, json_file)
    # df_kin = marker_to_kinematic_points(df_renamed)
    # T, error = find_reference_frame(df_kin, plot=True)
    # print("Mean point error:", error)
    # df_carRF = transform_points_to_car_RF(df_kin, T)
    # df = extract_steering_angle(df_carRF)

    # filtered_df = filter_markers(df_raw, fs=360, cutoff_hz=100, )
    # df_renamed2 = rename_markers(filtered_df, json_file)
    # df_kin2 = marker_to_kinematic_points(df_renamed2)
    # T2, error2 = find_reference_frame(df_kin2, plot=True)
    # print("Mean point error:", error2)
    # df_carRF2 = transform_points_to_car_RF(df_kin2, T2)
    # df2 = extract_steering_angle(df_carRF2)
    
    # plt.figure(figsize=(10, 4))
    # plt.plot(df["steering_angle"]*180/np.pi)
    # plt.plot(df2["steering_angle"]*180/np.pi)
    # plt.grid(True, alpha=0.3)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    fs = 360
    raw_filt = filter_markers(df_raw, fs=fs, cutoff_hz=8)

    # choose columns automatically: most-moving and least-moving X column
    x_cols = [c for c in df_raw.columns if str(c).endswith("_X")]
    std = df_raw[x_cols].std()
    moving_col = std.idxmax()
    static_col = std.idxmin()
    print("moving:", moving_col, "| static:", static_col)

    t = df_raw.index / fs  # seconds

    fig, axs = plt.subplots(2, 2, figsize=(13, 7))

    for row, col in enumerate([moving_col, static_col]):
        axs[row, 0].plot(t, df_raw[col], lw=0.8, alpha=0.6, label="raw")
        axs[row, 0].plot(t, raw_filt[col], lw=1.2, label="filtered")
        axs[row, 0].set_title(f"{col} - full signal")

        sl = slice(int(5 * fs), int(7 * fs))  # 2 s zoom
        axs[row, 1].plot(t[sl], df_raw[col].iloc[sl], lw=0.8, alpha=0.6, label="raw")
        axs[row, 1].plot(t[sl], raw_filt[col].iloc[sl], lw=1.2, label="filtered")
        axs[row, 1].set_title(f"{col} - zoom")

        for ax in axs[row]:
            ax.set_xlabel("time [s]")
            ax.set_ylabel("position [mm]")
            ax.grid(alpha=0.3)
            ax.legend()

    plt.tight_layout()
    plt.show()

    print("static std raw/filtered:", df_raw[static_col].std(), raw_filt[static_col].std())
    print("moving max |raw - filt|:", (df_raw[moving_col] - raw_filt[moving_col]).abs().max())

    return


if __name__ == "__main__":
    main()