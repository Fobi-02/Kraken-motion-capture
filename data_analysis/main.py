import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data_analysis.data_processing.experiment_list import EXPERIMENTS, get_experiment, get_processed_experiment
from data_analysis.data_processing.read_csv import read_processed_csv, read_csv
from data_analysis.visualization.plot import plot_markers_slider, plot_markers


def main():
    '''
    file = "SW"
    csv, map = get_experiment(file)
    df = read_csv(csv)
    plot_markers(df, 0)
    return
    '''

    file_select = "SW"
    csv_file = get_processed_experiment(file_select)
    df = read_processed_csv(csv_file)

    # Some statistics and reports on marker availability and point quality
    #report_point_quality(df)
    plot_markers_slider(df, step=100)

    # Plot links for a specific frame
    #plot_links(df, 0)

    # plot the steering angle
    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df["steering_angle"], color="tab:blue", linewidth=1.5)
    plt.xlabel("Frame")
    plt.ylabel("Steering angle")
    plt.title("Steering angle over time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()