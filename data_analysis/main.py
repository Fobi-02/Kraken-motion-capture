import numpy as np
import matplotlib.pyplot as plt

from data_analysis.data_processing.experiment_list import get_processed_experiment
from data_analysis.data_processing.read_csv import read_processed_csv
from data_analysis.visualization.plot import plot_markers_slider
from kinematic_model.Kraken_front_sus_kinematics import FrontKinematics


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
    #plot_markers_slider(df, step=100)

    # Plot links for a specific frame
    #plot_links(df, 0)

    rWheel = 0.2032
    front_kine = FrontKinematics()
    equations, points = front_kine.set_kinematic_problem()

    sample_step = 50
    sample_positions = np.arange(0, len(df), sample_step)
    if len(df) and sample_positions[-1] != len(df) - 1:
        sample_positions = np.append(sample_positions, len(df) - 1)
    sampled_df = df.iloc[sample_positions]

    model_positions = np.full((len(sampled_df), 3), np.nan)
    model_angles = np.full((len(sampled_df), 3), np.nan)
    p9l_model = points[27:30]

    for row_number, (_, row) in enumerate(sampled_df.iterrows()):
        inputs = row[["P9l_F_Z", "steering_angle"]].to_numpy(dtype=float)
        if not np.isfinite(inputs).all():
            continue

        solution = front_kine.solve_kinematics(
            equations,
            inputs[0] / 1000 - rWheel,
            -inputs[1],
        )
        model_positions[row_number] = [
            float(coordinate.subs(solution)) for coordinate in p9l_model
        ]
        model_angles[row_number] = [
            float(solution[front_kine.deltaWl]),
            float(solution[front_kine.gammaWl]),
            float(solution[front_kine.thetaWl]),
        ]

    position_columns = ("P9l_F_X", "P9l_F_Y", "P9l_F_Z")
    position_labels = ("X", "Y", "Z")
    position_colors = ("tab:red", "tab:green", "tab:blue")

    # Compare measured and modelled wheel center positions.
    plt.figure(figsize=(10, 4))
    for coordinate, label, color in zip(
        range(3), position_labels, position_colors
    ):
        plt.plot(
            df.index,
            df[position_columns[coordinate]] / 1000,
            color=color,
            linewidth=1.5,
            label=f"{label} measured",
        )
        plt.plot(
            sampled_df.index,
            model_positions[:, coordinate],
            color=color,
            linewidth=1.5,
            linestyle="--",
            label=f"{label} model",
        )
    plt.xlabel("Frame")
    plt.ylabel("Position (m)")
    plt.title("Measured and modelled wheel center position")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Compare measured and modelled wheel angles.
    angle_columns = ("delta_F", "gamma_F", "theta_F")
    angle_labels = ("delta", "gamma", "theta")
    plt.figure(figsize=(10, 4))
    for coordinate, label, color in zip(
        range(3), angle_labels, position_colors
    ):
        plt.plot(
            df.index,
            df[angle_columns[coordinate]],
            color=color,
            linewidth=1.5,
            label=f"{label} measured",
        )
        plt.plot(
            sampled_df.index,
            model_angles[:, coordinate],
            color=color,
            linewidth=1.5,
            linestyle="--",
            label=f"{label} model",
        )
    plt.xlabel("Frame")
    plt.ylabel("Angle (rad)")
    plt.title("Measured and modelled wheel angles")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()