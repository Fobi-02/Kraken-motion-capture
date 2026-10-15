import numpy as np
import matplotlib.pyplot as plt

from data_analysis.data_processing.experiment_list import get_processed_experiment
from data_analysis.data_processing.read_csv import read_processed_csv
from data_analysis.visualization.plot import plot_markers_slider, plot_model_comparison_slider
from kinematic_model.Kraken_front_sus_kinematics import FrontKinematics


def main():
    file_select = "F-30"
    csv_file = get_processed_experiment(file_select)
    df = read_processed_csv(csv_file)

    print("Mean steering angle: ", 180/np.pi*np.mean(df["steering_angle"]))

    # Some statistics and reports on marker availability and point quality
    #plot_markers_slider(df, step=100)

    # Plot links for a specific frame
    #plot_links(df, 0)

    rWheel = 0.2032
    front_kine = FrontKinematics()
    equations, points = front_kine.set_kinematic_problem()

    # ---------------------------------------------
    # Compare 3D coordinates between model and measurements
    # ---------------------------------------------

    sample_positions, model_positions, model_angles = plot_model_comparison_slider(
        df,
        front_kine,
        equations,
        points,
        wheel_radius=rWheel,
        sample_step=150,
        show=False,
    )
    sampled_df = df.iloc[sample_positions]

    position_columns = ("P9l_F_X", "P9l_F_Y", "P9l_F_Z")
    position_labels = ("X", "Y", "Z")
    position_colors = ("tab:red", "tab:green", "tab:blue")

    # ----------------------------------------------------
    # Compare measured and modelled wheel center positions.
    # ----------------------------------------------------

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

    # ---------------------------------------------
    # Compare measured and modelled wheel angles.
    # ---------------------------------------------

    angle_columns = ("delta_F", "gamma_F", "theta_F")
    angle_labels = ("delta", "gamma", "theta")
    plt.figure(figsize=(10, 4))
    for coordinate, label, color in zip(
        range(3), angle_labels, position_colors
    ):
        plt.plot(
            df.index,
            180/np.pi*df[angle_columns[coordinate]],
            color=color,
            linewidth=1.5,
            label=f"{label} measured",
        )
        plt.plot(
            sampled_df.index,
            180/np.pi*model_angles[:, coordinate],
            color=color,
            linewidth=1.5,
            linestyle="--",
            label=f"{label} model",
        )
    plt.xlabel("Frame")
    plt.ylabel("Angle (deg)")
    plt.title("Measured and modelled wheel angles")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # ---------------------------------------------
    # Compare camber angle against the wheel-travel
    # ---------------------------------------------

    measured_z_wheel = df["P9l_F_Z"].to_numpy(dtype=float) / 1000 - rWheel
    measured_gamma = df["gamma_F"].to_numpy(dtype=float)
    valid_measurements = np.isfinite(measured_z_wheel) & np.isfinite(measured_gamma)
    if "P9l_F_RF_status" in df.columns:
        valid_measurements &= df["P9l_F_RF_status"].eq("VALID").to_numpy()

    model_z_wheel = (
        sampled_df["P9l_F_Z"].to_numpy(dtype=float) / 1000 - rWheel
    )
    valid_model = np.isfinite(model_z_wheel) & np.isfinite(model_angles[:, 1])
    model_order = np.argsort(model_z_wheel[valid_model])

    plt.figure(figsize=(8, 5))
    plt.scatter(
        measured_z_wheel[valid_measurements] * 1000,
        np.degrees(measured_gamma[valid_measurements]),
        s=8,
        alpha=0.45,
        label="Misurato",
    )
    plt.plot(
        model_z_wheel[valid_model][model_order] * 1000,
        np.degrees(model_angles[valid_model, 1][model_order]),
        color="tab:red",
        marker="o",
        markersize=4,
        label="Modello cinematico",
    )
    plt.xlabel("z ruota (mm)")
    plt.ylabel("Angolo di camber γ (deg)")
    plt.title(f"Camber gain - {file_select}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()