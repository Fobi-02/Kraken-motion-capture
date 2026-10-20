import numpy as np
import matplotlib.pyplot as plt

from data_analysis.data_processing.experiment_list import get_processed_experiment
from data_analysis.data_processing.read_csv import read_processed_csv
from data_analysis.visualization.plot import plot_markers_slider, plot_links, plot_model_comparison_slider
from kinematic_model.Kraken_rear_sus_kinematics import RearKinematics


def main(file_select="R2"):
    tie_rod_positions = {
        "R0": "center",
        "R2": "front",
        "R-2": "rear",
    }
    if file_select not in tie_rod_positions:
        raise ValueError(
            f"Unsupported rear experiment {file_select!r}; "
            f"choose one of {tuple(tie_rod_positions)}."
        )
    csv_file = get_processed_experiment(file_select)
    df = read_processed_csv(csv_file)

    # Some statistics and reports on marker availability and point quality
    #plot_markers_slider(df, step=100)

    # Plot links for a specific frame
    #plot_links(df, 0)

    rWheel = 0.2032
    # static toe and camber (rad)
    # for R0 use delta0=-2.5*np.pi/180 and gamma0=0.35*np.pi/180
    # for R-2 use delta0=1*np.pi/180 and gamma0=0.35*np.pi/180
    # for R2 use delta0=-4*np.pi/180 and gamma0=0.35*np.pi/180
    delta0 = -4*np.pi/180
    gamma0 = 0.35*np.pi/180
    rear_kine = RearKinematics(gamma0, delta0)
    equations, points = rear_kine.set_kinematic_problem(
        pos=tie_rod_positions[file_select]
    )

    # ---------------------------------------------
    # Compare 3D coordinates between model and measurements
    # ---------------------------------------------

    sample_positions = np.arange(0, len(df), 50)
    if sample_positions[-1] != len(df) - 1:
        sample_positions = np.append(sample_positions, len(df) - 1)
    sampled_df = df.iloc[sample_positions]
    model_positions = np.full((len(sampled_df), 3), np.nan)
    model_angles = np.full((len(sampled_df), 3), np.nan)
    wheel_center_coordinates = points[27:30]

    for sample_index, (_, row) in enumerate(sampled_df.iterrows()):
        wheel_z = float(row["P9l_R_Z"]) / 1000 - rWheel
        if not np.isfinite(wheel_z):
            continue

        solution = rear_kine.solve_kinematics(equations, wheel_z)
        model_positions[sample_index] = [
            float(coordinate.subs(solution))
            for coordinate in wheel_center_coordinates
        ]
        model_angles[sample_index] = [
            delta0 + float(solution[rear_kine.deltaWl]),
            -gamma0 + float(solution[rear_kine.gammaWl]),
            float(solution[rear_kine.thetaWl]),
        ]

    position_columns = ("P9l_R_X", "P9l_R_Y", "P9l_R_Z")
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
    # Suspension link length
    # ---------------------------------------------

    p1 = df[["P1l_R_X", "P1l_R_Y", "P1l_R_Z"]].to_numpy(dtype=float)
    p2 = df[["P2l_R_X", "P2l_R_Y", "P2l_R_Z"]].to_numpy(dtype=float)
    p3 = df[["P3l_R_X", "P3l_R_Y", "P3l_R_Z"]].to_numpy(dtype=float)
    p4 = df[["P4l_R_X", "P4l_R_Y", "P4l_R_Z"]].to_numpy(dtype=float)
    p6 = df[["P6l_R_X", "P6l_R_Y", "P6l_R_Z"]].to_numpy(dtype=float)
    p7 = df[["P7l_R_X", "P7l_R_Y", "P7l_R_Z"]].to_numpy(dtype=float)

    d_p1_p7 = np.linalg.norm(p1 - p7, axis=1)
    d_p2_p7 = np.linalg.norm(p2 - p7, axis=1)
    d_p3_p6 = np.linalg.norm(p3 - p6, axis=1)
    d_p4_p6 = np.linalg.norm(p4 - p6, axis=1)

    plt.figure(figsize=(10, 4))
    plt.plot(df.index, d_p1_p7, color="tab:purple", linewidth=1.5, label="P1-P7")
    plt.plot(df.index, d_p2_p7, color="tab:orange", linewidth=1.5, label="P2-P7")
    plt.plot(df.index, d_p3_p6, color="tab:green", linewidth=1.5, label="P3-P6")
    plt.plot(df.index, d_p4_p6, color="tab:blue", linewidth=1.5, label="P4-P6")
    plt.xlabel("Frame")
    plt.ylabel("Length (mm)")
    plt.title("Suspension link length")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # ---------------------------------------------
    # Compare measured and modelled wheel angles.
    # ---------------------------------------------

    angle_columns = ("delta_R", "gamma_R", "theta_R")
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

    measured_z_wheel = df["P9l_R_Z"].to_numpy(dtype=float) / 1000 - rWheel
    measured_gamma = df["gamma_R"].to_numpy(dtype=float)
    valid_measurements = np.isfinite(measured_z_wheel) & np.isfinite(measured_gamma)
    if "P9l_R_RF_status" in df.columns:
        valid_measurements &= df["P9l_R_RF_status"].eq("VALID").to_numpy()

    model_z_wheel = (
        sampled_df["P9l_R_Z"].to_numpy(dtype=float) / 1000 - rWheel
    )
    valid_model = np.isfinite(model_z_wheel) & np.isfinite(model_angles[:, 1])
    model_order = np.argsort(model_z_wheel[valid_model])

    plt.figure(figsize=(8, 5))
    plt.scatter(
        measured_z_wheel[valid_measurements] * 1000,
        np.degrees(measured_gamma[valid_measurements]),
        s=8,
        alpha=0.45,
        label="Measured",
    )
    plt.plot(
        model_z_wheel[valid_model][model_order] * 1000,
        np.degrees(model_angles[valid_model, 1][model_order]),
        color="tab:red",
        marker="o",
        markersize=4,
        label="Kinematic model",
    )
    plt.xlabel("z ruota (mm)")
    plt.ylabel("Camber angle γ (deg)")
    plt.title(f"Camber gain - {file_select}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()