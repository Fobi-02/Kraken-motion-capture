import numpy as np
import matplotlib.pyplot as plt

from data_analysis.data_processing.experiment_list import get_processed_experiment
from data_analysis.data_processing.read_csv import read_processed_csv
from data_analysis.visualization.plot import plot_markers_slider, plot_model_comparison_slider, plot_links
from kinematic_model.Kraken_front_sus_kinematics import FrontKinematics


def main():
    # selecting the file to study
    file_select="SW"
    csv_file = get_processed_experiment(file_select)
    df = read_processed_csv(csv_file)

    #plot_markers_slider(df, step=100)

    # Plot links for a specific frame
    #plot_links(df, 0)

    # wheel radius in m
    rWheel = 0.2032
    # static toe and camber (rad)
    delta0 = -2*np.pi/180
    gamma0 = -0.5*np.pi/180

    # setting up the kinematic problem
    front_kine = FrontKinematics(gamma0, delta0)
    equations, points = front_kine.set_kinematic_problem()

    if file_select != "SW":
        print("Mean steering angle: ", 180/np.pi*np.mean(df["steering_angle"]))
    
    # -----------------------------------------------------
    # Compare 3D coordinates between model and measurements
    # -----------------------------------------------------

    # plotting the points in 3D and computing the model position equivalent to the given wheel travel and wheel heigth
    sample_positions, model_positions, model_angles = plot_model_comparison_slider(
        df,
        front_kine,
        equations,
        points,
        delta0,
        gamma0,
        wheel_radius=rWheel,
        sample_step=50
    )
    sampled_df = df.iloc[sample_positions]

    position_columns = ("P9l_F_X", "P9l_F_Y", "P9l_F_Z")
    position_labels = ("X", "Y", "Z")
    position_colors = ("tab:red", "tab:green", "tab:blue")

    # --------------------------------------------------------------
    # Compare measured and modelled wheel center positions over time
    # --------------------------------------------------------------

    plt.figure(figsize=(10, 4))
    for coordinate, label, color in zip(range(3), position_labels, position_colors):
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

    # -------------------------------
    # Measured suspension link length
    # -------------------------------

    # kinematic points
    p1 = df[["P1l_F_X", "P1l_F_Y", "P1l_F_Z"]].to_numpy(dtype=float)
    p2 = df[["P2l_F_X", "P2l_F_Y", "P2l_F_Z"]].to_numpy(dtype=float)
    p3 = df[["P3l_F_X", "P3l_F_Y", "P3l_F_Z"]].to_numpy(dtype=float)
    p4 = df[["P4l_F_X", "P4l_F_Y", "P4l_F_Z"]].to_numpy(dtype=float)
    p6 = df[["P6l_F_X", "P6l_F_Y", "P6l_F_Z"]].to_numpy(dtype=float)
    p7 = df[["P7l_F_X", "P7l_F_Y", "P7l_F_Z"]].to_numpy(dtype=float)

    # length
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

    # ------------------------------------------
    # Compare measured and modelled wheel angles
    # ------------------------------------------

    angle_columns = ("delta_F", "gamma_F", "theta_F")
    angle_labels = ("delta", "gamma", "theta")
    plt.figure(figsize=(10, 4))
    for coordinate, label, color in zip(range(3), angle_labels, position_colors):
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

    if file_select != "SW":

        measured_z_wheel = df["P9l_F_Z"].to_numpy(dtype=float) / 1000 - rWheel
        measured_gamma = df["gamma_F"].to_numpy(dtype=float)
        valid_measurements = np.isfinite(measured_z_wheel) & np.isfinite(measured_gamma)
        if "P9l_F_RF_status" in df.columns:
            valid_measurements &= df["P9l_F_RF_status"].eq("VALID").to_numpy()

        model_z_wheel = (sampled_df["P9l_F_Z"].to_numpy(dtype=float) / 1000 - rWheel)
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
        plt.xlabel("z wheel (mm)")
        plt.ylabel("Camber angle (deg)")
        plt.title(f"Camber gain - {file_select}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

    # -------------------------------------
    # Compare steering behavior (Ackermann)
    # -------------------------------------

    if file_select == "SW":

        steering_wheel_angle = np.degrees(df["steering_angle"].to_numpy(dtype=float))
        wheel_angle = np.degrees(df["delta_F"].to_numpy(dtype=float))

        valid_angles = (
            np.isfinite(steering_wheel_angle)
            & np.isfinite(wheel_angle)
            & (wheel_angle != 0)
        )
        model_steering_wheel_angle = np.degrees(
            sampled_df["steering_angle"].to_numpy(dtype=float) - delta0
        )
        model_wheel_angle = np.degrees(model_angles[:, 0])
        valid_model_angles = (
            np.isfinite(model_steering_wheel_angle)
            & np.isfinite(model_wheel_angle)
        )
        model_order = np.argsort(
            model_steering_wheel_angle[valid_model_angles]
        )

        plt.figure(figsize=(8, 5))
        plt.scatter(
            steering_wheel_angle[valid_angles],
            wheel_angle[valid_angles],
            s=10,
            alpha=0.5,
            label="Measured",
        )
        plt.plot(
            model_steering_wheel_angle[valid_model_angles][model_order],
            model_wheel_angle[valid_model_angles][model_order],
            color="tab:red",
            marker="o",
            markersize=4,
            label="Kinematic model",
        )
        plt.xlabel("Steering angle (deg)")
        plt.ylabel("Wheel angle (deg)")
        plt.title("Ackermann behavior")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

    plt.show()



if __name__ == "__main__":
    main()