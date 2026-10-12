import numpy as np
import json
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from kinematic_model.Kraken_front_sus_kinematics import translate, rotate

def transformation_matrix(params):
    x, y, z, theta_x, theta_y, theta_z = params
    return (rotate("X", theta_x) @ rotate("Y", theta_y) @ rotate("Z", theta_z) @ translate(x, y, z))


def find_reference_frame(df, plot=False):
    # loading the ideal points
    with open("data/marker_maps/model_positions.json", "r") as f:
        ideal = json.load(f)
    point_names = [
        "P1l_F", "P2l_F", "P3l_F", "P4l_F",
        "P1l_R", "P2l_R", "P3l_R", "P4l_R",
    ]
    measured_by_name = {}
    measured_points = []
    ideal_points = []

    for point_name in point_names:
        status_column = f"{point_name}_status"
        coordinate_columns = [f"{point_name}_{axis}" for axis in "XYZ"]
        if status_column not in df.columns:
            continue

        all_rows = df[status_column] == "ALL"
        if not all_rows.any():
            continue

        mean_position = df.loc[all_rows, coordinate_columns].mean().to_numpy(dtype=float)
        measured_by_name[point_name] = mean_position
        measured_points.append([*mean_position, 1])
        ideal_points.append(ideal[point_name])

    if not measured_points:
        raise ValueError("No reference points have status 'ALL'.")

    measured_points = np.array(measured_points)
    ideal_points = np.array(ideal_points)

    # cost function to minimize
    def objective(params, measured, ideal):
        valid = np.isfinite(measured[:, :3]).all(axis=1)
        measured_valid = measured[valid]
        ideal_valid = ideal[valid]
        T = transformation_matrix(params)
        transformed = (T @ measured_valid.T).T
        diff = ideal_valid[:, :3] - transformed[:, :3]
        return np.sum(np.sum(diff**2, axis=1))

    # Initial guess
    x0 = np.zeros(6)
    initial_point = next(
        (measured_by_name[name] for name in ("P1l_F", "P1l_R") if name in measured_by_name),
        None,
    )
    if initial_point is not None:
        initial_name = "P1l_F" if "P1l_F" in measured_by_name else "P1l_R"
        x0[:3] = np.array(ideal[initial_name]) - initial_point

    # Optimization
    result = minimize(objective, x0, args=(measured_points, ideal_points), method="BFGS")

    # apply result
    T_opt = transformation_matrix(result.x)
    error = result.fun
    aligned_points = (T_opt @ measured_points.T).T

    # -----------------------------
    # Plot 3D
    # -----------------------------
    if(plot):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            ideal_points[:, 0], ideal_points[:, 1], ideal_points[:, 2],
            s=80, label="Ideal points"
        )
        ax.scatter(
            aligned_points[:, 0], aligned_points[:, 1], aligned_points[:, 2],
            s=80, label="Aligned measured points"
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        plt.show()

    return T_opt, error


def transform_points_to_car_RF(df, T):
    """ Transform kinematic points to car RF. We transform all the kin points, also those with only one marker just for plotting
        and we keep the flag to understand if they are complete or not
        and we transform also the steering wheel points and the reference frame of P9"""
    df_transformed = df.copy()
    for point in df.columns:
        if point.endswith(("_X", "_Y", "_Z")):
            point_name = point[:-2]
            coords = np.array([df[point_name + "_X"], df[point_name + "_Y"], df[point_name + "_Z"], np.ones(len(df))])
            transformed_coords = (T @ coords).T
            df_transformed[point_name + "_X"] = transformed_coords[:, 0]
            df_transformed[point_name + "_Y"] = transformed_coords[:, 1]
            df_transformed[point_name + "_Z"] = transformed_coords[:, 2]

    for frame_column in ("P9l_F_RF", "P9l_R_RF"):
        if frame_column not in df.columns:
            continue
        status_column = f"{frame_column}_status"
        for index, reference_frame in df[frame_column].items():
            if reference_frame is None:
                continue
            if status_column in df.columns and df.at[index, status_column] != "VALID":
                continue
            df_transformed.at[index, frame_column] = T @ reference_frame

    return df_transformed
