import numpy as np
import json
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from kinematic_model.Kraken_front_sus_kinematics import translate, rotate

def transformation_matrix(params):
    x, y, z, theta_x, theta_y, theta_z = params
    return (rotate("X", theta_x) @ rotate("Y", theta_y) @ rotate("Z", theta_z) @ translate(x, y, z))


def find_reference_frame(df):
    # loading the ideal points
    with open("data/marker_maps/model_positions.json", "r") as f:
        ideal = json.load(f)
    ideal_points = np.array([
        ideal["P1l_F"],
        ideal["P2l_F"],
        ideal["P3l_F"],
        ideal["P4l_F"],
        ideal["P1l_R"],
        ideal["P2l_R"],
        ideal["P3l_R"],
        ideal["P4l_R"]
        ])

    # loading the measured points
    measured_points = np.array([
        [df["P1l_F_X"][0],df["P1l_F_Y"][0],df["P1l_F_Z"][0],1],
        [df["P2l_F_X"][0],df["P2l_F_Y"][0],df["P2l_F_Z"][0],1],
        [df["P3l_F_X"][0],df["P3l_F_Y"][0],df["P3l_F_Z"][0],1],
        [df["P4l_F_X"][0],df["P4l_F_Y"][0],df["P4l_F_Z"][0],1],
        [df["P1l_R_X"][0],df["P1l_R_Y"][0],df["P1l_R_Z"][0],1],
        [df["P2l_R_X"][0],df["P2l_R_Y"][0],df["P2l_R_Z"][0],1],
        [df["P3l_R_X"][0],df["P3l_R_Y"][0],df["P3l_R_Z"][0],1],
        [df["P4l_R_X"][0],df["P4l_R_Y"][0],df["P4l_R_Z"][0],1]
        ])

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
    if np.isfinite(df["P1l_F_X"][0]):
        x0[0] = ideal["P1l_F"][0] - df["P1l_F_X"][0]
        x0[1] = ideal["P1l_F"][1] - df["P1l_F_Y"][0]
        x0[2] = ideal["P1l_F"][2] - df["P1l_F_Z"][0]
    elif np.isfinite(df["P1l_R_X"][0]):
        x0[0] = ideal["P1l_R"][0] - df["P1l_R_X"][0]
        x0[1] = ideal["P1l_R"][1] - df["P1l_R_Y"][0]
        x0[2] = ideal["P1l_R"][2] - df["P1l_R_Z"][0]

    # Optimization
    result = minimize(objective, x0, args=(measured_points, ideal_points), method="BFGS")

    # apply result
    T_opt = transformation_matrix(result.x)
    aligned_points = (T_opt @ measured_points.T).T

    # -----------------------------
    # Plot 3D
    # -----------------------------
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

    return T_opt


