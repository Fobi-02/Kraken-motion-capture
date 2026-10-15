from pathlib import Path
from data_analysis.data_processing.reference_frame import RF_decomposition
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

WHEEL_CENTER_OFFSET = 18.16

FRONT_P9_REFERENCE_MARKER = "p25"
FRONT_P9_X_MARKERS = ("p26", "p27")

REAR_P9_REFERENCE_MARKER = "p49"
REAR_P9_X_MARKERS = ("p50", "p51")


def steering_angle(P4, P5, P6):
    '''
    Given the three steering markers it computes the steering angle
    !!! All the markers have to be in the correct car RF before using this function !!!
    '''
    # central point of the three markers
    Ps = np.mean([P4, P5, P6], axis=0)

    # steering wheel reference frame
    nz = (P4 - Ps) / np.linalg.norm(P4 - Ps)
    ny = (P5 - P6) / np.linalg.norm(P5 - P6)
    nx = np.cross(ny, nz) / np.linalg.norm(np.cross(ny, nz))
    RFsteering = np.array([
        [nx[0], ny[0], nz[0], Ps[0]],
        [nx[1], ny[1], nz[1], Ps[1]],
        [nx[2], ny[2], nz[2], Ps[2]],
        [0,     0,     0,     1    ]
    ])

    # Decomposes R = Ry(angle_y) @ Rz(angle_z) @ Rx(angle_x), in radians.
    rotation = RFsteering[:3, :3]
    angle_z = np.arcsin(np.clip(rotation[1, 0], -1.0, 1.0))
    cos_angle_z = np.cos(angle_z)

    if abs(cos_angle_z) > 1e-8:
        angle_y = np.arctan2(-rotation[2, 0], rotation[0, 0])
        angle_x = np.arctan2(-rotation[1, 2], rotation[1, 1])
    else:
        angle_y = np.arctan2(rotation[0, 2], rotation[2, 2])
        angle_x = 0.0

    return -angle_x


def extract_steering_angle(df):
    '''
    Given a dataframe with the kinematic points in the car RF, it computes the steering angle for each frame
    and adds it to the dataframe as a new column
    '''
    df = df.copy()
    df["steering_angle"] = np.nan

    for idx, row in df.iterrows():
        P4 = np.array([row["p04_X"], row["p04_Y"], row["p04_Z"]])
        P5 = np.array([row["p05_X"], row["p05_Y"], row["p05_Z"]])
        P6 = np.array([row["p06_X"], row["p06_Y"], row["p06_Z"]])

        if np.any(np.isnan(P4)) or np.any(np.isnan(P5)) or np.any(np.isnan(P6)):
            continue

        # Compute steering angle
        delta = steering_angle(P4, P5, P6)
        df.at[idx, "steering_angle"] = delta

    return df