from pathlib import Path

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
    nx = np.cross(ny, nz)
    RFsteering = np.array([
        [nx[0], ny[0], nz[0], Ps[0]],
        [nx[1], ny[1], nz[1], Ps[1]],
        [nx[2], ny[2], nz[2], Ps[2]],
        [0,     0,     0,     1    ]
    ])

    # steering wheel static reference frame (at delta=0°)
    nxs = nx
    nys = np.array([0, 1, 0])
    nzs = np.cross(nxs, nys)
    RFstatic = np.array([
            [nxs[0], nys[0], nzs[0], Ps[0]],
            [nxs[1], nys[1], nzs[1], Ps[1]],
            [nxs[2], nys[2], nzs[2], Ps[2]],
            [0,      0,      0,      1    ]
        ])

    # computing steering angle by solving the equation {RFstatic @ rotate("X", delta) == RFsteering} for delta
    cos_delta = ( RFstatic[0, 1]*RFsteering[0, 1] + RFstatic[0, 2]*RFsteering[0, 2]) / ( RFstatic[0, 1]**2 + RFstatic[0, 2]**2)
    sin_delta = (RFstatic[0, 2]*RFsteering[0, 1] -  RFstatic[0, 1]*RFsteering[0, 2]) / ( RFstatic[0, 1]**2 + RFstatic[0, 2]**2)
    steering_angle = np.arctan2(sin_delta, cos_delta) * 180 / np.pi
    
    return steering_angle


def extract_steering_angle(df, T):
    '''
    Given a dataframe with the kinematic points in the car RF, it computes the steering angle for each frame
    and adds it to the dataframe as a new column
    '''
    df = df.copy()
    df["steering_angle"] = np.nan

    for i, row in df.iterrows():
        P4 = np.array([row["SW_P4_X"], row["SW_P4_Y"], row["SW_P4_Z"]])
        P5 = np.array([row["SW_P5_X"], row["SW_P5_Y"], row["SW_P5_Z"]])
        P6 = np.array([row["SW_P6_X"], row["SW_P6_Y"], row["SW_P6_Z"]])

        if np.any(np.isnan(P4)) or np.any(np.isnan(P5)) or np.any(np.isnan(P6)):
            continue

        # Transform points to car RF
        P4_carRF = T @ np.append(P4, 1)
        P5_carRF = T @ np.append(P5, 1)
        P6_carRF = T @ np.append(P6, 1)

        # Compute steering angle
        angle = steering_angle(P4_carRF[:3], P5_carRF[:3], P6_carRF[:3])
        df.at[i, "steering_angle"] = angle

    return df