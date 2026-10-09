from pathlib import Path

import numpy as np
import pandas as pd

from kinematic_model.Kraken_front_sus_kinematics import get_point, translate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

AXES = ("X", "Y", "Z")
WHEEL_CENTER_OFFSET = 18.16

FRONT_P9_REFERENCE_MARKER = "p25"
FRONT_P9_X_MARKERS = ("p26", "p27")

REAR_P9_REFERENCE_MARKER = "p49"
REAR_P9_X_MARKERS = ("p50", "p51")

def _marker_is_valid(row, marker):
    """Return True if a marker has three finite XYZ coordinates."""
    columns = [f"{marker}_{axis}" for axis in AXES]

    if not all(column in row.index for column in columns):
        return False

    coordinates = row[columns].to_numpy(dtype=float)
    return np.isfinite(coordinates).all()


def _get_marker_position(row, marker):
    """Return a marker's XYZ coordinates, or None if unavailable."""
    if not _marker_is_valid(row, marker):
        return None

    columns = [f"{marker}_{axis}" for axis in AXES]
    return row[columns].to_numpy(dtype=float)



def _calculate_wheel_reference_frame(
    wheel_center,
    z_reference,
    x_reference_start,
    x_reference_end,
):
    """
    Calculate a wheel reference frame.

    Returns a 4x4 homogeneous transformation matrix, or None if the
    available geometry cannot define a valid frame.
    """
    nz = z_reference - wheel_center
    nx = x_reference_end - x_reference_start

    norm_nz = np.linalg.norm(nz)
    norm_nx = np.linalg.norm(nx)

    if (
        not np.isfinite(norm_nz)
        or not np.isfinite(norm_nx)
        or norm_nz < 1e-12
        or norm_nx < 1e-12
    ):
        return None

    nz = nz / norm_nz
    nx = nx / norm_nx

    cross = np.cross(nz, nx)
    norm_ny = np.linalg.norm(cross)

    # Parallel directions cannot define a valid reference frame.
    if not np.isfinite(norm_ny) or norm_ny < 1e-12:
        return None

    ny = cross / norm_ny

    # Re-orthogonalize X to ensure an orthonormal frame.
    nx = np.cross(ny, nz)
    nx /= np.linalg.norm(nx)

    transformation = np.array([
        [nx[0], ny[0], nz[0], wheel_center[0]],
        [nx[1], ny[1], nz[1], wheel_center[1]],
        [nx[2], ny[2], nz[2], wheel_center[2]],
        [0.0,   0.0,   0.0,   1.0],
    ])

    return transformation @ translate(
        0,
        -WHEEL_CENTER_OFFSET,
        0,
    )


def _update_wheel_center(df_new, index, wheel_name, reference_frame):
    """Update wheel-center coordinates and store its reference frame."""
    point = np.asarray(get_point(reference_frame), dtype=float).reshape(-1)

    if len(point) < 3 or not np.isfinite(point[:3]).all():
        return False

    for axis, value in zip(AXES, point[:3]):
        df_new.at[index, f"{wheel_name}_{axis}"] = float(value)

    df_new.at[index, f"{wheel_name}_RF"] = reference_frame
    df_new.at[index, f"{wheel_name}_RF_status"] = "VALID"

    return True


def _calculate_p9_wheel_frames(df, df_new):
    """
    Calculate front and rear P9 wheel centers and reference frames.

    The input df contains the renamed marker coordinates. The wheel-center
    coordinates in df_new are initially estimated by _average_marker_group.
    When the required reference markers define a valid frame, the center
    is updated using the existing wheel-offset transformation.
    """
    wheel_configs = {
        "P9l_F": (
            FRONT_P9_REFERENCE_MARKER,
            FRONT_P9_X_MARKERS,
        ),
        "P9l_R": (
            REAR_P9_REFERENCE_MARKER,
            REAR_P9_X_MARKERS,
        ),
    }

    for point_name in wheel_configs:
        df_new[f"{point_name}_RF"] = pd.Series(
            None,
            index=df.index,
            dtype=object,
        )
        df_new[f"{point_name}_RF_status"] = pd.Series(
            "MISSING",
            index=df.index,
            dtype="string",
        )

    for index in df.index:
        row = df.loc[index]

        for point_name, (reference_marker, x_markers) in wheel_configs.items():
            center_columns = [
                f"{point_name}_{axis}" for axis in AXES
            ]
            center = df_new.loc[index, center_columns].to_numpy(dtype=float)

            if not np.isfinite(center).all():
                continue

            required_markers = [reference_marker, *x_markers]
            positions = {
                marker: _get_marker_position(row, marker)
                for marker in required_markers
            }

            if any(position is None for position in positions.values()):
                continue

            reference_frame = _calculate_wheel_reference_frame(
                center,
                positions[reference_marker],
                positions[x_markers[0]],
                positions[x_markers[1]],
            )

            if reference_frame is None:
                continue

            _update_wheel_center(
                df_new,
                index,
                point_name,
                reference_frame,
            )