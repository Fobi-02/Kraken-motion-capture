import json
from pathlib import Path

import numpy as np
import pandas as pd

from kinematic_model.Kraken_front_sus_kinematics import get_point, translate


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

WHEEL_CENTER_OFFSET = 18.16

FRONT_P9_REFERENCE_MARKER = "p25"
FRONT_P9_X_MARKERS = ("p26", "p27")

REAR_P9_REFERENCE_MARKER = "p49"
REAR_P9_X_MARKERS = ("p50", "p51")

def rename_markers(df, json_path):
    """
    function to rename the markers based on the map in the json file
    """

    with open(json_path, "r") as f:
        marker_map = json.load(f)
    # groups columns
    mapped_columns = {}
    for column in df.columns:
        for old_name, new_name in marker_map.items():
            if column.startswith(old_name + "_"):
                suffix = column[len(old_name):]
                mapped_columns.setdefault(new_name, {})
                mapped_columns[new_name].setdefault(suffix, [])
                mapped_columns[new_name][suffix].append(column)
                break

    result_columns = {}

    for new_name, suffixes in mapped_columns.items():
        for suffix, columns in suffixes.items():
            new_column = new_name + suffix
            if len(columns) == 1:
                result_columns[new_column] = df[columns[0]]
            else:
                # selecting the first not NaN value of columns with the same name
                result_columns[new_column] = (
                    df[columns]
                    .bfill(axis=1)
                    .iloc[:, 0]
                )

    # creating the new data frame
    result = pd.DataFrame(result_columns, index=df.index)

    return result

def _calculate_wheel_reference_frame(
    wheel_center,
    z_reference,
    x_reference_start,
    x_reference_end,
):
    """
    Calculate the wheel reference frame.

    Parameters
    ----------
    wheel_center:
        Current wheel-center position.

    z_reference:
        Marker defining the local Z direction.

    x_reference_start:
        First marker defining the local X direction.

    x_reference_end:
        Second marker defining the local X direction.

    Returns
    -------
    numpy.ndarray
        4x4 homogeneous transformation matrix.
    """

    nz = z_reference - wheel_center
    nz = nz / np.linalg.norm(nz)

    nx = x_reference_end - x_reference_start
    nx = nx / np.linalg.norm(nx)

    ny = np.cross(nz, nx)

    transformation = np.array([
        [nx[0], ny[0], nz[0], wheel_center[0]],
        [nx[1], ny[1], nz[1], wheel_center[1]],
        [nx[2], ny[2], nz[2], wheel_center[2]],
        [0.0,  0.0,   0.0,   1.0],
    ])

    return transformation @ translate(
        0,
        -WHEEL_CENTER_OFFSET,
        0,
    )

def _update_wheel_center(
    df_new,
    index,
    wheel_name,
    reference_frame,
):
    """Update wheel-center coordinates and store its reference frame."""

    point = get_point(reference_frame)

    df_new.loc[index, f"{wheel_name}_X"] = float(point[0])
    df_new.loc[index, f"{wheel_name}_Y"] = float(point[1])
    df_new.loc[index, f"{wheel_name}_Z"] = float(point[2])

    df_new.at[index, f"{wheel_name}_RF"] = reference_frame

def _average_marker_group(df, markers):
    """
    Calculate a kinematic point from a group of markers.

    Rules:
    - all available markers are averaged;
    - if only one marker is available, that marker is used;
    - if no marker is available, the result is NaN.

    Returns
    -------
    position : pandas.DataFrame
        X/Y/Z coordinates of the kinematic point.

    status : pandas.Series
        "averaged", "single_marker", or "missing".

    available_markers : list[str]
        Markers that actually exist in the DataFrame.
    """

    axes = ["X", "Y", "Z"]

    available_markers = []

    for marker in markers:
        columns = [f"{marker}_{axis}" for axis in axes]

        if all(column in df.columns for column in columns):
            available_markers.append(marker)

    result = pd.DataFrame(
        np.nan,
        index=df.index,
        columns=axes,
    )

    status = pd.Series(
        "missing",
        index=df.index,
        dtype="string",
    )

    if not available_markers:
        return result, status, available_markers

    marker_data = [
        df[
            [f"{marker}_{axis}" for axis in axes]
        ].to_numpy(dtype=float)
        for marker in available_markers
    ]

    data = np.stack(marker_data, axis=1)

    valid = ~np.isnan(data).any(axis=2)

    for i in range(len(df)):
        valid_points = data[i][valid[i]]

        if len(valid_points) == 0:
            continue

        if len(valid_points) == 1:
            result.iloc[i] = valid_points[0]
            status.iloc[i] = "single_marker"
        else:
            result.iloc[i] = valid_points.mean(axis=0)
            status.iloc[i] = "averaged"

    return result, status, available_markers

def marker_to_kinematic_points(df):
    """
    Convert marker positions into kinematic points.

    For normal kinematic points:
    - all available markers in a frame -> average them
    - exactly one available marker -> use that marker
    - no available markers -> NaN

    P9 is treated specially because, after calculating its position,
    we also need additional markers to construct its wheel reference frame.
    """

    with open(
        DATA_DIR / "marker_maps" / "marker_to_kinematic_points.json",
        "r",
    ) as f:
        marker_groups = json.load(f)

    df_new = pd.DataFrame(index=df.index)

    # Wheel reference frames
    df_new["P9l_F_RF"] = pd.Series(index=df.index, dtype=object)
    df_new["P9l_R_RF"] = pd.Series(index=df.index, dtype=object)

    # Status of each calculated kinematic point
    point_status = {}

    # -------------------------------------------------------------------------
    # Calculate all normal kinematic points
    # -------------------------------------------------------------------------

    for new_marker, markers in marker_groups.items():

        position, status, available_markers = _average_marker_group(
            df,
            markers,
        )

        for axis in ["X", "Y", "Z"]:
            df_new[f"{new_marker}_{axis}"] = position[axis]

        point_status[new_marker] = {
            "status": status,
            "available_markers": available_markers,
            "expected_markers": markers,
        }

    # -------------------------------------------------------------------------
    # Adjust P9 positions and calculate wheel reference frames
    # -------------------------------------------------------------------------

    def marker_is_valid(row, marker):
        """Return True if all XYZ coordinates of a marker are valid."""

        columns = [
            f"{marker}_X",
            f"{marker}_Y",
            f"{marker}_Z",
        ]

        if not all(column in row.index for column in columns):
            return False

        return not row[columns].isna().any()

    def get_marker_position(row, marker):
        """Return a marker position as a NumPy array."""

        return np.array([
            row[f"{marker}_X"],
            row[f"{marker}_Y"],
            row[f"{marker}_Z"],
        ], dtype=float)

    for index in df.index:

        row = df.loc[index]
        row_new = df_new.loc[index]

        # =====================================================================
        # FRONT P9
        # =====================================================================

        if not pd.isna(row_new["P9l_F_X"]):

            required_markers = [
                FRONT_P9_REFERENCE_MARKER,
                FRONT_P9_X_MARKERS[0],
                FRONT_P9_X_MARKERS[1],
            ]

            if all(
                marker_is_valid(row, marker)
                for marker in required_markers
            ):

                wheel_center = np.array([
                    row_new["P9l_F_X"],
                    row_new["P9l_F_Y"],
                    row_new["P9l_F_Z"],
                ])

                z_reference = get_marker_position(
                    row,
                    FRONT_P9_REFERENCE_MARKER,
                )

                x_reference_start = get_marker_position(
                    row,
                    FRONT_P9_X_MARKERS[0],
                )

                x_reference_end = get_marker_position(
                    row,
                    FRONT_P9_X_MARKERS[1],
                )

                rf_p9 = _calculate_wheel_reference_frame(
                    wheel_center,
                    z_reference,
                    x_reference_start,
                    x_reference_end,
                )

                _update_wheel_center(
                    df_new,
                    index,
                    "P9l_F",
                    rf_p9,
                )

        # =====================================================================
        # REAR P9
        # =====================================================================

        if not pd.isna(row_new["P9l_R_X"]):

            required_markers = [
                REAR_P9_REFERENCE_MARKER,
                REAR_P9_X_MARKERS[0],
                REAR_P9_X_MARKERS[1],
            ]

            if all(
                marker_is_valid(row, marker)
                for marker in required_markers
            ):

                wheel_center = np.array([
                    row_new["P9l_R_X"],
                    row_new["P9l_R_Y"],
                    row_new["P9l_R_Z"],
                ])

                z_reference = get_marker_position(
                    row,
                    REAR_P9_REFERENCE_MARKER,
                )

                x_reference_start = get_marker_position(
                    row,
                    REAR_P9_X_MARKERS[0],
                )

                x_reference_end = get_marker_position(
                    row,
                    REAR_P9_X_MARKERS[1],
                )

                rf_p9 = _calculate_wheel_reference_frame(
                    wheel_center,
                    z_reference,
                    x_reference_start,
                    x_reference_end,
                )

                _update_wheel_center(
                    df_new,
                    index,
                    "P9l_R",
                    rf_p9,
                )

    df_new.attrs["point_status"] = point_status

    return df_new

def report_point_quality(df):
    """
    Print a summary of how each kinematic point was calculated.
    """

    point_status = df.attrs.get("point_status")

    if point_status is None:
        raise ValueError(
            "No point-status information found in DataFrame."
        )

    print("\nKinematic point quality:")
    print("-" * 80)

    for point, information in point_status.items():

        status = information["status"]
        expected = information["expected_markers"]
        available = information["available_markers"]

        counts = status.value_counts()

        averaged = counts.get("averaged", 0)
        single = counts.get("single_marker", 0)
        missing = counts.get("missing", 0)

        print(f"{point:15s}")
        print(f"  expected:  {expected}")
        print(f"  available: {available}")
        print(
            f"  frames: averaged={averaged:5d}, "
            f"single={single:5d}, "
            f"missing={missing:5d}"
        )

def report_marker_availability(df, marker_groups):
    """
    Report whether each expected marker exists in the dataset and,
    if it exists, how many frames have valid coordinates.
    """

    markers = sorted({
        marker
        for group in marker_groups.values()
        for marker in group
    })

    print("\nMarker availability:")
    print("-" * 80)

    for marker in markers:
        columns = [
            f"{marker}_X",
            f"{marker}_Y",
            f"{marker}_Z",
        ]

        # Marker is not present at all in the DataFrame
        if not all(column in df.columns for column in columns):
            print(f"{marker:5s} NOT IN DATASET")
            continue

        # Marker exists, so check frame-by-frame availability
        valid = ~df[columns].isna().any(axis=1)

        present = valid.sum()
        missing = (~valid).sum()

        print(
            f"{marker:5s} present: {present:5d}/{len(df)} "
            f"missing: {missing:5d}"
        )
