import json
from pathlib import Path

import numpy as np
import pandas as pd

from kinematic_model.Kraken_front_sus_kinematics import get_point, translate
from data_analysis.data_processing.wheel import _calculate_p9_wheel_frames


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

AXES = ("X", "Y", "Z")
WHEEL_CENTER_OFFSET = 18.16

FRONT_P9_REFERENCE_MARKER = "p25"
FRONT_P9_X_MARKERS = ("p26", "p27")

REAR_P9_REFERENCE_MARKER = "p49"
REAR_P9_X_MARKERS = ("p50", "p51")


# ----------------------------------- UTILITIES -----------------------------------
def rename_markers(df, json_path):
    """Rename raw marker columns using the JSON mapping."""
    with open(json_path, encoding="utf-8") as file:
        marker_map = json.load(file)
    mapped_columns = {}

    # Loop through each column in the DataFrame and check if it starts with any of the old marker names.
    # Then loop through the suffixes to find the corresponding new marker name and create a new column with the new name.
    # If multiple old marker names map to the same new marker name, we will use the first non-missing value when columns map together.
    for column in df.columns:
        for old_name, new_name in marker_map.items():
            # Check if the column starts with the old marker name and has a suffix (e.g., "_X", "_Y", "_Z")
            if column.startswith(f"{old_name}_"):
                # Extract the suffix (e.g., "_X", "_Y", "_Z") from the column name.
                suffix = column[len(old_name):]
                # Initialize the new marker name in the mapped_columns dictionary if it doesn't exist.
                mapped_columns.setdefault(new_name, {})
                mapped_columns[new_name].setdefault(suffix, [])
                mapped_columns[new_name][suffix].append(column)
                break

    result_columns = {}

    # Loop through the mapped columns and create new columns in the result DataFrame
    for new_name, suffixes in mapped_columns.items():
        for suffix, columns in suffixes.items():
            new_column = f"{new_name}{suffix}"

            if len(columns) == 1:
                result_columns[new_column] = df[columns[0]]
            else:
                # Use the first non-missing value when columns map together.
                result_columns[new_column] = df[columns].bfill(axis=1).iloc[:, 0]

    return pd.DataFrame(result_columns, index=df.index)


# ------------------------------- AVERAGE KINEMATIC POINTS -----------------------------------

def _average_marker_group(df, markers):
    """
    Calculate a kinematic point from its contributing markers.

    Status for each frame:
    - ALL: two or more markers contribute to the average.
    - SINGLE: exactly one marker contributes.
    - MISSING: no markers contribute.
    """
    available_markers = [
        marker
        for marker in markers
        if all(f"{marker}_{axis}" in df.columns for axis in AXES)
    ]

    status = pd.Series("MISSING", index=df.index, dtype="string")

    if not available_markers:
        position = pd.DataFrame(
            np.nan, index=df.index, columns=AXES, dtype=float
        )
        return position, status, available_markers

    # Shape: (frames, markers, coordinates).
    data = np.stack(
        [
            df[[f"{marker}_{axis}" for axis in AXES]].to_numpy(dtype=float)
            for marker in available_markers
        ],
        axis=1,
    )

    # A marker contributes only if all XYZ coordinates are finite.
    valid = np.isfinite(data).all(axis=2)
    count = valid.sum(axis=1)

    # Sum only valid marker coordinates.
    sums = np.where(valid[:, :, None], data, 0.0).sum(axis=1)

    # Use a separate, writable NumPy array for the division output.
    averaged = np.full(sums.shape, np.nan, dtype=float)

    np.divide(
        sums,
        count[:, None],
        out=averaged,
        where=count[:, None] > 0,
    )

    position = pd.DataFrame(
        averaged,
        index=df.index,
        columns=AXES,
    )

    status.iloc[count == 1] = "SINGLE"
    status.iloc[count >= 2] = "ALL"

    return position, status, available_markers


# ------------------------------- MAIN FUNCTION -----------------------------------
def marker_to_kinematic_points(df):
    """
    Convert renamed marker coordinates into kinematic points.

    Each point has XYZ coordinate columns and a status column.

    Normal points:
    - ALL: two or more markers contribute to the average.
    - SINGLE: exactly one marker contributes.
    - MISSING: no markers contribute.

    Steering:
    - Keep the three markers individually, without averaging.

    P9:
    - Calculate wheel centers and reference frames when possible.

    The returned DataFrame stores statuses directly in columns.
    A summary of expected and available markers is also stored in attrs.
    """
    mapping_path = DATA_DIR / "marker_maps" / "marker_to_kinematic_points.json"

    with open(mapping_path, encoding="utf-8") as file:
        marker_groups = json.load(file)

    df_new = pd.DataFrame(index=df.index)
    point_status = {}

    # for each group, get the name of the group and the list of markers
    for point_name, markers in marker_groups.items():

        # Steering markers are individual points, not an averaged group.
        if point_name == "Steering":
            for marker in markers:
                columns = [f"{marker}_{axis}" for axis in AXES]

                # If the marker columns exist in df, copy them to df_new; otherwise, fill with NaN.
                for column in columns:
                    if column in df.columns:
                        df_new[column] = df[column].to_numpy(dtype=float)
                    else:
                        df_new[column] = np.nan

                # Store the status for each frame: ALL if the marker is valid, MISSING otherwise.
                valid = np.isfinite(df_new[columns].to_numpy(dtype=float)).all(axis=1)
                status = pd.Series(
                    np.where(valid, "ALL", "MISSING"),
                    index=df.index,
                    dtype="string",
                )
                df_new[f"{marker}_status"] = status

                point_status[marker] = {
                    "status": status,
                    "expected_markers": [marker],
                    "available_markers": (
                        [marker] if all(c in df.columns for c in columns) else []
                    ),
                }

            continue

        # Calculate normal kinematic points, including the initial P9 centers.
        position, status, available_markers = _average_marker_group(df, markers)

        for axis in AXES:
            df_new[f"{point_name}_{axis}"] = position[axis]

        df_new[f"{point_name}_status"] = status

        point_status[point_name] = {
            "status": status,
            "expected_markers": markers,
            "available_markers": available_markers,
        }

    # Update P9 wheel centers and calculate their reference frames.
    _calculate_p9_wheel_frames(df, df_new)

    df_new.attrs["point_status"] = point_status

    return df_new


# ------------------------------- REPORTING -----------------------------------
def report_point_quality(df):
    """Print per-point counts for ALL, SINGLE, and MISSING frames."""
    point_status = df.attrs.get("point_status")

    if point_status is None:
        # Also support DataFrames loaded from a file, where attrs are lost.
        point_names = [
            column[:-len("_status")]
            for column in df.columns
            if column.endswith("_status")
            and not column.endswith("_RF_status")
        ]

        if not point_names:
            raise ValueError("No point-status columns found in DataFrame.")

        point_status = {
            point: {
                "status": df[f"{point}_status"],
                "expected_markers": [],
                "available_markers": [],
            }
            for point in point_names
        }

    print("\nKinematic point quality:")
    print("-" * 80)

    for point, information in point_status.items():
        status = information["status"]
        counts = status.value_counts()

        print(f"{point:15s}")
        print(f"  expected:  {information['expected_markers']}")
        print(f"  available: {information['available_markers']}")
        print(
            f"  frames: ALL={counts.get('ALL', 0):5d}, "
            f"SINGLE={counts.get('SINGLE', 0):5d}, "
            f"MISSING={counts.get('MISSING', 0):5d}"
        )

    # Report the independent P9 reference-frame status.
    for point_name in ("P9l_F", "P9l_R"):
        column = f"{point_name}_RF_status"

        if column in df.columns:
            counts = df[column].value_counts()
            print(
                f"{point_name} reference frames: "
                f"VALID={counts.get('VALID', 0):5d}, "
                f"MISSING={counts.get('MISSING', 0):5d}"
            )


