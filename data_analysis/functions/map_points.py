import json
import pandas as pd

def rename_markers(df, json_path):
    '''
    function to rename the markers based on the map in the json file
    '''
    with open(json_path, "r") as f:
        marker_map = json.load(f)

    new_columns = {}
    for column in df.columns:
        for old_name, new_name in marker_map.items():
            if column.startswith(old_name + "_"):
                suffix = column[len(old_name):]
                new_columns[column] = new_name + suffix
                break

    # keeping only the columns mapped on the map
    df_new = df[list(new_columns.keys())].copy()

    # renaming the columns
    df_new = df_new.rename(columns=new_columns)

    return df_new

def marker_to_kinematic_points(df):
    '''
    Function that finds the kinematic points averaging the position of the markers (if found)
    '''
    with open("../data/marker_maps/marker_to_kinematic_points.json", "r") as f:
        marker_groups = json.load(f)

    df_new = pd.DataFrame(index=df.index)
    for new_marker, markers in marker_groups.items():
        for axis in ["X", "Y", "Z"]:
            columns = [f"{marker}_{axis}" for marker in markers]
            # If one of the involved column does not exist i reutrn NaN
            if not all(column in df.columns for column in columns):
                df_new[f"{new_marker}_{axis}"] = float("nan")
                continue

            # only if all values are present i can compute the mean
            df_new[f"{new_marker}_{axis}"] = df[columns].mean(axis=1, skipna=False)

    return df_new



