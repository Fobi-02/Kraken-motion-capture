import json
import pandas as pd
import sys
import os
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)
from kinematic_model.Kraken_front_sus_kinematics import *

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

    # --------------------------------------------------------------------------------
    # adjusting the position of P9l_F e P9l_R to make them in the correct wheel center
    # --------------------------------------------------------------------------------

    d = 18.16 # distance to adjust (mm)
    # elaborating rows one by one
    for i in range(df_new.shape[0]):
        row = df.iloc[i]
        row_new = df_new.iloc[i]

        # FRONT P9
        if not pd.isna(row_new[f"P9l_F_X"]):
            # z axis unit vector
            nz = np.array([row[f"p25_X"]-row_new[f"P9l_F_X"], row[f"p25_Y"]-row_new[f"P9l_F_Y"], row[f"p25_Z"]-row_new[f"P9l_F_Z"]])
            nz = nz / np.linalg.norm(nz)
            # x axis unit vector
            nx = np.array([row[f"p27_X"]-row[f"p26_X"], row[f"p27_Y"]-row[f"p26_Y"], row[f"p27_Z"]-row[f"p26_Z"]])
            nx = nx / np.linalg.norm(nx)
            # y axis unit vector
            ny = np.cross(nz, nx)
            # transformation matrix
            T = np.array([[nx[0],ny[0],nz[0],row_new[f"P9l_F_X"]],
                        [nx[1],ny[1],nz[1],row_new[f"P9l_F_Y"]],
                        [nx[2],ny[2],nz[2],row_new[f"P9l_F_Z"]],
                        [0,0,0,1]])
            # finding the new (and correct) position for P9
            P9 = get_point(T @ translate(0, -d, 0))
            # saving the values
            df_new.loc[df_new.index[i], "P9l_F_X"] = float(P9[0])
            df_new.loc[df_new.index[i], "P9l_F_Y"] = float(P9[1])
            df_new.loc[df_new.index[i], "P9l_F_Z"] = float(P9[2])

        # REAR P9
        if not pd.isna(row_new[f"P9l_R_X"]):
            print("prova")
            # z axis unit vector
            nz = np.array([row[f"p49_X"]-row_new[f"P9l_R_X"], row[f"p49_Y"]-row_new[f"P9l_R_Y"], row[f"p49_Z"]-row_new[f"P9l_R_Z"]])
            nz = nz / np.linalg.norm(nz)
            # x axis unit vector
            nx = np.array([row[f"p51_X"]-row[f"p50_X"], row[f"p51_Y"]-row[f"p50_Y"], row[f"p51_Z"]-row[f"p50_Z"]])
            nx = nx / np.linalg.norm(nx)
            # y axis unit vector
            ny = np.cross(nz, nx)
            # transformation matrix
            T = np.array([[nx[0],ny[0],nz[0],row_new[f"P9l_R_X"]],
                        [nx[1],ny[1],nz[1],row_new[f"P9l_R_Y"]],
                        [nx[2],ny[2],nz[2],row_new[f"P9l_R_Z"]],
                        [0,0,0,1]])
            # finding the new (and correct) position for P9
            P9 = get_point(T @ translate(0, -d, 0))
            # saving the values
            df_new.loc[df_new.index[i], "P9l_R_X"] = float(P9[0])
            df_new.loc[df_new.index[i], "P9l_R_Y"] = float(P9[1])
            df_new.loc[df_new.index[i], "P9l_R_Z"] = float(P9[2])




    return df_new



