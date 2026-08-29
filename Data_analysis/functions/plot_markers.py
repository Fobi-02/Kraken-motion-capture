import matplotlib.pyplot as plt
import pandas as pd

def plot_markers(df, frame_number):
    row = df.iloc[frame_number]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    x_min, x_max, y_min, y_max, z_min, z_max = 0, 0, 0, 0, 0, 0

    for column in df.columns:
        if column.endswith("_X"):
            marker_name = column[:-2]
            x = row[f"{marker_name}_X"]
            y = row[f"{marker_name}_Y"]
            z = row[f"{marker_name}_Z"]

            if not any(pd.isna([x, y, z])):
                ax.scatter(x, y, z, s=30, color='0')

            # finding axis limit
            if x > x_max: x_max = x
            if x < x_min: x_min = x
            if y > y_max: y_max = y
            if y < y_min: y_min = y
            if z > z_max: z_max = z
            if z < z_min: z_min = z

    # Auto axis limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")
    plt.show()