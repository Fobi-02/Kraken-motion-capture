import numpy as np

def _get_point_names(df):
    """Return points name for which X, Y and Z exist."""
    columns = set(df.columns)

    # Find all unique points by checking for columns ending with "_X" and remove the suffix
    x_points = {
        column[:-2]
        for column in columns
        if column.endswith("_X")
    }

    # Filter points to ensure that corresponding "_Y" and "_Z" columns exist
    points = [
        point
        for point in x_points
        if f"{point}_Y" in columns
        and f"{point}_Z" in columns
    ]

    if not points:
        raise ValueError("No points found in the DataFrame.")

    return points

def _get_axis_limits(df, points, margin=0.05):
    """Calculate axis limits from all valid point positions."""
    limits = []

    # For each axis (X, Y, Z), find the min and max values across all points
    for axis in ("X", "Y", "Z"):
        # Get all values for the current axis across all points, flatten the array, and remove NaN values
        values = df[[f"{point}_{axis}" for point in points]].to_numpy().flatten()
        values = values[~np.isnan(values)]

        if len(values) == 0:
            raise ValueError("DataFrame contains no valid point data.")

        # Calculate the min and max values
        vmin = values.min()
        vmax = values.max()

        # Add a margin to the limits to ensure points are not at the edge of the plot
        extra = margin * (vmax - vmin)
        if extra == 0:
            extra = 1

        limits.append((vmin - extra, vmax + extra))
    return tuple(limits)

def _set_axis_limits(ax, axis_limits):
    """Apply calculated axis limits to a 3D axis."""

    (x_min, x_max), (y_min, y_max), (z_min, z_max) = axis_limits

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

def _label_axes(ax):
    """Configure labels for a 3D plot."""

    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")

def _plot_points(ax, row, points, show_names=True):
    """Plot all valid points from one dataframe row."""

    for point in points:
        coordinates = row[[f"{point}_X", f"{point}_Y", f"{point}_Z"]]

        if coordinates.isna().any():
            continue

        x, y, z = coordinates

        ax.scatter(x, y, z, s=30, color="black")

        if show_names:
            ax.text(x, y, z, point, fontsize=10)

def _get_point(df, index, point_name):
    """Return XYZ coordinates for a kinematic point, or None if unavailable."""

    columns = [f"{point_name}_{axis}" for axis in ("X", "Y", "Z")]

    if not all(column in df.columns for column in columns):
        return None

    point = df.loc[index, columns].to_numpy(dtype=float)

    if np.isnan(point).any():
        return None

    return point