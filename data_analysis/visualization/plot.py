import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from data_analysis.visualization.plot_config import CONNECTIONS
from data_analysis.visualization.utils import _get_point_names, _get_axis_limits, _set_axis_limits, _label_axes, _plot_points

def plot_markers(df, frame_number):
    """Plot marker positions for a single frame."""
    # Defining the markers and axis limits
    markers = _get_point_names(df)
    axis_limits = _get_axis_limits(df, markers)

    # Getting the row corresponding to the specified frame number
    row = df.iloc[frame_number]

    # Creating a 3D plot
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111, projection="3d")
    _plot_points(ax, row, markers)
    _set_axis_limits(ax, axis_limits)
    _label_axes(ax)
    plt.show()

def plot_markers_slider(df, step=1):
    """Display kinematic points and their connections with a frame slider."""
    # Defining the names and axis limits
    points = _get_point_names(df)
    axis_limits = _get_axis_limits(df, points)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_axes(
        [0.10, 0.15, 0.80, 0.75],
        projection="3d",
    )

    # Function to draw a specific frame
    def draw_frame(frame_number):
        ax.clear()
        row = df.iloc[frame_number]

        _plot_points(ax, row, points)
        #_plot_connections(ax, row, frame_number)
        _set_axis_limits(ax, axis_limits)
        _label_axes(ax)
        ax.set_title(f"Frame {frame_number} / {len(df) - 1}")

    draw_frame(0)

    slider_ax = fig.add_axes([0.15, 0.05, 0.70, 0.03])

    slider = Slider(
        ax=slider_ax,
        label="Frame",
        valmin=0,
        valmax=len(df) - 1,
        valinit=0,
        valstep=step,
    )

    def update(_):
        draw_frame(int(slider.val))
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

def plot_links(df, frame_number):
    """Plot suspension links and markers for a single frame."""

    row = df.iloc[frame_number]

    points = _get_point_names(df)
    axis_limits = _get_axis_limits(df, points)

    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111, projection="3d")

    _plot_points(ax, row, points, show_names=False)
    _plot_connections(ax, df, frame_number)

    _set_axis_limits(ax, axis_limits)
    _label_axes(ax)

    plt.show()

def _plot_connections(ax, df, frame_number):
    """Plot the kinematic-point connections for one frame."""

    row = df.iloc[frame_number]

    for color, links in CONNECTIONS:
        for point_a, point_b in links:

            columns = [
                f"{point_a}_X",
                f"{point_a}_Y",
                f"{point_a}_Z",
                f"{point_b}_X",
                f"{point_b}_Y",
                f"{point_b}_Z",
            ]

            # Don't draw incomplete links.
            if any(pd.isna(row[column]) for column in columns):
                continue

            ax.plot(
                [row[f"{point_a}_X"], row[f"{point_b}_X"]],
                [row[f"{point_a}_Y"], row[f"{point_b}_Y"]],
                [row[f"{point_a}_Z"], row[f"{point_b}_Z"]],
                linewidth=2,
                color=color,
            )