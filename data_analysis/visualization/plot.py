import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider
from data_analysis.visualization.plot_config import CONNECTIONS
from data_analysis.visualization.utils import _get_point_names, _get_axis_limits, _set_axis_limits, _label_axes, _plot_points
from kinematic_model.Kraken_front_sus_kinematics import rotate


FRONT_MODEL_POINT_NAMES = (
    "P1l_F",
    "P2l_F",
    "P3l_F",
    "P4l_F",
    "P5l_F",
    "P6l_F",
    "P7l_F",
    "P8l_F",
    "P9l_F",
)


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
    triad_length = 0.05 * max(
        upper - lower for lower, upper in axis_limits
    )

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
        for frame_name in ("P9l_F_RF", "P9l_R_RF"):
            status_column = f"{frame_name}_status"
            if (
                frame_name not in row.index
                or status_column not in row.index
                or row[status_column] != "VALID"
            ):
                continue

            reference_frame = np.asarray(row[frame_name], dtype=float)
            if reference_frame.shape != (4, 4) or not np.isfinite(reference_frame).all():
                continue

            origin = reference_frame[:3, 3]
            for axis_index, color in enumerate(("red", "green", "blue")):
                direction = reference_frame[:3, axis_index]
                ax.quiver(
                    *origin,
                    *direction,
                    length=triad_length,
                    normalize=True,
                    arrow_length_ratio=0.2,
                    color=color,
                    linewidth=2,
                )
            ax.text(*origin, frame_name, fontsize=9)

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

def plot_model_comparison_slider(
    df,
    front_kinematics,
    equations,
    model_coordinates,
    wheel_radius=0.2032,
    sample_step=150,
    slider_step=1,
    show=True,
):
    """Solve and plot sampled front-left model and motion-capture points.

    Returns the sampled DataFrame positions, modeled P9 center positions in
    meters, and modeled wheel angles in radians.
    """
    if sample_step < 1:
        raise ValueError("sample_step must be a positive integer.")
    if len(df) == 0:
        raise ValueError(
            "Cannot compare model and motion-capture data: DataFrame is empty."
        )

    frame_positions = np.arange(0, len(df), sample_step)
    if frame_positions[-1] != len(df) - 1:
        frame_positions = np.append(frame_positions, len(df) - 1)

    sampled_df = df.iloc[frame_positions]
    model_positions = np.full((len(sampled_df), 3), np.nan)
    model_angles = np.full((len(sampled_df), 3), np.nan)
    model_points = np.full(
        (len(sampled_df), len(FRONT_MODEL_POINT_NAMES), 3),
        np.nan,
    )
    model_frames = np.full((len(sampled_df), 4, 4), np.nan)
    wheel_center_coordinates = model_coordinates[27:30]
    front_point_coordinates = model_coordinates[3:30]

    for row_number, (_, row) in enumerate(sampled_df.iterrows()):
        inputs = row[["P9l_F_Z", "steering_angle"]].to_numpy(dtype=float)
        if not np.isfinite(inputs).all():
            continue

        solution = front_kinematics.solve_kinematics(
            equations,
            inputs[0] / 1000 - wheel_radius,
            inputs[1] * 180 / np.pi,
        )
        model_positions[row_number] = [
            float(coordinate.subs(solution))
            for coordinate in wheel_center_coordinates
        ]
        model_points[row_number] = np.asarray(
            [
                float(coordinate.subs(solution))
                for coordinate in front_point_coordinates
            ]
        ).reshape(len(FRONT_MODEL_POINT_NAMES), 3)
        model_angles[row_number] = [
            float(solution[front_kinematics.deltaWl]),
            float(solution[front_kinematics.gammaWl]),
            float(solution[front_kinematics.thetaWl]),
        ]
        model_rotation = (
            rotate("Z", model_angles[row_number, 0])
            @ rotate("X", model_angles[row_number, 1])
            @ rotate("Y", model_angles[row_number, 2])
        )
        model_frames[row_number] = np.eye(4)
        model_frames[row_number, :3, :3] = model_rotation[:3, :3]
        model_frames[row_number, :3, 3] = model_positions[row_number] * 1000

    model_points *= 1000

    expected_points_shape = (len(frame_positions), len(FRONT_MODEL_POINT_NAMES), 3)
    if model_points.shape != expected_points_shape:
        raise ValueError(
            f"Expected model_points shape {expected_points_shape}, "
            f"got {model_points.shape}."
        )
    expected_frames_shape = (len(frame_positions), 4, 4)
    if model_frames.shape != expected_frames_shape:
        raise ValueError(
            f"Expected model_frames shape {expected_frames_shape}, "
            f"got {model_frames.shape}."
        )
    if len(frame_positions) == 0:
        raise ValueError("At least one model frame is required.")
    if np.any(frame_positions < 0) or np.any(frame_positions >= len(df)):
        raise IndexError("A frame position is outside the DataFrame.")

    complete_points = np.zeros(
        (len(df), len(FRONT_MODEL_POINT_NAMES)),
        dtype=bool,
    )
    measured_coordinates = []
    for point_index, point_name in enumerate(FRONT_MODEL_POINT_NAMES):
        columns = [f"{point_name}_{axis}" for axis in ("X", "Y", "Z")]
        status_column = f"{point_name}_status"
        if (
            all(column in df.columns for column in columns)
            and status_column in df.columns
        ):
            complete_points[:, point_index] = df[status_column].eq("ALL").to_numpy()
            coordinates = df[columns].to_numpy(dtype=float, copy=True)
            coordinates[~complete_points[:, point_index]] = np.nan
            measured_coordinates.append(coordinates)

    all_coordinates = np.concatenate(
        [*measured_coordinates, model_points.reshape(-1, 3)],
        axis=0,
    )
    valid_coordinates = all_coordinates[
        np.isfinite(all_coordinates).all(axis=1)
    ]
    if len(valid_coordinates) == 0:
        raise ValueError("No valid measured or model point coordinates to plot.")

    lower = valid_coordinates.min(axis=0)
    upper = valid_coordinates.max(axis=0)
    spans = upper - lower
    margins = np.where(spans > 0, spans * 0.05, 1.0)
    axis_limits = tuple(
        (minimum - margin, maximum + margin)
        for minimum, maximum, margin in zip(lower, upper, margins)
    )
    triad_length = max(float(np.max(spans)) * 0.06, 1.0)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_axes([0.08, 0.18, 0.84, 0.74], projection="3d")

    def draw_frame(sample_index):
        ax.clear()
        row_position = frame_positions[sample_index]
        row = df.iloc[row_position]
        model_row = model_points[sample_index]

        for point_index, point_name in enumerate(FRONT_MODEL_POINT_NAMES):
            columns = [f"{point_name}_{axis}" for axis in ("X", "Y", "Z")]
            measured = (
                row[columns].to_numpy(dtype=float)
                if (
                    all(column in row.index for column in columns)
                    and row.get(f"{point_name}_status") == "ALL"
                )
                else np.full(3, np.nan)
            )
            modeled = model_row[point_index]
            measured_valid = np.isfinite(measured).all()
            modeled_valid = np.isfinite(modeled).all()

            if measured_valid:
                ax.scatter(
                    *measured,
                    color="tab:blue",
                    marker="o",
                    s=36,
                )
                ax.text(*measured, point_name, fontsize=8)
            if modeled_valid:
                ax.scatter(
                    *modeled,
                    color="tab:orange",
                    marker="x",
                    s=42,
                )
            if measured_valid and modeled_valid:
                ax.plot(
                    [measured[0], modeled[0]],
                    [measured[1], modeled[1]],
                    [measured[2], modeled[2]],
                    color="0.45",
                    linestyle=":",
                    linewidth=1,
                )

        measured_frame_name = "P9l_F_RF"
        measured_frame = row.get(measured_frame_name)
        measured_status = row.get(f"{measured_frame_name}_status")
        if measured_status == "VALID" and measured_frame is not None:
            measured_frame = np.asarray(measured_frame, dtype=float)
            if (
                measured_frame.shape == (4, 4)
                and np.isfinite(measured_frame).all()
            ):
                _plot_comparison_triad(
                    ax,
                    measured_frame,
                    triad_length,
                    linestyle="-",
                )

        model_frame = model_frames[sample_index]
        if model_frame.shape == (4, 4) and np.isfinite(model_frame).all():
            _plot_comparison_triad(
                ax,
                model_frame,
                triad_length,
                linestyle="--",
            )

        _set_axis_limits(ax, axis_limits)
        _label_axes(ax)
        ax.set_title(f"Frame {df.index[row_position]}: mocap vs modello")
        ax.legend(
            handles=[
                Line2D(
                    [], [], marker="o", linestyle="None",
                    color="tab:blue", label="Motion capture",
                ),
                Line2D(
                    [], [], marker="x", linestyle="None",
                    color="tab:orange", label="Modello cinematico",
                ),
                Line2D(
                    [], [], color="0.45", linestyle=":",
                    label="Scarto tra punti corrispondenti",
                ),
                Line2D(
                    [], [], color="0.25", linestyle="-",
                    label="Frame P9 motion capture",
                ),
                Line2D(
                    [], [], color="0.25", linestyle="--",
                    label="Frame P9 modello",
                ),
            ],
            loc="upper left",
            fontsize=8,
        )

    draw_frame(0)

    slider_ax = fig.add_axes([0.15, 0.07, 0.70, 0.035])
    slider = Slider(
        ax=slider_ax,
        label="Campione",
        valmin=0,
        valmax=len(frame_positions) - 1,
        valinit=0,
        valstep=slider_step,
    )

    def update(_):
        draw_frame(int(slider.val))
        fig.canvas.draw_idle()

    slider.on_changed(update)
    fig._model_comparison_slider = slider
    if show:
        plt.show()

    return frame_positions, model_positions, model_angles

def _plot_comparison_triad(ax, reference_frame, length, linestyle):
    """Draw a scaled reference-frame triad, preserving its axis directions."""
    origin = reference_frame[:3, 3]
    for axis_index, color in enumerate(("red", "green", "blue")):
        direction = reference_frame[:3, axis_index]
        ax.quiver(
            *origin,
            *direction,
            length=length,
            normalize=True,
            arrow_length_ratio=0.18,
            color=color,
            linewidth=2,
            linestyles=linestyle,
        )

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