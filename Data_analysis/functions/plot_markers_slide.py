import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pandas as pd
import numpy as np


def plot_markers_slider(df, step=1):
    # reading markers
    markers = []
    for column in df.columns:
        if column.endswith("_X"):
            marker_name = column[:-2]

            if (f"{marker_name}_Y" in df.columns
                and f"{marker_name}_Z" in df.columns):
                markers.append(marker_name)

    if not markers:
        raise ValueError("Nessun marker trovato nel DataFrame.")

    # axis limits
    x_values = []
    y_values = []
    z_values = []

    for marker in markers:
        x_values.extend(df[f"{marker}_X"].dropna().values)
        y_values.extend(df[f"{marker}_Y"].dropna().values)
        z_values.extend(df[f"{marker}_Z"].dropna().values)

    if len(x_values) == 0:
        raise ValueError("Il DataFrame non contiene dati validi.")

    x_min, x_max = np.min(x_values), np.max(x_values)
    y_min, y_max = np.min(y_values), np.max(y_values)
    z_min, z_max = np.min(z_values), np.max(z_values)

    # additional margin
    def add_margin(vmin, vmax):
        margin = 0.05 * (vmax - vmin)
        if margin == 0:
            margin = 1

        return vmin - margin, vmax + margin

    x_min, x_max = add_margin(x_min, x_max)
    y_min, y_max = add_margin(y_min, y_max)
    z_min, z_max = add_margin(z_min, z_max)

    # plot
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_axes([0.10, 0.15, 0.80, 0.75], projection="3d")
    def draw_frame(frame_number):
        ax.clear()
        row = df.iloc[frame_number]

        for marker in markers:
            x = row[f"{marker}_X"]
            y = row[f"{marker}_Y"]
            z = row[f"{marker}_Z"]

            if not any(pd.isna([x, y, z])):
                ax.scatter(x, y, z, s=30, color='black')

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)

        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Y [mm]")
        ax.set_zlabel("Z [mm]")

        ax.set_title(f"Frame {frame_number} / {len(df) - 1}")

    # first frame
    draw_frame(0)
    # slider
    slider_ax = fig.add_axes([0.15, 0.05, 0.70, 0.03])
    slider = Slider(
        ax=slider_ax,
        label="Frame",
        valmin=0,
        valmax=len(df) - 1,
        valinit=0,
        valstep=step
    )

    def update(val):
        frame_number = int(slider.val)
        draw_frame(frame_number)
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()