import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import median_filter

from data_analysis.data_processing.read_csv import read_csv
from data_analysis.data_processing.experiment_list import get_experiment

def filter_markers(
    df,
    fs=360.0,
    cutoff_hz=10.0,
    order=4,
    despike=True,
    spike_window=10,
    spike_n_mad=8.0,
    spike_floor_mm=0.1,
    max_gap=10,
    suffixes=("_X", "_Y", "_Z"),
):
    """0-phase LP filter. Some details:

    - Only columns whose name ends with one of `suffixes` are filtered, every other column untouched. Sampling rate and index are unchanged;
    - 4th order Butterworth, applied in zero-phase (back and forth, sosfiltfilt) to avoid phase distortion and delay;
    - The gaps with NaN, shorter than `max_gap`, are linearly interpolated before filtering;
    - Long gaps stays NaN;
    - Despiking to remove isolated glitches that are due to occlusions and reflections. 
        Samples that deviate from local median of 10 frames, by more than 8·MAD + 0.1 mm, are replaced by the median
    - Runs too short to be filtered are left as they are.
    """
    # Check Nyquist
    if cutoff_hz >= fs / 2:
        raise ValueError("cutoff_hz must be below the Nyquist frequency")

    # Design the filter, using second-order sections (sos) for numerical stability
    sos = signal.butter(order, cutoff_hz, btype="low", fs=fs, output="sos")
    # Define minimum length to be filtered with sosfiltfilt, based on filter order and number of sections
    min_len = 3 * (2 * sos.shape[0] + 1) + 1
    # A pad of 120 samples (~0.33 s at 360 Hz) is used to ensure that the filter has enough data to operate on, especially at the edges of the segment
    pad = 120

    # Filter each column ending with the specified suffixes
    out = df.copy()
    cols = [c for c in df.columns if str(c).endswith(tuple(suffixes))]

    # Process each column individually
    for col in cols:
        # Convert to float and identify valid (non-NaN) samples
        s = df[col].astype(float)
        valid = s.notna().values
        # If there are no valid samples, skip this column
        if not valid.any():
            continue

        # Interpolate short gaps and mark long gaps as NaN
        x = s.interpolate(method="linear", limit_area="inside").to_numpy(copy=True)
        gap_edges = np.flatnonzero(np.diff(np.r_[0, (~valid).astype(int), 0]))

        # Remove gaps longer than max_gap by setting them to NaN
        for g0, g1 in zip(gap_edges[::2], gap_edges[1::2]):
            if g1 - g0 > max_gap:
                x[g0:g1] = np.nan
        # Identify valid samples after interpolation and gap removal
        ok = ~np.isnan(x)

        # Identify contiguous segments of valid samples
        edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
        y = x.copy()

        # Process each contiguous segment of valid samples
        for start, stop in zip(edges[::2], edges[1::2]):
            # Extract the segment and check its length
            seg = x[start:stop]
            if len(seg) < min_len:
                continue

            # Apply despiking if enabled
            if despike:
                med = median_filter(seg, size=spike_window, mode="nearest")
                mad = median_filter(np.abs(seg - med), size=spike_window, mode="nearest") * 1.4826
                seg = np.where(np.abs(seg - med) > spike_n_mad * mad + spike_floor_mm, med, seg)

            # Apply the filter to the segment, using padding to handle edge effects
            y[start:stop] = signal.sosfiltfilt(sos, seg, padtype="odd", padlen=min(pad, len(seg) - 1))

        out[col] = y

    return out

def _amplitude_spectrum(x, fs):
    """Single-sided amplitude spectrum [mm] of a signal (mean removed, Hann window)."""
    x = pd.Series(x).interpolate(limit_direction="both").to_numpy()
    x = x - x.mean()
    w = np.hanning(len(x))
    amp = 2 * np.abs(np.fft.rfft(x * w)) / w.sum()
    freqs = np.fft.rfftfreq(len(x), d=1 / fs)
    return freqs, amp


def fourier(df_raw, df_filt, fs=360.0, cutoff_hz=10.0, suffixes=("_X", "_Y", "_Z")):
    """Plot the amplitude spectrum before/after filtering for a moving and a static marker."""
    import matplotlib.pyplot as plt

    cols = [c for c in df_raw.columns if str(c).endswith(tuple(suffixes))]
    std = df_raw[cols].std()
    examples = {"moving marker": std.idxmax(), "static marker": std.idxmin()}        

    fig, axs = plt.subplots(2, 1, figsize=(8, 8))
    for ax, (label, col) in zip(axs, examples.items()):
        f, a_raw = _amplitude_spectrum(df_raw[col].to_numpy(float), fs)
        _, a_filt = _amplitude_spectrum(df_filt[col].to_numpy(float), fs)

        # Frequency below which 99% / 99.9% of the raw signal energy lies
        cum = np.cumsum(a_raw**2) / np.sum(a_raw**2)
        f90, f99, f999 = f[np.searchsorted(cum, 0.90)], f[np.searchsorted(cum, 0.99)], f[np.searchsorted(cum, 0.999)]
        print(f"{label} ({col}): 90% energy < {f90:.2f} Hz, 99% energy < {f99:.2f} Hz, 99.9% energy < {f999:.2f} Hz")

        ax.semilogx(f[1:], a_raw[1:], lw=1.5, alpha=0.7, label="Raw")
        ax.semilogx(f[1:], a_filt[1:], lw=2.0, label="Filtered")
        ax.axvline(cutoff_hz, color="k", ls="--", lw=1, label=f"Cutoff {cutoff_hz:g} Hz")
        ax.set_xlim(f[1], fs / 2)
        ax.set_xlabel("frequency [Hz]")
        ax.set_ylabel("amplitude [mm]")
        ax.set_title(f"{label}: {col}")
        ax.grid(alpha=0.3, which="both")
        ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import sys

    # Experiment name from the command line, e.g. `python filter_markers.py SW`
    experiment = sys.argv[1] if len(sys.argv) > 1 else "SW"
    cutoff_hz = 10.0

    csv_path, _ = get_experiment(experiment)   # marker map not needed here
    df_raw = read_csv(csv_path)

    fs = 1 / df_raw["Time"].diff().median()
    print(f"{experiment}: sampling rate {fs:.1f} Hz")

    df_filt = filter_markers(df_raw, fs=fs, cutoff_hz=cutoff_hz)
    fourier(df_raw, df_filt, fs=fs, cutoff_hz=cutoff_hz)