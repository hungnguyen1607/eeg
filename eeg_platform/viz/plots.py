# plotting functions for eeg

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_timeseries(
    time: np.ndarray,
    data: np.ndarray,
    ch_names: list[str],
    out_path: str | Path,
    max_seconds: float = 10.0,
    scale_uv: float | None = None,
) -> None:
    # plots stacked eeg timeseries
    n_channels, n_samples = data.shape

    # Limit to max_seconds
    max_idx = np.searchsorted(time, time[0] + max_seconds)
    max_idx = min(max_idx, n_samples)
    time_plot = time[:max_idx]
    data_plot = data[:, :max_idx]

    # Auto-scale using robust percentile if not provided
    if scale_uv is None:
        # Use 1st and 99th percentile for robust scaling
        p01 = np.percentile(data_plot, 1)
        p99 = np.percentile(data_plot, 99)
        scale_uv = max(abs(p01), abs(p99)) * 2.5
        if scale_uv == 0:
            scale_uv = 1.0  # Fallback for flat signals

    # Create figure
    fig_height = max(4, n_channels * 0.8)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    # Plot each channel with vertical offset
    offsets = np.arange(n_channels) * scale_uv
    yticks = []
    yticklabels = []

    for i, ch_name in enumerate(ch_names):
        offset = offsets[n_channels - 1 - i]  # Top channel first
        ax.plot(time_plot, data_plot[i, :] + offset, linewidth=0.5, color="C0")
        yticks.append(offset)
        yticklabels.append(ch_name)

    # Configure axes
    ax.set_xlim(time_plot[0], time_plot[-1])
    ax.set_ylim(-scale_uv * 0.5, offsets[-1] + scale_uv * 0.5)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude (μV, relative)")
    ax.set_title(f"EEG Time Series ({len(ch_names)} channels)")
    ax.grid(True, alpha=0.3, axis="x")

    # Add scale bar annotation
    ax.annotate(
        f"scale: {scale_uv:.1f} μV",
        xy=(0.98, 0.02),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=8,
        color="gray",
    )

    plt.tight_layout()

    # Save figure
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_signals(
    data: pd.DataFrame | np.ndarray,
    fs: float = 256.0,
    channels: list[str] | None = None,
    title: str = "EEG Signals",
    figsize: tuple[int, int] = (12, 6),
    save_path: str | Path | None = None,
) -> plt.Figure:
    # plots eeg signals
    if isinstance(data, pd.DataFrame):
        if channels is None:
            channels = list(data.columns)
        data = data.values
    else:
        if channels is None:
            channels = [f"Ch{i+1}" for i in range(data.shape[1])]

    n_samples = data.shape[0]
    time = np.arange(n_samples) / fs

    fig, axes = plt.subplots(len(channels), 1, figsize=figsize, sharex=True)
    if len(channels) == 1:
        axes = [axes]

    for i, (ax, ch) in enumerate(zip(axes, channels)):
        ax.plot(time, data[:, i], linewidth=0.5)
        ax.set_ylabel(ch)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_psd(
    freqs: np.ndarray,
    psd: np.ndarray,
    channels: list[str] | None = None,
    title: str = "Power Spectral Density",
    figsize: tuple[int, int] = (10, 6),
    xlim: tuple[float, float] = (0, 50),
    save_path: str | Path | None = None,
) -> plt.Figure:
    # plots power spectral density
    if psd.ndim == 1:
        psd = psd.reshape(-1, 1)

    n_channels = psd.shape[1]
    if channels is None:
        channels = [f"Ch{i+1}" for i in range(n_channels)]

    fig, ax = plt.subplots(figsize=figsize)

    for i, ch in enumerate(channels):
        ax.semilogy(freqs, psd[:, i], label=ch, linewidth=1)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (V²/Hz)")
    ax.set_title(title)
    ax.set_xlim(xlim)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig
