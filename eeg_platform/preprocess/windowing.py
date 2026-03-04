# segments eeg into windows

import numpy as np
from typing import Iterator


def segment_into_windows(
    data: np.ndarray,
    sfreq: float,
    window_sec: float = 2.0,
    overlap: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    # chops data into overlapping windows
    n_channels, n_samples = data.shape
    window_samples = int(window_sec * sfreq)
    step_samples = int(window_samples * (1 - overlap))

    if step_samples < 1:
        step_samples = 1

    # Calculate number of windows
    n_windows = (n_samples - window_samples) // step_samples + 1

    if n_windows < 1:
        # Data too short for even one window
        # Return single window padded with zeros if needed
        if n_samples > 0:
            padded = np.zeros((n_channels, window_samples))
            padded[:, :n_samples] = data
            return padded.reshape(1, n_channels, window_samples), np.array([0.0])
        else:
            return np.empty((0, n_channels, window_samples)), np.array([])

    windows = np.zeros((n_windows, n_channels, window_samples))
    window_times = np.zeros(n_windows)

    for i in range(n_windows):
        start = i * step_samples
        end = start + window_samples
        windows[i] = data[:, start:end]
        window_times[i] = start / sfreq

    return windows, window_times


def window_iterator(
    data: np.ndarray,
    sfreq: float,
    window_sec: float = 2.0,
    overlap: float = 0.5,
) -> Iterator[tuple[int, np.ndarray, float]]:
    # iterates windows without storing all in memory
    n_channels, n_samples = data.shape
    window_samples = int(window_sec * sfreq)
    step_samples = int(window_samples * (1 - overlap))

    if step_samples < 1:
        step_samples = 1

    window_id = 0
    start = 0

    while start + window_samples <= n_samples:
        window_data = data[:, start:start + window_samples]
        start_time = start / sfreq
        yield window_id, window_data, start_time
        window_id += 1
        start += step_samples


def get_window_count(
    n_samples: int,
    sfreq: float,
    window_sec: float = 2.0,
    overlap: float = 0.5,
) -> int:
    # calculates how many windows will fit
    window_samples = int(window_sec * sfreq)
    step_samples = int(window_samples * (1 - overlap))

    if step_samples < 1:
        step_samples = 1

    n_windows = (n_samples - window_samples) // step_samples + 1
    return max(0, n_windows)
