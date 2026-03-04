# cleaning functions for eeg data

import numpy as np

from .filters import bandpass_filter, notch_filter


def remove_dc_offset(data: np.ndarray) -> np.ndarray:
    # removes dc offset (mean) from each channel
    return data - np.mean(data, axis=1, keepdims=True)


def robust_zscore(data: np.ndarray) -> np.ndarray:
    # robust z-score using median/MAD instead of mean/std
    median = np.median(data, axis=1, keepdims=True)
    mad = np.median(np.abs(data - median), axis=1, keepdims=True)
    # Scale MAD to approximate std for normal distribution
    mad_std = mad * 1.4826
    # Avoid division by zero
    mad_std = np.where(mad_std == 0, 1.0, mad_std)
    return (data - median) / mad_std


def normalize_channels(
    data: np.ndarray,
    method: str = "zscore",
) -> np.ndarray:
    # normalizes each channel (zscore, robust, or minmax)
    if method == "zscore":
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        std = np.where(std == 0, 1.0, std)
        return (data - mean) / std

    elif method == "robust":
        return robust_zscore(data)

    elif method == "minmax":
        min_val = np.min(data, axis=1, keepdims=True)
        max_val = np.max(data, axis=1, keepdims=True)
        range_val = max_val - min_val
        range_val = np.where(range_val == 0, 1.0, range_val)
        return (data - min_val) / range_val

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def clean_signal(
    data: np.ndarray,
    sfreq: float,
    bandpass: tuple[float, float] = (1.0, 45.0),
    notch: float | None = 60.0,
    normalize: str | None = None,
) -> np.ndarray:
    # cleans signal: notch filter -> bandpass -> optional normalization
    cleaned = data.copy()

    # Step 1: Apply notch filter to remove line noise
    if notch is not None:
        cleaned = notch_filter(cleaned, sfreq, notch_freq=notch)

    # Step 2: Apply bandpass filter
    l_freq, h_freq = bandpass
    cleaned = bandpass_filter(cleaned, sfreq, l_freq, h_freq)

    # Step 3: Optional normalization
    if normalize is not None:
        cleaned = normalize_channels(cleaned, method=normalize)

    return cleaned
