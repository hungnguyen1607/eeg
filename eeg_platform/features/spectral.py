# spectral features for eeg

import numpy as np
from scipy import signal


# Standard EEG frequency bands
EEG_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def compute_psd_welch(
    data: np.ndarray,
    sfreq: float,
    nperseg: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    # computes psd using welch's method
    n_samples = data.shape[1]

    # Adjust nperseg if data is shorter
    nperseg = min(nperseg, n_samples)

    # Compute PSD along axis=1 (samples)
    freqs, psd = signal.welch(
        data,
        fs=sfreq,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        axis=1,
    )

    return freqs, psd


def bandpower(
    psd: np.ndarray,
    freqs: np.ndarray,
    band: tuple[float, float],
) -> np.ndarray:
    # power in a frequency band
    low, high = band

    # Find frequency indices within band
    band_mask = (freqs >= low) & (freqs <= high)

    if not np.any(band_mask):
        # No frequencies in band
        return np.zeros(psd.shape[0])

    # Integrate PSD over band using trapezoidal rule
    band_freqs = freqs[band_mask]
    band_psd = psd[:, band_mask]

    # np.trapezoid integrates along last axis by default
    band_power = np.trapezoid(band_psd, band_freqs, axis=1)

    return band_power


def bandpower_from_data(
    data: np.ndarray,
    sfreq: float,
    band: str | tuple[float, float],
    nperseg: int = 512,
) -> np.ndarray:
    # computes band power directly from raw data
    if isinstance(band, str):
        if band not in EEG_BANDS:
            raise ValueError(f"Unknown band: {band}. Use one of {list(EEG_BANDS.keys())}")
        band = EEG_BANDS[band]

    freqs, psd = compute_psd_welch(data, sfreq, nperseg)
    return bandpower(psd, freqs, band)


def dominant_frequency(
    psd: np.ndarray,
    freqs: np.ndarray,
    fmin: float = 1.0,
    fmax: float = 45.0,
) -> np.ndarray:
    # finds the peak frequency per channel
    # Find frequency indices within range
    freq_mask = (freqs >= fmin) & (freqs <= fmax)

    if not np.any(freq_mask):
        return np.zeros(psd.shape[0])

    masked_freqs = freqs[freq_mask]
    masked_psd = psd[:, freq_mask]

    # Find index of max power for each channel
    peak_indices = np.argmax(masked_psd, axis=1)

    # Convert indices to frequencies
    dominant_freqs = masked_freqs[peak_indices]

    return dominant_freqs


def compute_all_bandpowers(
    data: np.ndarray,
    sfreq: float,
    nperseg: int = 512,
) -> dict[str, np.ndarray]:
    # computes power for all standard eeg bands
    freqs, psd = compute_psd_welch(data, sfreq, nperseg)

    powers = {}
    for band_name, band_range in EEG_BANDS.items():
        powers[band_name] = bandpower(psd, freqs, band_range)

    return powers


def relative_bandpower(
    psd: np.ndarray,
    freqs: np.ndarray,
    band: tuple[float, float],
    total_band: tuple[float, float] = (1.0, 45.0),
) -> np.ndarray:
    # relative band power (band / total)
    band_pow = bandpower(psd, freqs, band)
    total_pow = bandpower(psd, freqs, total_band)

    # Avoid division by zero
    total_pow = np.where(total_pow == 0, 1.0, total_pow)

    return band_pow / total_pow


# legacy psd func - data shape (samples, channels)
def compute_psd(data, fs, nperseg=None):
    if nperseg is None:
        nperseg = min(256, len(data) // 4)
    freqs, psd = signal.welch(data, fs=fs, nperseg=nperseg, axis=0)
    return freqs, psd


def compute_band_power(data, fs, band):
    # legacy band power - data shape (samples, channels)
    if isinstance(band, str):
        if band not in EEG_BANDS:
            raise ValueError(f"Unknown band: {band}. Use one of {list(EEG_BANDS.keys())}")
        low, high = EEG_BANDS[band]
    else:
        low, high = band

    freqs, psd = compute_psd(data, fs)
    band_mask = (freqs >= low) & (freqs <= high)
    band_power = np.trapezoid(psd[band_mask], freqs[band_mask], axis=0)
    return band_power
