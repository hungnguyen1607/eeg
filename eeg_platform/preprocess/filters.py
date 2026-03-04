# signal filtering functions for eeg preprocessing

import numpy as np
from scipy import signal


# raised when filter parameters are invalid
class FilterError(Exception):
    pass


# validate frequencies against nyquist
def _validate_frequencies(sfreq: float, *freqs: float) -> None:
    nyquist = sfreq / 2
    for freq in freqs:
        if freq is not None and freq >= nyquist:
            raise FilterError(
                f"Frequency {freq} Hz >= Nyquist frequency {nyquist} Hz "
                f"(sfreq={sfreq} Hz). Reduce frequency or increase sampling rate."
            )
        if freq is not None and freq <= 0:
            raise FilterError(f"Frequency must be positive, got {freq} Hz")


# apply a butterworth bandpass filter using zero-phase filtering
# data: eeg data shape (n_channels, n_samples)
# sfreq: sampling frequency in hz
# l_freq: low cutoff frequency in hz
# h_freq: high cutoff frequency in hz
# order: filter order (default 4)
# returns filtered data, same shape as input
def bandpass_filter(
    data: np.ndarray,
    sfreq: float,
    l_freq: float,
    h_freq: float,
    order: int = 4,
) -> np.ndarray:
    if l_freq >= h_freq:
        raise FilterError(
            f"Low frequency ({l_freq} Hz) must be less than "
            f"high frequency ({h_freq} Hz)"
        )

    _validate_frequencies(sfreq, l_freq, h_freq)

    nyquist = sfreq / 2
    low_norm = l_freq / nyquist
    high_norm = h_freq / nyquist

    # design butterworth bandpass filter
    sos = signal.butter(order, [low_norm, high_norm], btype="band", output="sos")

    # apply zero-phase filtering along samples axis (axis=1)
    filtered = signal.sosfiltfilt(sos, data, axis=1)

    return filtered.astype(np.float64)


# apply a notch filter to remove line noise using zero-phase filtering
# data: eeg data shape (n_channels, n_samples)
# sfreq: sampling frequency in hz
# notch_freq: frequency to remove in hz (default 60 for us power line)
# quality: quality factor q (default 30). higher q = narrower notch
# returns filtered data, same shape as input
def notch_filter(
    data: np.ndarray,
    sfreq: float,
    notch_freq: float = 60.0,
    quality: float = 30.0,
) -> np.ndarray:
    _validate_frequencies(sfreq, notch_freq)

    nyquist = sfreq / 2
    norm_freq = notch_freq / nyquist

    # design iir notch filter
    b, a = signal.iirnotch(norm_freq, quality)

    # apply zero-phase filtering along samples axis (axis=1)
    filtered = signal.filtfilt(b, a, data, axis=1)

    return filtered.astype(np.float64)


# apply a butterworth highpass filter using zero-phase filtering
# data: eeg data shape (n_channels, n_samples)
# sfreq: sampling frequency in hz
# cutoff: cutoff frequency in hz
# order: filter order (default 4)
# returns filtered data, same shape as input
def highpass_filter(
    data: np.ndarray,
    sfreq: float,
    cutoff: float,
    order: int = 4,
) -> np.ndarray:
    _validate_frequencies(sfreq, cutoff)

    nyquist = sfreq / 2
    norm_cutoff = cutoff / nyquist

    sos = signal.butter(order, norm_cutoff, btype="high", output="sos")
    filtered = signal.sosfiltfilt(sos, data, axis=1)

    return filtered.astype(np.float64)


# apply a butterworth lowpass filter using zero-phase filtering
# data: eeg data shape (n_channels, n_samples)
# sfreq: sampling frequency in hz
# cutoff: cutoff frequency in hz
# order: filter order (default 4)
# returns filtered data, same shape as input
def lowpass_filter(
    data: np.ndarray,
    sfreq: float,
    cutoff: float,
    order: int = 4,
) -> np.ndarray:
    _validate_frequencies(sfreq, cutoff)

    nyquist = sfreq / 2
    norm_cutoff = cutoff / nyquist

    sos = signal.butter(order, norm_cutoff, btype="low", output="sos")
    filtered = signal.sosfiltfilt(sos, data, axis=1)

    return filtered.astype(np.float64)
