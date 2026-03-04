# time-domain features for eeg

import numpy as np
from scipy import stats


def compute_mean(data: np.ndarray) -> np.ndarray:
    # mean per channel
    return np.mean(data, axis=1)


def compute_std(data: np.ndarray) -> np.ndarray:
    # std per channel
    return np.std(data, axis=1)


def compute_rms(data: np.ndarray) -> np.ndarray:
    # rms per channel
    return np.sqrt(np.mean(data**2, axis=1))


def compute_variance(data: np.ndarray) -> np.ndarray:
    # variance per channel
    return np.var(data, axis=1)


def compute_peak_to_peak(data: np.ndarray) -> np.ndarray:
    # peak-to-peak amplitude per channel
    return np.ptp(data, axis=1)


def compute_skewness(data: np.ndarray) -> np.ndarray:
    # skewness per channel (asymmetry measure)
    return stats.skew(data, axis=1)


def compute_kurtosis(data: np.ndarray) -> np.ndarray:
    # kurtosis per channel (tailedness measure)
    return stats.kurtosis(data, axis=1, fisher=True)


def compute_zero_crossings(data: np.ndarray) -> np.ndarray:
    # counts zero crossings per channel
    return np.sum(np.abs(np.diff(np.sign(data), axis=1)) > 0, axis=1)


def compute_min(data: np.ndarray) -> np.ndarray:
    # min per channel
    return np.min(data, axis=1)


def compute_max(data: np.ndarray) -> np.ndarray:
    # max per channel
    return np.max(data, axis=1)


def extract_time_features(data: np.ndarray) -> dict[str, np.ndarray]:
    # extracts all time-domain features
    return {
        "mean": compute_mean(data),
        "std": compute_std(data),
        "rms": compute_rms(data),
        "variance": compute_variance(data),
        "peak_to_peak": compute_peak_to_peak(data),
        "min": compute_min(data),
        "max": compute_max(data),
        "zero_crossings": compute_zero_crossings(data),
        "skewness": compute_skewness(data),
        "kurtosis": compute_kurtosis(data),
    }
