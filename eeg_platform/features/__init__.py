# feature extraction for eeg analysis

from .spectral import (
    compute_psd_welch,
    bandpower,
    bandpower_from_data,
    dominant_frequency,
    compute_all_bandpowers,
    relative_bandpower,
    EEG_BANDS,
    # Legacy
    compute_psd,
    compute_band_power,
)
from .time_domain import (
    compute_mean,
    compute_std,
    compute_rms,
    compute_variance,
    compute_peak_to_peak,
    compute_skewness,
    compute_kurtosis,
    compute_zero_crossings,
    compute_min,
    compute_max,
    extract_time_features,
)

__all__ = [
    # Spectral
    "compute_psd_welch",
    "bandpower",
    "bandpower_from_data",
    "dominant_frequency",
    "compute_all_bandpowers",
    "relative_bandpower",
    "EEG_BANDS",
    "compute_psd",
    "compute_band_power",
    # Time domain
    "compute_mean",
    "compute_std",
    "compute_rms",
    "compute_variance",
    "compute_peak_to_peak",
    "compute_skewness",
    "compute_kurtosis",
    "compute_zero_crossings",
    "compute_min",
    "compute_max",
    "extract_time_features",
]
