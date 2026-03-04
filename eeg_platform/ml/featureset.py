# loads and selects features for ml models

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

# Core ML feature columns used for artifact detection
# These are features that indicate signal quality
ML_FEATURE_COLUMNS = [
    # Time-domain features
    "rms_uv",
    "ptp_uv",  # peak-to-peak amplitude
    "std_uv",
    "variance",
    "kurtosis",
    "skewness",
    "zero_crossings",
    # Spectral features
    "alpha_power",
    "beta_power",
    "theta_power",
    "delta_power",
    "gamma_power",
    "line_noise_power",  # 60 Hz power
    "theta_beta_ratio",
    "alpha_theta_ratio",
    "dominant_freq_hz",
    # Derived quality indicators
    "high_freq_power",  # > 30 Hz, often muscle artifact
    "low_freq_power",   # < 1 Hz, often movement/drift
]


class FeatureLoadError(Exception):
    # feature loading failed
    pass


def load_window_features(
    csv_path: str | Path,
    required_columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    # loads per-window features from csv
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FeatureLoadError(f"Feature file not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise FeatureLoadError(f"Error reading CSV: {e}")

    if len(df) == 0:
        raise FeatureLoadError("Feature file is empty")

    # Check for required columns
    if required_columns is None:
        required_columns = []  # Will be checked later in select_ml_features

    missing = set(required_columns) - set(df.columns)
    if missing:
        raise FeatureLoadError(
            f"Missing required columns: {sorted(missing)}. "
            f"Available: {sorted(df.columns)}"
        )

    return df


def select_ml_features(
    df: pd.DataFrame,
    feature_columns: Optional[list[str]] = None,
    min_required: int = 3,
) -> tuple[np.ndarray, list[str]]:
    # picks and validates feature columns, returns X matrix and column names
    if feature_columns is None:
        # Auto-detect available features from standard list
        available = [col for col in ML_FEATURE_COLUMNS if col in df.columns]

        # Also check for alternative naming conventions
        alt_mappings = {
            "rms_uv": ["rms", "rms_amplitude"],
            "ptp_uv": ["peak_to_peak", "ptp", "amplitude_range"],
            "std_uv": ["std", "standard_deviation"],
            "alpha_power": ["bandpower_alpha", "alpha"],
            "beta_power": ["bandpower_beta", "beta"],
            "theta_power": ["bandpower_theta", "theta"],
            "delta_power": ["bandpower_delta", "delta"],
            "gamma_power": ["bandpower_gamma", "gamma"],
        }

        for primary, alternatives in alt_mappings.items():
            if primary not in available:
                for alt in alternatives:
                    if alt in df.columns and alt not in available:
                        available.append(alt)
                        break

        feature_columns = available

    if len(feature_columns) < min_required:
        raise FeatureLoadError(
            f"Need at least {min_required} features, found {len(feature_columns)}. "
            f"Available columns: {sorted(df.columns)}"
        )

    # Extract feature matrix
    X = df[feature_columns].values.astype(np.float64)

    # Handle NaN/Inf values
    nan_mask = ~np.isfinite(X)
    if nan_mask.any():
        # Replace with column median
        for col_idx in range(X.shape[1]):
            col = X[:, col_idx]
            col_nan = ~np.isfinite(col)
            if col_nan.any():
                median_val = np.nanmedian(col[np.isfinite(col)])
                if np.isfinite(median_val):
                    col[col_nan] = median_val
                else:
                    col[col_nan] = 0.0

    return X, feature_columns


def compute_window_features(
    window_data: np.ndarray,
    sfreq: float,
    ch_names: Optional[list[str]] = None,
) -> dict:
    # computes all features for one window, returns dict
    from ..features.time_domain import extract_time_features
    from ..features.spectral import (
        compute_psd_welch,
        bandpower,
        dominant_frequency,
        EEG_BANDS,
    )

    n_channels, n_samples = window_data.shape

    # Time-domain features
    time_feats = extract_time_features(window_data)

    # Average across channels
    features = {
        "rms_uv": float(np.mean(time_feats["rms"])),
        "ptp_uv": float(np.mean(time_feats["peak_to_peak"])),
        "std_uv": float(np.mean(time_feats["std"])),
        "variance": float(np.mean(time_feats["variance"])),
        "kurtosis": float(np.mean(time_feats["kurtosis"])),
        "skewness": float(np.mean(time_feats["skewness"])),
        "zero_crossings": float(np.mean(time_feats["zero_crossings"])),
        "mean": float(np.mean(time_feats["mean"])),
        "min": float(np.mean(time_feats["min"])),
        "max": float(np.mean(time_feats["max"])),
    }

    # Spectral features
    nperseg = min(256, n_samples // 2)
    if nperseg >= 16:
        freqs, psd = compute_psd_welch(window_data, sfreq, nperseg=nperseg)

        for band_name, band_range in EEG_BANDS.items():
            bp = bandpower(psd, freqs, band_range)
            features[f"{band_name}_power"] = float(np.mean(bp))

        # Dominant frequency
        dom_freq = dominant_frequency(psd, freqs, fmin=1.0, fmax=45.0)
        features["dominant_freq_hz"] = float(np.mean(dom_freq))

        # Line noise power (55-65 Hz)
        line_noise = bandpower(psd, freqs, (55.0, 65.0))
        features["line_noise_power"] = float(np.mean(line_noise))

        # High frequency power (muscle artifact indicator)
        high_freq = bandpower(psd, freqs, (30.0, sfreq / 2 - 1))
        features["high_freq_power"] = float(np.mean(high_freq))

        # Low frequency power (drift indicator)
        low_freq = bandpower(psd, freqs, (0.1, 1.0))
        features["low_freq_power"] = float(np.mean(low_freq))

        # Ratios
        theta = features.get("theta_power", 0)
        beta = features.get("beta_power", 0)
        alpha = features.get("alpha_power", 0)

        features["theta_beta_ratio"] = theta / max(beta, 1e-10)
        features["alpha_theta_ratio"] = alpha / max(theta, 1e-10)

    return features


def extract_all_window_features(
    data: np.ndarray,
    sfreq: float,
    window_sec: float = 2.0,
    overlap: float = 0.5,
    ch_names: Optional[list[str]] = None,
    recording_id: Optional[str] = None,
) -> pd.DataFrame:
    # extracts features for all windows, returns df with one row per window
    from ..preprocess.windowing import segment_into_windows

    windows, window_times = segment_into_windows(
        data, sfreq, window_sec=window_sec, overlap=overlap
    )

    rows = []
    for i, (window, start_time) in enumerate(zip(windows, window_times)):
        feats = compute_window_features(window, sfreq, ch_names)
        feats["window_id"] = i
        feats["start_time_s"] = float(start_time)
        feats["duration_s"] = window_sec
        if recording_id:
            feats["recording_id"] = recording_id
        rows.append(feats)

    return pd.DataFrame(rows)
