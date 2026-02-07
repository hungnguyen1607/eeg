"""Load EEG data from CSV files."""

from pathlib import Path

import numpy as np
import pandas as pd


class EEGLoadError(Exception):
    """Raised when EEG data cannot be loaded."""

# nan = not a number 
def load_eeg_csv(path: str, sfreq: float | None = None) -> dict:
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    #  loading csv 
    try:
        df = pd.read_csv(filepath)
    except pd.errors.EmptyDataError:
        raise EEGLoadError(f"File is empty: {filepath}")

    if df.empty:
        raise EEGLoadError(f"File contains no data rows: {filepath}")

    # checking for time column
    has_time_col = "time" in df.columns.str.lower()
    time_col_name = None
    if has_time_col:
        for col in df.columns:
            if col.lower() == "time":
                time_col_name = col
                break

    
    if time_col_name:
        time_data = pd.to_numeric(df[time_col_name], errors="coerce")
        if time_data.isna().all():  # take time array and check if all values are NaN
            raise EEGLoadError(f"Time column '{time_col_name}' contains no valid numeric values")
        time = time_data.to_numpy(dtype=np.float64) # convert time column to numeric coercing errors to nan then to numpy array

        # use sfreq from time column if not provided
        if sfreq is None:
            if len(time) < 2:
                raise EEGLoadError("Cannot infer sfreq: need at least 2 time samples")
            dt = np.median(np.diff(time[:min(100, len(time))]))
            if dt <= 0:
                raise EEGLoadError(f"Invalid time column: non-positive time step ({dt})")
            sfreq = 1.0 / dt

        df = df.drop(columns=[time_col_name])
    else:
        # use sfreq to make time array if no time column was detected
        if sfreq is None:
            raise EEGLoadError(
                "No 'time' column found and sfreq not provided. "
                "Either include a 'time' column or specify sfreq."
            )
        n_samples = len(df)
        time = np.arange(n_samples, dtype=np.float64) / sfreq

    # convert all columns to numeric, coercing errors to nan 
    df_numeric = df.apply(pd.to_numeric, errors="coerce")

    # Check for numeric columns
    numeric_cols = df_numeric.columns[df_numeric.notna().any()].tolist()
    if not numeric_cols:
        raise EEGLoadError(
            f"No numeric columns found. Columns present: {list(df.columns)}"
        )

    df_numeric = df_numeric[numeric_cols]
    ch_names = numeric_cols

    # finding nans values
    nan_counts = df_numeric.isna().sum()
    total_values = len(df_numeric)
    nan_threshold = 0.1  # 10% threshold

    for col in ch_names:
        nan_ratio = nan_counts[col] / total_values
        if nan_ratio > nan_threshold:
            raise EEGLoadError(
                f"Channel '{col}' has {nan_ratio:.1%} NaN values (threshold: {nan_threshold:.0%}). "
                f"Check for non-numeric data in this column."
            )

    # fill all the nans with 0 
    total_nans = df_numeric.isna().sum().sum()
    if total_nans > 0:
        df_numeric = df_numeric.fillna(0.0)

    # convert to numpy array (n_channels, n_samples)
    data = df_numeric.to_numpy(dtype=np.float64).T

    n_channels, n_samples = data.shape

    #  making sure time array length = number of samples
    if len(time) != n_samples:
        raise EEGLoadError(
            f"Time array length ({len(time)}) does not match data samples ({n_samples})"
        ) #checking time stamps, making sure its correctly spaced

    meta = {
        "filepath": str(filepath.resolve()),
        "n_channels": n_channels,
        "n_samples": n_samples,
    }

    return {
        "data": data,
        "ch_names": ch_names,
        "time": time,
        "sfreq": float(sfreq),
        "meta": meta,
    }
