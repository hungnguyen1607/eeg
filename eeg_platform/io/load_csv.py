# load eeg data from csv files

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .validation import (
    validate_eeg_data,
    normalize_channel_name,
    print_validation_report,
    ValidationResult,
    STANDARD_10_20_CHANNELS,
)


# custom error for when eeg data cant be loaded
class EEGLoadError(Exception):
    pass


# main function to load eeg csv files
# path: path to csv file
# sfreq: sampling frequency in hz (required if no time column)
# validate: check if data follows eeg standards
# strict: reject files with non-standard columns
# auto_map_channels: rename generic channels like EEG1 to Fp1
def load_eeg_csv(
    path: str,
    sfreq: float | None = None,
    validate: bool = True,
    strict: bool = False,
    auto_map_channels: bool = True,
) -> dict:
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # load the csv
    try:
        df = pd.read_csv(filepath)
    except pd.errors.EmptyDataError:
        raise EEGLoadError(f"File is empty: {filepath}")

    if df.empty:
        raise EEGLoadError(f"File contains no data rows: {filepath}")

    # check if theres a time column
    has_time_col = "time" in df.columns.str.lower()
    time_col_name = None
    if has_time_col:
        for col in df.columns:
            if col.lower() == "time":
                time_col_name = col
                break

    if time_col_name:
        time_data = pd.to_numeric(df[time_col_name], errors="coerce")
        # check if all values are nan
        if time_data.isna().all():
            raise EEGLoadError(f"Time column '{time_col_name}' contains no valid numeric values")
        # convert to numpy array
        time = time_data.to_numpy(dtype=np.float64)

        # figure out sample rate from time column if not given
        if sfreq is None:
            if len(time) < 2:
                raise EEGLoadError("Cannot infer sfreq: need at least 2 time samples")
            dt = np.median(np.diff(time[:min(100, len(time))]))
            if dt <= 0:
                raise EEGLoadError(f"Invalid time column: non-positive time step ({dt})")
            sfreq = 1.0 / dt

        df = df.drop(columns=[time_col_name])
    else:
        # no time column, so we need sfreq to create one
        if sfreq is None:
            raise EEGLoadError(
                "No 'time' column found and sfreq not provided. "
                "Either include a 'time' column or specify sfreq."
            )
        n_samples = len(df)
        time = np.arange(n_samples, dtype=np.float64) / sfreq

    # convert everything to numbers, turn bad values into nan
    df_numeric = df.apply(pd.to_numeric, errors="coerce")

    # find columns that actually have numbers
    numeric_cols = df_numeric.columns[df_numeric.notna().any()].tolist()
    if not numeric_cols:
        raise EEGLoadError(
            f"No numeric columns found. Columns present: {list(df.columns)}"
        )

    df_numeric = df_numeric[numeric_cols]
    ch_names = numeric_cols

    # check for too many nan values
    nan_counts = df_numeric.isna().sum()
    total_values = len(df_numeric)
    nan_threshold = 0.1  # 10% max

    for col in ch_names:
        nan_ratio = nan_counts[col] / total_values
        if nan_ratio > nan_threshold:
            raise EEGLoadError(
                f"Channel '{col}' has {nan_ratio:.1%} NaN values (threshold: {nan_threshold:.0%}). "
                f"Check for non-numeric data in this column."
            )

    # fill remaining nans with 0
    total_nans = df_numeric.isna().sum().sum()
    if total_nans > 0:
        df_numeric = df_numeric.fillna(0.0)

    # convert to numpy array with shape (n_channels, n_samples)
    data = df_numeric.to_numpy(dtype=np.float64).T

    n_channels, n_samples = data.shape

    # make sure time array matches data length
    if len(time) != n_samples:
        raise EEGLoadError(
            f"Time array length ({len(time)}) does not match data samples ({n_samples})"
        )

    # validate the eeg data if enabled
    validation_result = None
    if validate:
        validation_result = validate_eeg_data(
            columns=ch_names,
            n_samples=n_samples,
            sfreq=sfreq,
            strict=strict,
        )

        if not validation_result.is_valid:
            error_msg = "; ".join(validation_result.errors)
            raise EEGLoadError(f"Validation failed: {error_msg}")

        # auto rename channels like EEG1 -> Fp1
        if auto_map_channels and validation_result.channel_mapping:
            new_ch_names = []
            valid_indices = []

            for i, ch in enumerate(ch_names):
                if ch in validation_result.channel_mapping:
                    new_ch_names.append(validation_result.channel_mapping[ch])
                    valid_indices.append(i)

            # only keep valid channels
            if valid_indices:
                ch_names = new_ch_names
                data = data[valid_indices, :]
                n_channels = len(ch_names)

    meta = {
        "filepath": str(filepath.resolve()),
        "n_channels": n_channels,
        "n_samples": n_samples,
    }

    result = {
        "data": data,
        "ch_names": ch_names,
        "time": time,
        "sfreq": float(sfreq),
        "meta": meta,
    }

    if validation_result:
        result["validation"] = validation_result

    return result
