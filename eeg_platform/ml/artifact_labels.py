# generates and loads artifact labels

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


# Default thresholds for weak-label generation
DEFAULT_THRESHOLDS = {
    "ptp_max": 150.0,      # uV - peak-to-peak above this is likely artifact
    "rms_max": 75.0,       # uV - RMS above this is likely artifact
    "rms_min": 0.5,        # uV - RMS below this is likely flatline
    "line_noise_max": 10.0, # Power at 60Hz above this indicates noise
    "kurtosis_max": 10.0,  # Extreme kurtosis indicates spikes
    "kurtosis_min": -5.0,  # Very negative kurtosis is also suspicious
}


class LabelError(Exception):
    # something went wrong with labels
    pass


def generate_weak_labels(
    df: pd.DataFrame,
    thresholds: Optional[dict] = None,
    strict: bool = False,
) -> np.ndarray:
    # generates weak labels with rule-based heuristics (0=clean, 1=artifact)
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS.copy()

    n_windows = len(df)
    violations = np.zeros(n_windows, dtype=int)

    # Check peak-to-peak amplitude
    if "ptp_uv" in df.columns:
        ptp_max = thresholds.get("ptp_max", 150.0)
        violations += (df["ptp_uv"].values > ptp_max).astype(int)
    elif "peak_to_peak" in df.columns:
        ptp_max = thresholds.get("ptp_max", 150.0)
        violations += (df["peak_to_peak"].values > ptp_max).astype(int)

    # Check RMS amplitude (too high = artifact, too low = flatline)
    rms_col = None
    for col in ["rms_uv", "rms", "rms_amplitude"]:
        if col in df.columns:
            rms_col = col
            break

    if rms_col:
        rms_max = thresholds.get("rms_max", 75.0)
        rms_min = thresholds.get("rms_min", 0.5)
        rms_vals = df[rms_col].values
        violations += (rms_vals > rms_max).astype(int)
        violations += (rms_vals < rms_min).astype(int)

    # Check line noise
    if "line_noise_power" in df.columns:
        line_max = thresholds.get("line_noise_max", 10.0)
        violations += (df["line_noise_power"].values > line_max).astype(int)

    # Check kurtosis (extreme values indicate spikes or artifacts)
    if "kurtosis" in df.columns:
        kurt_max = thresholds.get("kurtosis_max", 10.0)
        kurt_min = thresholds.get("kurtosis_min", -5.0)
        kurt_vals = df["kurtosis"].values
        violations += (kurt_vals > kurt_max).astype(int)
        violations += (kurt_vals < kurt_min).astype(int)

    # Generate labels
    if strict:
        # Any violation = artifact
        labels = (violations > 0).astype(int)
    else:
        # Multiple violations = artifact (more lenient)
        labels = (violations >= 2).astype(int)

    return labels


def load_artifact_labels(
    label_path: str | Path,
    df: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    # loads manual labels from csv
    label_path = Path(label_path)

    if not label_path.exists():
        raise LabelError(f"Label file not found: {label_path}")

    try:
        labels_df = pd.read_csv(label_path)
    except Exception as e:
        raise LabelError(f"Error reading label file: {e}")

    # Determine format
    if "label" not in labels_df.columns:
        raise LabelError("Label file must have 'label' column")

    if "window_id" in labels_df.columns:
        # Sort by window_id to ensure correct order
        labels_df = labels_df.sort_values("window_id")
        labels = labels_df["label"].values
    else:
        # Assume rows are in order
        labels = labels_df["label"].values

    # Validate against feature DataFrame if provided
    if df is not None:
        if len(labels) != len(df):
            raise LabelError(
                f"Label count ({len(labels)}) != window count ({len(df)})"
            )

    # Validate label values
    unique_labels = np.unique(labels)
    if not np.all(np.isin(unique_labels, [0, 1])):
        raise LabelError(
            f"Labels must be 0 or 1, found: {unique_labels}"
        )

    return labels.astype(int)


def save_weak_labels(
    df: pd.DataFrame,
    labels: np.ndarray,
    output_path: str | Path,
    recording_id: Optional[str] = None,
) -> Path:
    # saves weak labels to csv
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build label DataFrame
    label_data = {
        "window_id": np.arange(len(labels)),
        "label": labels,
    }

    if recording_id:
        label_data["recording_id"] = recording_id

    # Add key feature values for inspection
    for col in ["ptp_uv", "rms_uv", "kurtosis", "line_noise_power"]:
        if col in df.columns:
            label_data[col] = df[col].values

    label_df = pd.DataFrame(label_data)
    label_df.to_csv(output_path, index=False)

    return output_path


def get_label_statistics(labels: np.ndarray) -> dict:
    # stats about label distribution
    n_total = len(labels)
    n_artifact = int(np.sum(labels == 1))
    n_clean = int(np.sum(labels == 0))

    return {
        "n_total": n_total,
        "n_artifact": n_artifact,
        "n_clean": n_clean,
        "artifact_rate": n_artifact / max(n_total, 1),
        "clean_rate": n_clean / max(n_total, 1),
    }
