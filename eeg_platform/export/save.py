# export functions for saving eeg data

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def save_json(path: str | Path, obj: Any) -> Path:
    # saves object to json
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

    return filepath


def save_csv_rows(path: str | Path, rows: list[dict]) -> Path:
    # saves list of dicts to csv
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        # Empty file with no rows
        filepath.touch()
        return filepath

    # Get all unique keys maintaining order from first row
    fieldnames = list(rows[0].keys())

    # Add any additional keys from other rows
    for row in rows[1:]:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return filepath


def save_csv(
    data: pd.DataFrame | np.ndarray,
    filepath: str | Path,
    channels: list[str] | None = None,
    include_index: bool = True,
) -> Path:
    # saves eeg data to csv
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(data, np.ndarray):
        if channels is None:
            channels = [f"Ch{i+1}" for i in range(data.shape[1])]
        data = pd.DataFrame(data, columns=channels)

    data.to_csv(filepath, index=include_index)
    return filepath


def save_numpy(
    data: pd.DataFrame | np.ndarray,
    filepath: str | Path,
) -> Path:
    # saves to numpy .npy file
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(data, pd.DataFrame):
        data = data.values

    np.save(filepath, data)
    return filepath
