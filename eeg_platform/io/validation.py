# eeg data validation and standardization
# checks if input data follows eeg conventions and maps channel names to 10-20 system

from typing import Optional
import re


# standard 10-20 system channel names
STANDARD_10_20_CHANNELS = {
    # frontal pole
    "Fp1", "Fpz", "Fp2",
    # frontal
    "F7", "F3", "Fz", "F4", "F8",
    # temporal
    "T3", "T4", "T5", "T6", "T7", "T8",
    # central
    "C3", "Cz", "C4",
    # parietal
    "P3", "Pz", "P4", "P7", "P8",
    # occipital
    "O1", "Oz", "O2",
    # ear references
    "A1", "A2",
    # extended 10-10 system channels
    "AF3", "AF4", "AF7", "AF8", "AFz",
    "FC1", "FC2", "FC3", "FC4", "FC5", "FC6", "FCz",
    "CP1", "CP2", "CP3", "CP4", "CP5", "CP6", "CPz",
    "PO3", "PO4", "PO7", "PO8", "POz",
    "TP7", "TP8", "TP9", "TP10",
    "FT7", "FT8", "FT9", "FT10",
}

# common aliases and case variations
CHANNEL_ALIASES = {
    "fp1": "Fp1", "fp2": "Fp2", "fpz": "Fpz",
    "fz": "Fz", "cz": "Cz", "pz": "Pz", "oz": "Oz",
    # modern naming conventions
    "T7": "T3", "T8": "T4",
    "P7": "T5", "P8": "T6",
}

# generic channel names that we can auto-map to standard names
GENERIC_PATTERNS = {
    # brainflow synthetic board defaults
    "EEG1": "Fp1", "EEG2": "Fp2", "EEG3": "F3", "EEG4": "F4",
    "EEG5": "C3", "EEG6": "C4", "EEG7": "P3", "EEG8": "P4",
    "EEG9": "O1", "EEG10": "O2", "EEG11": "F7", "EEG12": "F8",
    "EEG13": "T3", "EEG14": "T4", "EEG15": "T5", "EEG16": "T6",
    # other common generic patterns
    "CH1": "Fp1", "CH2": "Fp2", "CH3": "F3", "CH4": "F4",
    "CH5": "C3", "CH6": "C4", "CH7": "P3", "CH8": "P4",
    "CHAN1": "Fp1", "CHAN2": "Fp2", "CHAN3": "F3", "CHAN4": "F4",
}

# columns to ignore (not eeg data, but not errors either)
IGNORED_COLUMNS = {
    "time", "timestamp", "sample", "index", "marker", "trigger",
    "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z",
    "battery", "counter", "event",
}


# error for when validation fails
class ValidationError(Exception):
    pass


# holds the result of validating eeg data
class ValidationResult:
    def __init__(self):
        self.is_valid = True
        self.channel_mapping = {}  # original_name -> standard_name
        self.valid_channels = []   # valid eeg channels found
        self.invalid_channels = [] # unrecognized columns
        self.ignored_columns = []  # non-eeg columns like time
        self.warnings = []
        self.errors = []

    def __repr__(self):
        return (
            f"ValidationResult(valid={self.is_valid}, "
            f"channels={len(self.valid_channels)}, "
            f"invalid={len(self.invalid_channels)})"
        )


# convert channel name to standard format
def normalize_channel_name(name: str) -> str:
    name = name.strip()

    # check aliases
    if name.lower() in CHANNEL_ALIASES:
        return CHANNEL_ALIASES[name.lower()]

    # check generic patterns
    if name.upper() in GENERIC_PATTERNS:
        return GENERIC_PATTERNS[name.upper()]

    # check if already valid (case insensitive)
    for standard in STANDARD_10_20_CHANNELS:
        if name.lower() == standard.lower():
            return standard

    return name


# check if a channel name is valid eeg
def is_valid_eeg_channel(name: str) -> bool:
    normalized = normalize_channel_name(name)
    return normalized in STANDARD_10_20_CHANNELS


# check if column should be ignored (like time, timestamp etc)
def is_ignored_column(name: str) -> bool:
    return name.lower().strip() in IGNORED_COLUMNS


# validate a list of column names against eeg standards
# strict mode will fail on any unrecognized columns
def validate_channels(columns: list[str], strict: bool = False) -> ValidationResult:
    result = ValidationResult()

    for col in columns:
        col_stripped = col.strip()

        # skip ignored columns
        if is_ignored_column(col_stripped):
            result.ignored_columns.append(col_stripped)
            continue

        # try to normalize to standard name
        normalized = normalize_channel_name(col_stripped)

        if normalized in STANDARD_10_20_CHANNELS:
            result.valid_channels.append(normalized)
            result.channel_mapping[col_stripped] = normalized
        else:
            result.invalid_channels.append(col_stripped)
            if strict:
                result.errors.append(f"Unrecognized channel: '{col_stripped}'")
            else:
                result.warnings.append(f"Unrecognized channel: '{col_stripped}' (will be skipped)")

    # figure out if overall valid
    if strict and result.invalid_channels:
        result.is_valid = False
    elif len(result.valid_channels) == 0:
        result.is_valid = False
        result.errors.append("No valid EEG channels found")
    else:
        result.is_valid = True

    return result


# validate eeg data including sample count and sample rate
def validate_eeg_data(
    columns: list[str],
    n_samples: int,
    sfreq: float,
    strict: bool = False,
) -> ValidationResult:
    result = validate_channels(columns, strict=strict)

    # check if we have enough samples (at least 1 second)
    min_samples = int(sfreq)
    if n_samples < min_samples:
        result.warnings.append(
            f"Very short recording: {n_samples} samples "
            f"({n_samples/sfreq:.2f}s at {sfreq}Hz)"
        )

    # check if sample rate is reasonable
    if sfreq < 64:
        result.warnings.append(f"Low sample rate: {sfreq}Hz (typical EEG is 128-512Hz)")
    elif sfreq > 2048:
        result.warnings.append(f"Very high sample rate: {sfreq}Hz (typical EEG is 128-512Hz)")

    # need at least 1 channel
    if len(result.valid_channels) < 1:
        result.is_valid = False
        result.errors.append("Need at least 1 valid EEG channel")

    return result


# get list of all standard channel names
def get_standard_channels() -> list[str]:
    return sorted(STANDARD_10_20_CHANNELS)


# suggest mappings for non-standard channel names
def suggest_channel_mapping(columns: list[str]) -> dict[str, str]:
    suggestions = {}

    for col in columns:
        if is_ignored_column(col):
            continue

        normalized = normalize_channel_name(col)
        if normalized != col and normalized in STANDARD_10_20_CHANNELS:
            suggestions[col] = normalized

    return suggestions


# generate a nice looking validation report
def print_validation_report(result: ValidationResult) -> str:
    lines = []
    lines.append("=" * 50)
    lines.append("EEG DATA VALIDATION REPORT")
    lines.append("=" * 50)

    status = "PASSED" if result.is_valid else "FAILED"
    lines.append(f"\nStatus: {status}")

    lines.append(f"\nChannels:")
    lines.append(f"  Valid EEG channels: {len(result.valid_channels)}")
    if result.valid_channels:
        lines.append(f"    {', '.join(result.valid_channels)}")

    if result.ignored_columns:
        lines.append(f"  Ignored columns: {len(result.ignored_columns)}")
        lines.append(f"    {', '.join(result.ignored_columns)}")

    if result.invalid_channels:
        lines.append(f"  Unrecognized: {len(result.invalid_channels)}")
        lines.append(f"    {', '.join(result.invalid_channels)}")

    if result.channel_mapping:
        remapped = [(k, v) for k, v in result.channel_mapping.items() if k != v]
        if remapped:
            lines.append(f"\nChannel mappings applied:")
            for orig, new in remapped:
                lines.append(f"    {orig} -> {new}")

    if result.warnings:
        lines.append(f"\nWarnings:")
        for w in result.warnings:
            lines.append(f"  - {w}")

    if result.errors:
        lines.append(f"\nErrors:")
        for e in result.errors:
            lines.append(f"  - {e}")

    lines.append("=" * 50)

    return "\n".join(lines)
