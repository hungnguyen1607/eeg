# filtering and cleaning eeg signals

from .clean import (
    remove_dc_offset,
    normalize_channels,
    robust_zscore,
    clean_signal,
)
from .filters import (
    bandpass_filter,
    lowpass_filter,
    highpass_filter,
    notch_filter,
    FilterError,
)
from .windowing import (
    segment_into_windows,
    window_iterator,
    get_window_count,
)

__all__ = [
    "remove_dc_offset",
    "normalize_channels",
    "robust_zscore",
    "clean_signal",
    "bandpass_filter",
    "lowpass_filter",
    "highpass_filter",
    "notch_filter",
    "FilterError",
    "segment_into_windows",
    "window_iterator",
    "get_window_count",
]
