

import numpy as np
import pytest

from eeg_platform.preprocess.filters import (
    bandpass_filter,
    notch_filter,
    highpass_filter,
    lowpass_filter,
    FilterError,
)


class TestBandpassFilter:
    """Tests for bandpass_filter function."""

    def test_output_shape(self):
        """Output shape should match input shape."""
        sfreq = 256.0
        n_channels, n_samples = 3, 1000
        data = np.random.randn(n_channels, n_samples)

        filtered = bandpass_filter(data, sfreq, l_freq=1.0, h_freq=40.0)

        assert filtered.shape == data.shape

    def test_output_dtype(self):
        """Output should be float64."""
        sfreq = 256.0
        data = np.random.randn(2, 500).astype(np.float32)

        filtered = bandpass_filter(data, sfreq, l_freq=1.0, h_freq=40.0)

        assert filtered.dtype == np.float64

    def test_removes_low_frequencies(self):
        """Bandpass should attenuate frequencies below l_freq."""
        sfreq = 256.0
        n_samples = 2048
        t = np.arange(n_samples) / sfreq

        #testing signal with 0.5 hz 
        low_freq_signal = np.sin(2 * np.pi * 0.5 * t)
        data = low_freq_signal.reshape(1, -1)

        filtered = bandpass_filter(data, sfreq, l_freq=1.0, h_freq=40.0)

        # reducing low frequency components
        assert np.std(filtered) < np.std(data) * 0.5

    def test_removes_high_frequencies(self):
        """Bandpass should attenuate frequencies above h_freq."""
        sfreq = 256.0
        n_samples = 2048
        t = np.arange(n_samples) / sfreq

        # testing signal with 80 Hz component 
        high_freq_signal = np.sin(2 * np.pi * 80 * t)
        data = high_freq_signal.reshape(1, -1)

        filtered = bandpass_filter(data, sfreq, l_freq=1.0, h_freq=40.0)

        # hgih frequency components should be reduced here
        assert np.std(filtered) < np.std(data) * 0.5

    def test_preserves_passband_frequencies(self):
        """Bandpass should preserve frequencies in passband."""
        sfreq = 256.0
        n_samples = 2048
        t = np.arange(n_samples) / sfreq

        # makign signals at 10hz here
        passband_signal = np.sin(2 * np.pi * 10 * t)
        data = passband_signal.reshape(1, -1)

        filtered = bandpass_filter(data, sfreq, l_freq=1.0, h_freq=40.0)

        # passband signal should be largely preserved
        assert np.std(filtered) > np.std(data) * 0.7

    def test_invalid_frequency_order(self):
        """Should raise error if l_freq >= h_freq."""
        data = np.random.randn(2, 500)

        with pytest.raises(FilterError, match="must be less than"):
            bandpass_filter(data, sfreq=256.0, l_freq=40.0, h_freq=10.0)

    def test_frequency_exceeds_nyquist(self):
        """Should raise error if frequency exceeds Nyquist."""
        data = np.random.randn(2, 500)

        with pytest.raises(FilterError, match="Nyquist"):
            bandpass_filter(data, sfreq=100.0, l_freq=1.0, h_freq=60.0)

    def test_negative_frequency(self):
        """Should raise error for negative frequencies."""
        data = np.random.randn(2, 500)

        with pytest.raises(FilterError, match="positive"):
            bandpass_filter(data, sfreq=256.0, l_freq=-1.0, h_freq=40.0)


class TestNotchFilter:
    """Tests for notch_filter function."""

    def test_output_shape(self):
        """Output shape should match input shape."""
        sfreq = 256.0
        n_channels, n_samples = 3, 1000
        data = np.random.randn(n_channels, n_samples)

        filtered = notch_filter(data, sfreq, notch_freq=60.0)

        assert filtered.shape == data.shape

    def test_removes_notch_frequency(self):
        """Notch filter should attenuate the target frequency."""
        sfreq = 256.0
        n_samples = 2048
        t = np.arange(n_samples) / sfreq

        # 60hz signals here
        noise_signal = np.sin(2 * np.pi * 60 * t)
        data = noise_signal.reshape(1, -1)

        filtered = notch_filter(data, sfreq, notch_freq=60.0)

        # altering 60hz components
        assert np.std(filtered) < np.std(data) * 0.3

    def test_preserves_other_frequencies(self):
        """Notch filter should preserve frequencies away from notch."""
        sfreq = 256.0
        n_samples = 2048
        t = np.arange(n_samples) / sfreq

        # making 10hz signals from 60hz here
        signal = np.sin(2 * np.pi * 10 * t)
        data = signal.reshape(1, -1)

        filtered = notch_filter(data, sfreq, notch_freq=60.0)

        # 10 Hz signal should be preserved
        assert np.std(filtered) > np.std(data) * 0.9

    def test_frequency_exceeds_nyquist(self):
        """Should raise error if notch frequency exceeds Nyquist."""
        data = np.random.randn(2, 500)

        with pytest.raises(FilterError, match="Nyquist"):
            notch_filter(data, sfreq=100.0, notch_freq=60.0)


class TestHighpassFilter:
    """Tests for highpass_filter function."""

    def test_output_shape(self):
        """Output shape should match input shape."""
        sfreq = 256.0
        data = np.random.randn(3, 1000)

        filtered = highpass_filter(data, sfreq, cutoff=1.0)

        assert filtered.shape == data.shape

    def test_removes_dc_and_low_frequencies(self):
        """Highpass should remove DC offset and low frequencies."""
        sfreq = 256.0
        n_samples = 2048
        t = np.arange(n_samples) / sfreq

        # create signals with direct current offset and low freqeuncy
        dc_offset = 100.0
        low_freq = np.sin(2 * np.pi * 0.1 * t)  # 0.1 hz for example
        data = (dc_offset + low_freq).reshape(1, -1)

        filtered = highpass_filter(data, sfreq, cutoff=1.0)

        # removing them here
        assert np.abs(np.mean(filtered)) < 1.0


class TestLowpassFilter:
    """Tests for lowpass_filter function."""

    def test_output_shape(self):
        """Output shape should match input shape."""
        sfreq = 256.0
        data = np.random.randn(3, 1000)

        filtered = lowpass_filter(data, sfreq, cutoff=40.0)

        assert filtered.shape == data.shape

    def test_removes_high_frequencies(self):
        """Lowpass should attenuate high frequencies."""
        sfreq = 256.0
        n_samples = 2048
        t = np.arange(n_samples) / sfreq

        # making high frequency at 80hz here
        high_freq = np.sin(2 * np.pi * 80 * t)
        data = high_freq.reshape(1, -1)

        filtered = lowpass_filter(data, sfreq, cutoff=40.0)

        # cutting them down abit
        assert np.std(filtered) < np.std(data) * 0.5


class TestZeroPhaseFiltering:
    """Tests to verify zero-phase filtering behavior."""

    def test_no_phase_shift(self):
        """Zero-phase filtering should not introduce phase shift."""
        sfreq = 256.0
        n_samples = 2048
        t = np.arange(n_samples) / sfreq

        # making a clean sine wave here
        freq = 10.0  # Hz
        original = np.sin(2 * np.pi * freq * t)
        data = original.reshape(1, -1)

        filtered = bandpass_filter(data, sfreq, l_freq=1.0, h_freq=40.0)

       
        trim = 200
        orig_trim = original[trim:-trim]
        filt_trim = filtered[0, trim:-trim]

       
        orig_norm = orig_trim / np.std(orig_trim)
        filt_norm = filt_trim / np.std(filt_trim)

       
        correlation = np.correlate(orig_norm, filt_norm, mode="full")
        lag = np.argmax(correlation) - (len(orig_norm) - 1)

        
        assert abs(lag) <= 2, f"Phase shift detected: lag={lag} samples"
