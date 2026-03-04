"""Tests for spectral feature extraction."""

import numpy as np
import pytest

from eeg_platform.features.spectral import (
    compute_psd_welch,
    bandpower,
    dominant_frequency,
    bandpower_from_data,
    compute_all_bandpowers,
    relative_bandpower,
    EEG_BANDS,
)


class TestComputePsdWelch:
    """Tests for compute_psd_welch function."""

    def test_output_shapes(self):
        """PSD output should have correct shapes."""
        sfreq = 256.0
        n_channels, n_samples = 3, 2048
        data = np.random.randn(n_channels, n_samples)

        freqs, psd = compute_psd_welch(data, sfreq, nperseg=512)

       
        assert freqs.ndim == 1
      
        assert psd.shape[0] == n_channels
        assert psd.shape[1] == len(freqs)

    def test_frequency_range(self):
        """Frequencies should range from 0 to Nyquist."""
        sfreq = 256.0
        data = np.random.randn(2, 2048)

        freqs, psd = compute_psd_welch(data, sfreq)

        assert freqs[0] == 0.0
        assert freqs[-1] <= sfreq / 2

    def test_psd_non_negative(self):
        """PSD values should be non-negative."""
        sfreq = 256.0
        data = np.random.randn(2, 2048)

        freqs, psd = compute_psd_welch(data, sfreq)

        assert np.all(psd >= 0)

    def test_short_data_adjustment(self):
        """nperseg should adjust for short data."""
        sfreq = 256.0
        n_samples = 100  # shorter than default nperseg=512
        data = np.random.randn(2, n_samples)

        # Should not raise error
        freqs, psd = compute_psd_welch(data, sfreq, nperseg=512)

        assert len(freqs) > 0
        assert psd.shape[1] == len(freqs)


class TestBandpower:
    """Tests for bandpower function."""

    def test_output_shape(self):
        """Bandpower should return one value per channel."""
        sfreq = 256.0
        n_channels = 4
        data = np.random.randn(n_channels, 2048)
        freqs, psd = compute_psd_welch(data, sfreq)

        power = bandpower(psd, freqs, (8.0, 13.0))

        assert power.shape == (n_channels,)

    def test_non_negative(self):
        """Band power should be non-negative."""
        sfreq = 256.0
        data = np.random.randn(2, 2048)
        freqs, psd = compute_psd_welch(data, sfreq)

        for band_range in EEG_BANDS.values():
            power = bandpower(psd, freqs, band_range)
            assert np.all(power >= 0)

    def test_empty_band_returns_zero(self):
        """Band outside frequency range should return zero."""
        sfreq = 100.0  # Nyquist = 50 Hz
        data = np.random.randn(2, 1024)
        freqs, psd = compute_psd_welch(data, sfreq)

        # Band above Nyquist
        power = bandpower(psd, freqs, (60.0, 80.0))

        assert np.all(power == 0)


class TestDominantFrequency:
    """Tests for dominant_frequency function."""

    def test_output_shape(self):
        """Should return one frequency per channel."""
        sfreq = 256.0
        n_channels = 3
        data = np.random.randn(n_channels, 2048)
        freqs, psd = compute_psd_welch(data, sfreq)

        dom_freqs = dominant_frequency(psd, freqs)

        assert dom_freqs.shape == (n_channels,)

    def test_within_range(self):
        """Dominant frequency should be within fmin-fmax range."""
        sfreq = 256.0
        data = np.random.randn(2, 2048)
        freqs, psd = compute_psd_welch(data, sfreq)

        fmin, fmax = 5.0, 30.0
        dom_freqs = dominant_frequency(psd, freqs, fmin=fmin, fmax=fmax)

        assert np.all(dom_freqs >= fmin)
        assert np.all(dom_freqs <= fmax)


class TestPureSineWave:
    """Tests using a pure 10 Hz sine wave."""

    @pytest.fixture
    def sine_10hz_data(self):
        """Create a pure 10 Hz sine wave."""
        sfreq = 256.0
        duration = 10.0  # 10 seconds for good frequency resolution
        n_samples = int(sfreq * duration)
        t = np.arange(n_samples) / sfreq

        # 10 Hz sine wave
        signal = np.sin(2 * np.pi * 10.0 * t)

        # Shape: (1 channel, n_samples)
        data = signal.reshape(1, -1)
        return data, sfreq

    def test_dominant_frequency_10hz(self, sine_10hz_data):
        """Pure 10 Hz sine should have dominant frequency ~10 Hz."""
        data, sfreq = sine_10hz_data
        freqs, psd = compute_psd_welch(data, sfreq, nperseg=512)

        dom_freq = dominant_frequency(psd, freqs, fmin=1.0, fmax=45.0)

        # Should be within 1 Hz of 10 Hz
        assert abs(dom_freq[0] - 10.0) < 1.0, f"Expected ~10 Hz, got {dom_freq[0]} Hz"

    def test_high_alpha_bandpower(self, sine_10hz_data):
        """Pure 10 Hz sine should have high alpha (8-13 Hz) power."""
        data, sfreq = sine_10hz_data
        freqs, psd = compute_psd_welch(data, sfreq, nperseg=512)

        alpha_power = bandpower(psd, freqs, EEG_BANDS["alpha"])
        delta_power = bandpower(psd, freqs, EEG_BANDS["delta"])
        theta_power = bandpower(psd, freqs, EEG_BANDS["theta"])
        beta_power = bandpower(psd, freqs, EEG_BANDS["beta"])
        gamma_power = bandpower(psd, freqs, EEG_BANDS["gamma"])

        # Alpha should be much higher than other bands
        assert alpha_power[0] > delta_power[0] * 10, "Alpha should dominate delta"
        assert alpha_power[0] > theta_power[0] * 10, "Alpha should dominate theta"
        assert alpha_power[0] > beta_power[0] * 10, "Alpha should dominate beta"
        assert alpha_power[0] > gamma_power[0] * 10, "Alpha should dominate gamma"

    def test_relative_alpha_bandpower(self, sine_10hz_data):
        """Pure 10 Hz sine should have relative alpha > 0.9."""
        data, sfreq = sine_10hz_data
        freqs, psd = compute_psd_welch(data, sfreq, nperseg=512)

        rel_alpha = relative_bandpower(psd, freqs, EEG_BANDS["alpha"])

        # Relative alpha should be > 90% of total power
        assert rel_alpha[0] > 0.9, f"Expected rel_alpha > 0.9, got {rel_alpha[0]}"


class TestBandpowerFromData:
    """Tests for bandpower_from_data convenience function."""

    def test_with_band_name(self):
        """Should accept band names like 'alpha'."""
        sfreq = 256.0
        data = np.random.randn(2, 2048)

        power = bandpower_from_data(data, sfreq, "alpha")

        assert power.shape == (2,)
        assert np.all(power >= 0)

    def test_with_tuple(self):
        """Should accept (low, high) tuples."""
        sfreq = 256.0
        data = np.random.randn(2, 2048)

        power = bandpower_from_data(data, sfreq, (8.0, 13.0))

        assert power.shape == (2,)
        assert np.all(power >= 0)

    def test_invalid_band_name(self):
        """Should raise error for unknown band name."""
        sfreq = 256.0
        data = np.random.randn(2, 2048)

        with pytest.raises(ValueError, match="Unknown band"):
            bandpower_from_data(data, sfreq, "invalid_band")


class TestComputeAllBandpowers:
    """Tests for compute_all_bandpowers function."""

    def test_returns_all_bands(self):
        """Should return power for all EEG bands."""
        sfreq = 256.0
        data = np.random.randn(3, 2048)

        powers = compute_all_bandpowers(data, sfreq)

        assert set(powers.keys()) == set(EEG_BANDS.keys())
        for band_name in EEG_BANDS:
            assert powers[band_name].shape == (3,)
            assert np.all(powers[band_name] >= 0)


class TestFiniteValues:
    """Tests to ensure outputs are finite."""

    def test_psd_finite(self):
        """PSD values should be finite."""
        sfreq = 256.0
        data = np.random.randn(2, 2048)
        freqs, psd = compute_psd_welch(data, sfreq)

        assert np.all(np.isfinite(freqs))
        assert np.all(np.isfinite(psd))

    def test_bandpower_finite(self):
        """Band power values should be finite."""
        sfreq = 256.0
        data = np.random.randn(2, 2048)
        freqs, psd = compute_psd_welch(data, sfreq)

        for band_range in EEG_BANDS.values():
            power = bandpower(psd, freqs, band_range)
            assert np.all(np.isfinite(power))

    def test_dominant_frequency_finite(self):
        """Dominant frequency values should be finite."""
        sfreq = 256.0
        data = np.random.randn(2, 2048)
        freqs, psd = compute_psd_welch(data, sfreq)

        dom_freqs = dominant_frequency(psd, freqs)

        assert np.all(np.isfinite(dom_freqs))
