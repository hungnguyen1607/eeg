"""Tests for time-domain feature extraction."""

import numpy as np
import pytest

from eeg_platform.features.time_domain import (
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


class TestComputeMean:
    """Tests for compute_mean function."""

    def test_output_shape(self):
        """Should return one value per channel."""
        data = np.random.randn(3, 1000)
        result = compute_mean(data)
        assert result.shape == (3,)

    def test_known_value(self):
        """Should compute correct mean."""
        data = np.array([[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]])
        result = compute_mean(data)
        np.testing.assert_array_almost_equal(result, [3.0, 30.0])


class TestComputeStd:
    """Tests for compute_std function."""

    def test_output_shape(self):
        """Should return one value per channel."""
        data = np.random.randn(3, 1000)
        result = compute_std(data)
        assert result.shape == (3,)

    def test_known_value(self):
        """Should compute correct std."""
        # Std of [1,2,3,4,5] = sqrt(2) ≈ 1.4142
        data = np.array([[1, 2, 3, 4, 5]])
        result = compute_std(data)
        expected = np.std([1, 2, 3, 4, 5])
        np.testing.assert_almost_equal(result[0], expected)


class TestComputeRms:
    """Tests for compute_rms function."""

    def test_output_shape(self):
        """Should return one value per channel."""
        data = np.random.randn(4, 500)
        result = compute_rms(data)
        assert result.shape == (4,)

    def test_known_signal(self):
        """RMS of sine wave with amplitude A should be A/sqrt(2)."""
        sfreq = 256.0
        duration = 10.0
        n_samples = int(sfreq * duration)
        t = np.arange(n_samples) / sfreq

        amplitude = 10.0
        signal = amplitude * np.sin(2 * np.pi * 5 * t)
        data = signal.reshape(1, -1)

        result = compute_rms(data)

        expected_rms = amplitude / np.sqrt(2)
        np.testing.assert_almost_equal(result[0], expected_rms, decimal=2)

    def test_dc_signal(self):
        """RMS of constant signal equals the constant."""
        data = np.full((2, 100), fill_value=5.0)
        result = compute_rms(data)
        np.testing.assert_array_almost_equal(result, [5.0, 5.0])


class TestComputePeakToPeak:
    """Tests for compute_peak_to_peak function."""

    def test_output_shape(self):
        """Should return one value per channel."""
        data = np.random.randn(3, 1000)
        result = compute_peak_to_peak(data)
        assert result.shape == (3,)

    def test_known_signal(self):
        """Peak-to-peak of sine wave with amplitude A should be 2A."""
        sfreq = 256.0
        duration = 10.0
        n_samples = int(sfreq * duration)
        t = np.arange(n_samples) / sfreq

        amplitude = 7.5
        signal = amplitude * np.sin(2 * np.pi * 3 * t)
        data = signal.reshape(1, -1)

        result = compute_peak_to_peak(data)

        expected_ptp = 2 * amplitude
        np.testing.assert_almost_equal(result[0], expected_ptp, decimal=2)

    def test_simple_array(self):
        """Peak-to-peak of [1, 5, 2] should be 4."""
        data = np.array([[1, 5, 2], [0, 10, 5]])
        result = compute_peak_to_peak(data)
        np.testing.assert_array_equal(result, [4, 10])


class TestComputeVariance:
    """Tests for compute_variance function."""

    def test_output_shape(self):
        """Should return one value per channel."""
        data = np.random.randn(2, 500)
        result = compute_variance(data)
        assert result.shape == (2,)

    def test_non_negative(self):
        """Variance should be non-negative."""
        data = np.random.randn(3, 1000)
        result = compute_variance(data)
        assert np.all(result >= 0)


class TestComputeSkewness:
    """Tests for compute_skewness function."""

    def test_output_shape(self):
        """Should return one value per channel."""
        data = np.random.randn(3, 1000)
        result = compute_skewness(data)
        assert result.shape == (3,)

    def test_symmetric_distribution(self):
        """Symmetric distribution should have near-zero skewness."""
        # Normal distribution is symmetric
        np.random.seed(42)
        data = np.random.randn(1, 10000)
        result = compute_skewness(data)
        # Should be close to 0
        assert abs(result[0]) < 0.1


class TestComputeKurtosis:
    """Tests for compute_kurtosis function."""

    def test_output_shape(self):
        """Should return one value per channel."""
        data = np.random.randn(3, 1000)
        result = compute_kurtosis(data)
        assert result.shape == (3,)

    def test_normal_distribution(self):
        """Normal distribution should have near-zero excess kurtosis."""
        np.random.seed(42)
        data = np.random.randn(1, 10000)
        result = compute_kurtosis(data)
        # Excess kurtosis of normal is 0
        assert abs(result[0]) < 0.2


class TestComputeZeroCrossings:
    """Tests for compute_zero_crossings function."""

    def test_output_shape(self):
        """Should return one value per channel."""
        data = np.random.randn(3, 1000)
        result = compute_zero_crossings(data)
        assert result.shape == (3,)

    def test_known_crossings(self):
        """Should count zero crossings correctly."""
        # Signal: [1, -1, 1, -1, 1] has 4 zero crossings
        data = np.array([[1, -1, 1, -1, 1]])
        result = compute_zero_crossings(data)
        assert result[0] == 4

    def test_no_crossings(self):
        """All positive signal should have 0 crossings."""
        data = np.array([[1, 2, 3, 4, 5]])
        result = compute_zero_crossings(data)
        assert result[0] == 0


class TestComputeMinMax:
    """Tests for compute_min and compute_max functions."""

    def test_known_values(self):
        """Should find correct min and max."""
        data = np.array([[1, 5, 2, 8, 3], [-10, 0, 10, 5, -5]])

        min_vals = compute_min(data)
        max_vals = compute_max(data)

        np.testing.assert_array_equal(min_vals, [1, -10])
        np.testing.assert_array_equal(max_vals, [8, 10])


class TestExtractTimeFeatures:
    """Tests for extract_time_features function."""

    def test_returns_all_features(self):
        """Should return all expected features."""
        data = np.random.randn(3, 1000)
        features = extract_time_features(data)

        expected_keys = {
            "mean", "std", "rms", "variance", "peak_to_peak",
            "min", "max", "zero_crossings", "skewness", "kurtosis"
        }
        assert set(features.keys()) == expected_keys

    def test_all_shapes_correct(self):
        """All feature arrays should have shape (n_channels,)."""
        n_channels = 5
        data = np.random.randn(n_channels, 1000)
        features = extract_time_features(data)

        for name, values in features.items():
            assert values.shape == (n_channels,), f"{name} has wrong shape"

    def test_all_finite(self):
        """All feature values should be finite."""
        data = np.random.randn(3, 1000)
        features = extract_time_features(data)

        for name, values in features.items():
            assert np.all(np.isfinite(values)), f"{name} has non-finite values"


class TestKnownSignalFeatures:
    """Tests with known signals to verify feature values."""

    @pytest.fixture
    def sine_wave_data(self):
        """Create a known sine wave."""
        sfreq = 256.0
        duration = 10.0
        n_samples = int(sfreq * duration)
        t = np.arange(n_samples) / sfreq

        amplitude = 10.0
        frequency = 5.0
        signal = amplitude * np.sin(2 * np.pi * frequency * t)
        return signal.reshape(1, -1), amplitude

    def test_sine_rms(self, sine_wave_data):
        """Sine wave RMS should be amplitude / sqrt(2)."""
        data, amplitude = sine_wave_data
        rms = compute_rms(data)
        expected = amplitude / np.sqrt(2)
        np.testing.assert_almost_equal(rms[0], expected, decimal=2)

    def test_sine_peak_to_peak(self, sine_wave_data):
        """Sine wave peak-to-peak should be 2 * amplitude."""
        data, amplitude = sine_wave_data
        ptp = compute_peak_to_peak(data)
        expected = 2 * amplitude
        np.testing.assert_almost_equal(ptp[0], expected, decimal=2)

    def test_sine_mean_near_zero(self, sine_wave_data):
        """Sine wave mean should be near zero."""
        data, _ = sine_wave_data
        mean = compute_mean(data)
        np.testing.assert_almost_equal(mean[0], 0.0, decimal=2)

    def test_sine_min_max(self, sine_wave_data):
        """Sine wave min/max should be -amplitude/+amplitude."""
        data, amplitude = sine_wave_data
        min_val = compute_min(data)
        max_val = compute_max(data)
        np.testing.assert_almost_equal(min_val[0], -amplitude, decimal=2)
        np.testing.assert_almost_equal(max_val[0], amplitude, decimal=2)
