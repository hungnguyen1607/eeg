"""Tests for ML artifact detection module."""

import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path

# Skip all tests if sklearn not available
pytest.importorskip("sklearn")


class TestTrainingProducesValidProbabilities:
    """Test that training produces probabilities in [0, 1]."""

    def test_probabilities_in_valid_range(self):
        """Training should produce probabilities in [0, 1]."""
        from eeg_platform.ml.train_artifact import train_artifact_classifier
        from eeg_platform.ml.predict_artifact import predict_artifacts

        # Create synthetic data
        np.random.seed(42)
        n_samples = 100
        n_features = 5

        X = np.random.randn(n_samples, n_features)
        # Make labels correlate with first feature for learnable pattern
        y = (X[:, 0] > 0).astype(int)

        feature_names = [f"feature_{i}" for i in range(n_features)]

        with tempfile.TemporaryDirectory() as tmpdir:
            # Train model
            result = train_artifact_classifier(
                X, y, feature_names,
                output_dir=tmpdir,
                model_type="logreg",
            )

            # Get predictions
            probs, preds = predict_artifacts(X, tmpdir)

            # Check probabilities are in [0, 1]
            assert np.all(probs >= 0), "Probabilities should be >= 0"
            assert np.all(probs <= 1), "Probabilities should be <= 1"

            # Check predictions are binary
            assert set(np.unique(preds)).issubset({0, 1}), "Predictions should be 0 or 1"


class TestEvalReturnsF1AboveChance:
    """Test that evaluation returns F1 above chance on synthetic data."""

    def test_f1_above_chance_on_learnable_data(self):
        """F1 should be > 0.5 (chance) on learnable synthetic data."""
        from eeg_platform.ml.train_artifact import train_artifact_classifier
        from eeg_platform.ml.eval_artifact import evaluate_artifact_model

        np.random.seed(42)
        n_samples = 200
        n_features = 5

        # Create clearly separable classes
        X_clean = np.random.randn(n_samples // 2, n_features)
        X_clean[:, 0] -= 2  # Shift clean samples

        X_artifact = np.random.randn(n_samples // 2, n_features)
        X_artifact[:, 0] += 2  # Shift artifact samples

        X = np.vstack([X_clean, X_artifact])
        y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

        # Shuffle
        perm = np.random.permutation(n_samples)
        X = X[perm]
        y = y[perm]

        feature_names = [f"feature_{i}" for i in range(n_features)]

        with tempfile.TemporaryDirectory() as tmpdir:
            # Train
            result = train_artifact_classifier(
                X, y, feature_names,
                output_dir=tmpdir,
                model_type="rf",
                test_size=0.3,
            )

            # Evaluate on test set from training
            metrics = evaluate_artifact_model(
                result["model"],
                result["scaler"],
                result["X_test"],
                result["y_test"],
            )

            # F1 should be significantly above chance (0.5)
            assert metrics["f1"] > 0.5, f"F1 should be > 0.5, got {metrics['f1']}"

            # Also check other metrics are reasonable
            assert metrics["accuracy"] > 0.5, "Accuracy should be > 0.5"
            assert metrics["precision"] >= 0, "Precision should be >= 0"
            assert metrics["recall"] >= 0, "Recall should be >= 0"


class TestImprovementMetricsValid:
    """Test that improvement metrics JSON contains required keys."""

    def test_improvement_metrics_has_required_keys(self):
        """improvement_metrics.json should have required keys and valid ranges."""
        from eeg_platform.ml.improvement import compute_improvement_metrics

        # Create synthetic before/after DataFrames
        np.random.seed(42)

        # "Before" has some extreme values
        df_before = pd.DataFrame({
            "ptp_uv": np.concatenate([
                np.random.uniform(50, 100, 80),  # Normal
                np.random.uniform(200, 300, 20),  # Extreme (will be rejected)
            ]),
            "rms_uv": np.concatenate([
                np.random.uniform(10, 30, 80),
                np.random.uniform(100, 150, 20),
            ]),
            "alpha_power": np.random.uniform(0.1, 1.0, 100),
            "beta_power": np.random.uniform(0.05, 0.5, 100),
            "kurtosis": np.concatenate([
                np.random.uniform(-2, 2, 80),
                np.random.uniform(15, 25, 20),  # Extreme kurtosis
            ]),
        })

        # "After" has only the clean windows (first 80)
        df_after = df_before.iloc[:80].copy()

        # Compute improvement
        metrics = compute_improvement_metrics(df_before, df_after)

        # Check required top-level keys
        assert "window_counts" in metrics
        assert "variance_reduction" in metrics
        assert "extreme_window_reduction" in metrics
        assert "variance_reduction_pct" in metrics
        assert "extreme_window_reduction_pct" in metrics

        # Check window counts
        counts = metrics["window_counts"]
        assert counts["total_windows"] == 100
        assert counts["clean_windows"] == 80
        assert counts["rejected_windows"] == 20
        assert 0 <= counts["rejection_rate"] <= 1

        # Check variance reduction is computed
        var_reduction = metrics["variance_reduction"]
        assert "median_variance_reduction_pct" in var_reduction
        assert "per_feature" in var_reduction

        # Check extreme window reduction
        extreme = metrics["extreme_window_reduction"]
        assert "overall_extreme_reduction_pct" in extreme

        # Values should be reasonable percentages
        assert -100 <= metrics["variance_reduction_pct"] <= 100
        assert 0 <= metrics["extreme_window_reduction_pct"] <= 100


class TestWeakLabelGeneration:
    """Test weak label generation."""

    def test_weak_labels_binary(self):
        """Weak labels should be 0 or 1."""
        from eeg_platform.ml.artifact_labels import generate_weak_labels

        df = pd.DataFrame({
            "ptp_uv": [50, 200, 75, 180, 60],
            "rms_uv": [20, 100, 25, 90, 22],
            "kurtosis": [1, 15, 0.5, 12, 0.8],
        })

        labels = generate_weak_labels(df)

        assert len(labels) == 5
        assert set(np.unique(labels)).issubset({0, 1})

    def test_weak_labels_detect_artifacts(self):
        """Weak labeling should mark extreme values as artifacts."""
        from eeg_platform.ml.artifact_labels import generate_weak_labels

        df = pd.DataFrame({
            "ptp_uv": [50, 300, 60],  # Second value is extreme
            "rms_uv": [20, 150, 25],  # Second value is extreme
            "kurtosis": [1, 1, 1],
        })

        labels = generate_weak_labels(df, strict=False)

        # The middle sample (index 1) should be labeled as artifact
        assert labels[1] == 1, "Extreme sample should be labeled as artifact"


class TestFeatureSelection:
    """Test feature selection from DataFrames."""

    def test_feature_selection_handles_missing(self):
        """Should handle missing columns gracefully."""
        from eeg_platform.ml.featureset import select_ml_features

        df = pd.DataFrame({
            "rms_uv": [1, 2, 3],
            "ptp_uv": [10, 20, 30],
            "kurtosis": [0.1, 0.2, 0.3],
        })

        X, feature_names = select_ml_features(df, min_required=2)

        assert X.shape[0] == 3
        assert X.shape[1] >= 2
        assert len(feature_names) >= 2


class TestBenchmark:
    """Test benchmarking functionality."""

    def test_benchmark_returns_throughput_metrics(self):
        """Benchmark should return throughput metrics."""
        from eeg_platform.ml.benchmark import run_benchmark

        np.random.seed(42)
        data = np.random.randn(3, 2560)  # 3 channels, 10 seconds at 256 Hz
        sfreq = 256.0

        result = run_benchmark(
            data, sfreq,
            window_sec=2.0,
            overlap=0.5,
            n_iterations=1,
            include_ml=False,
        )

        # Check required keys
        assert "throughput" in result
        assert "windows_per_sec" in result["throughput"]
        assert "realtime_factor" in result["throughput"]

        # Throughput should be positive
        assert result["throughput"]["windows_per_sec"] > 0
        assert result["throughput"]["realtime_factor"] > 0

        # Timing breakdown should exist
        assert "timing_mean_sec" in result
        assert "preprocess" in result["timing_mean_sec"]
        assert "feature_extraction" in result["timing_mean_sec"]
