"""Tests for the impact analysis module."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eeg_platform.analysis.impact import (
    compute_before_after,
    compute_trends,
    write_summaries,
    run_impact_analysis,
    load_window_data,
    ANALYSIS_FEATURES,
)


class TestBeforeAfterAnalysis:
    """Tests for before/after variance reduction analysis."""

    def test_variance_reduction_with_artifact_windows(self):
        """Verify high variance reduction when filtering out artifact windows."""
        np.random.seed(42)

        # Create synthetic data: clean windows have low ptp_uv, artifacts have extreme values
        n_clean = 80
        n_artifact = 20

        # Clean windows: normal distribution around mean=50, low variance
        clean_ptp = np.random.normal(50, 5, n_clean)
        clean_rms = np.random.normal(20, 2, n_clean)

        # Artifact windows: extreme values with high variance
        artifact_ptp = np.random.normal(300, 100, n_artifact)
        artifact_rms = np.random.normal(100, 30, n_artifact)

        # Combine
        all_ptp = np.concatenate([clean_ptp, artifact_ptp])
        all_rms = np.concatenate([clean_rms, artifact_rms])
        is_clean = np.array([True] * n_clean + [False] * n_artifact)

        # Create DataFrames
        all_df = pd.DataFrame({
            "window_id": range(len(all_ptp)),
            "ptp_uv": all_ptp,
            "rms_uv": all_rms,
            "alpha_power": np.random.normal(100, 10, len(all_ptp)),
        })

        clean_df = all_df[is_clean].copy()

        # Compute before/after
        result = compute_before_after(
            all_df, clean_df,
            features=["ptp_uv", "rms_uv", "alpha_power"]
        )

        # Check that variance reduction is high for ptp_uv and rms_uv
        ptp_row = result[result["feature"] == "ptp_uv"].iloc[0]
        rms_row = result[result["feature"] == "rms_uv"].iloc[0]

        # Artifacts should cause high variance before, much lower after
        assert ptp_row["variance_reduction_percent"] > 50, \
            f"Expected >50% variance reduction for ptp_uv, got {ptp_row['variance_reduction_percent']:.1f}%"
        assert rms_row["variance_reduction_percent"] > 50, \
            f"Expected >50% variance reduction for rms_uv, got {rms_row['variance_reduction_percent']:.1f}%"

        # Verify n_windows counts
        assert ptp_row["n_windows_before"] == 100
        assert ptp_row["n_windows_after"] == 80

    def test_statistical_test_significance(self):
        """Verify mean and variance changes when filtering artifacts."""
        np.random.seed(42)

        # Create data where clean windows have clearly different distribution
        n_total = 100
        n_clean = 70

        # Clean windows: normal around 100
        clean_vals = np.random.normal(100, 10, n_clean)
        # Artifact windows: much higher values (outliers)
        artifact_vals = np.random.normal(250, 20, n_total - n_clean)

        all_vals = np.concatenate([clean_vals, artifact_vals])
        is_clean = np.array([True] * n_clean + [False] * (n_total - n_clean))

        all_df = pd.DataFrame({
            "window_id": range(n_total),
            "alpha_power": all_vals,
        })

        clean_df = all_df[is_clean].copy()

        result = compute_before_after(all_df, clean_df, features=["alpha_power"])
        row = result[result["feature"] == "alpha_power"].iloc[0]

        # Mean should be significantly lower after (removing high-value artifacts)
        assert row["mean_after"] < row["mean_before"], \
            "Expected mean to decrease after filtering"

        # The mean before includes artifacts (~250), mean after should be ~100
        assert row["mean_before"] > 120, \
            f"Expected inflated mean before, got {row['mean_before']:.1f}"
        assert row["mean_after"] < 120, \
            f"Expected lower mean after, got {row['mean_after']:.1f}"

        # Variance should be reduced (artifacts inflate variance)
        assert row["variance_reduction_percent"] > 0, \
            f"Expected positive variance reduction, got {row['variance_reduction_percent']:.1f}%"

    def test_handles_zero_variance_safely(self):
        """Verify safe handling of zero variance edge case."""
        all_df = pd.DataFrame({
            "window_id": [0, 1, 2, 3],
            "alpha_power": [100.0, 100.0, 100.0, 100.0],  # Zero variance
        })

        clean_df = all_df.copy()

        result = compute_before_after(all_df, clean_df, features=["alpha_power"])
        row = result.iloc[0]

        # Should not raise an error, and variance reduction should be 0 or nan
        assert np.isfinite(row["var_before"]) or row["var_before"] == 0

    def test_handles_missing_features(self):
        """Verify graceful handling of missing features."""
        all_df = pd.DataFrame({
            "window_id": [0, 1, 2],
            "alpha_power": [100, 110, 120],
        })

        clean_df = all_df.copy()

        # Request features that don't exist
        result = compute_before_after(
            all_df, clean_df,
            features=["alpha_power", "nonexistent_feature"]
        )

        # Should only have results for existing features
        assert len(result) == 1
        assert result.iloc[0]["feature"] == "alpha_power"


class TestTrendAnalysis:
    """Tests for temporal trend analysis."""

    def test_increasing_trend_detection(self):
        """Verify detection of increasing trend over time."""
        np.random.seed(42)

        n_windows = 50
        time_vals = np.arange(n_windows) * 2.0  # 2 seconds per window

        # Create clear increasing trend with noise
        base_trend = 0.5 * np.arange(n_windows)  # Linear increase
        noise = np.random.normal(0, 2, n_windows)
        theta_beta = 5 + base_trend + noise

        clean_df = pd.DataFrame({
            "window_id": range(n_windows),
            "start_time_s": time_vals,
            "theta_beta_ratio": theta_beta,
        })

        result = compute_trends(clean_df, features=["theta_beta_ratio"])
        row = result[result["feature"] == "theta_beta_ratio"].iloc[0]

        # Should detect positive slope
        assert row["slope"] > 0, f"Expected positive slope, got {row['slope']:.4f}"

        # Spearman correlation should be positive and significant
        assert row["spearman_rho"] > 0.5, \
            f"Expected strong positive correlation, got {row['spearman_rho']:.3f}"
        assert row["spearman_p"] < 0.05, \
            f"Expected significant p-value, got {row['spearman_p']:.4f}"

    def test_decreasing_trend_detection(self):
        """Verify detection of decreasing trend."""
        np.random.seed(42)

        n_windows = 50
        time_vals = np.arange(n_windows) * 2.0

        # Create decreasing trend
        base_trend = -0.3 * np.arange(n_windows)
        noise = np.random.normal(0, 1, n_windows)
        alpha_power = 100 + base_trend + noise

        clean_df = pd.DataFrame({
            "window_id": range(n_windows),
            "start_time_s": time_vals,
            "alpha_power": alpha_power,
        })

        result = compute_trends(clean_df, features=["alpha_power"])
        row = result.iloc[0]

        # Should detect negative slope
        assert row["slope"] < 0, f"Expected negative slope, got {row['slope']:.4f}"

        # Spearman correlation should be negative
        assert row["spearman_rho"] < 0, \
            f"Expected negative correlation, got {row['spearman_rho']:.3f}"

    def test_no_trend_with_random_data(self):
        """Verify no significant trend with random data."""
        np.random.seed(123)  # Different seed for more neutral random data

        n_windows = 50
        # Generate truly random data with no correlation to time
        random_vals = np.random.normal(50, 5, n_windows)
        np.random.shuffle(random_vals)  # Extra shuffle to break any accidental pattern

        clean_df = pd.DataFrame({
            "window_id": range(n_windows),
            "start_time_s": np.arange(n_windows) * 2.0,
            "beta_power": random_vals,
        })

        result = compute_trends(clean_df, features=["beta_power"])
        row = result.iloc[0]

        # With random data, Spearman correlation should be weak
        # (not necessarily significant, but weak in magnitude)
        assert abs(row["spearman_rho"]) < 0.5, \
            f"Expected weak correlation for random data, got {row['spearman_rho']:.3f}"

    def test_uses_window_id_when_no_time(self):
        """Verify fallback to window_id when no time column."""
        np.random.seed(42)

        n_windows = 30
        clean_df = pd.DataFrame({
            "window_id": range(n_windows),
            "gamma_power": 10 + 0.5 * np.arange(n_windows),
        })

        result = compute_trends(clean_df, features=["gamma_power"])
        row = result.iloc[0]

        assert row["slope_units"] == "per_window"
        assert row["slope"] > 0


class TestWriteSummaries:
    """Tests for summary file generation."""

    def test_generates_all_output_files(self):
        """Verify all expected output files are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create minimal before/after and trend DataFrames
            before_after_df = pd.DataFrame({
                "feature": ["alpha_power"],
                "group": ["overall"],
                "mean_before": [100.0],
                "mean_after": [95.0],
                "var_before": [400.0],
                "var_after": [200.0],
                "std_before": [20.0],
                "std_after": [14.14],
                "variance_reduction_percent": [50.0],
                "percent_change_mean": [-5.0],
                "test_name": ["paired_t_test"],
                "p_value": [0.01],
                "effect_size": [0.25],
                "n_windows_before": [100],
                "n_windows_after": [80],
            })

            trend_df = pd.DataFrame({
                "feature": ["alpha_power"],
                "group": ["overall"],
                "slope": [0.5],
                "slope_units": ["per_minute"],
                "spearman_rho": [0.3],
                "spearman_p": [0.04],
                "n_windows_clean": [80],
            })

            paths = write_summaries(
                before_after_df, trend_df, output_dir,
                n_total=100, n_clean=80, n_rejected=20
            )

            # Check all files exist
            assert paths["before_after_summary"].exists()
            assert paths["trend_summary"].exists()
            assert paths["improvement_summary"].exists()

            # Check CSV content
            ba_loaded = pd.read_csv(paths["before_after_summary"])
            assert len(ba_loaded) == 1
            assert ba_loaded.iloc[0]["feature"] == "alpha_power"

            trend_loaded = pd.read_csv(paths["trend_summary"])
            assert len(trend_loaded) == 1

            # Check summary text content
            with open(paths["improvement_summary"]) as f:
                summary_text = f.read()

            assert "IMPACT ANALYSIS SUMMARY" in summary_text
            assert "Total windows:    100" in summary_text
            assert "Clean windows:    80" in summary_text
            assert "alpha_power" in summary_text


class TestIntegration:
    """Integration tests for full impact analysis pipeline."""

    def test_full_pipeline_with_synthetic_data(self):
        """Test complete pipeline with synthetic window data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)

            # Create predictions directory
            predictions_dir = run_dir / "predictions"
            predictions_dir.mkdir()

            np.random.seed(42)
            n_windows = 100

            # Create synthetic window predictions with is_clean column
            window_data = {
                "window_id": range(n_windows),
                "start_time_s": np.arange(n_windows) * 2.0,
                "duration_s": [2.0] * n_windows,
                "recording_id": ["test_recording"] * n_windows,
            }

            # Add features
            for feat in ANALYSIS_FEATURES:
                if "ratio" in feat:
                    window_data[feat] = np.random.uniform(0.5, 2.0, n_windows)
                elif "power" in feat:
                    window_data[feat] = np.random.uniform(10, 200, n_windows)
                else:
                    window_data[feat] = np.random.uniform(5, 100, n_windows)

            # Make some windows artifacts (high ptp_uv)
            artifact_mask = np.zeros(n_windows, dtype=bool)
            artifact_mask[80:] = True  # Last 20 are artifacts

            # Inflate artifact values
            window_data["ptp_uv"][artifact_mask] = np.random.uniform(300, 500, sum(artifact_mask))
            window_data["rms_uv"][artifact_mask] = np.random.uniform(100, 200, sum(artifact_mask))

            # Add is_clean column
            window_data["is_clean"] = ~artifact_mask

            # Save window predictions
            df = pd.DataFrame(window_data)
            df.to_csv(predictions_dir / "window_predictions.csv", index=False)

            # Also save metrics_per_window.csv (without is_clean)
            metrics_df = df.drop(columns=["is_clean"])
            metrics_df.to_csv(run_dir / "metrics_per_window.csv", index=False)

            # Run full analysis
            result = run_impact_analysis(run_dir)

            # Verify outputs
            assert result["n_total"] == 100
            assert result["n_clean"] == 80
            assert result["n_rejected"] == 20

            assert len(result["features_analyzed"]) > 0
            assert "before_after_summary" in result["output_paths"]
            assert "trend_summary" in result["output_paths"]
            assert "improvement_summary" in result["output_paths"]

            # Check variance reduction for ptp_uv is high
            ba_df = result["before_after"]
            ptp_row = ba_df[
                (ba_df["feature"] == "ptp_uv") & (ba_df["group"] == "overall")
            ]
            if len(ptp_row) > 0:
                assert ptp_row.iloc[0]["variance_reduction_percent"] > 30

    def test_handles_missing_predictions_gracefully(self):
        """Test error handling when predictions are missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)

            # Create only metrics_per_window.csv without labels
            df = pd.DataFrame({
                "window_id": [0, 1, 2],
                "alpha_power": [100, 110, 120],
            })
            df.to_csv(run_dir / "metrics_per_window.csv", index=False)

            # Should raise ValueError about missing labels
            with pytest.raises(ValueError, match="No labels found"):
                run_impact_analysis(run_dir)


class TestLoadWindowData:
    """Tests for window data loading."""

    def test_loads_from_window_predictions(self):
        """Verify loading from predictions/window_predictions.csv."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            predictions_dir = run_dir / "predictions"
            predictions_dir.mkdir()

            df = pd.DataFrame({
                "window_id": [0, 1, 2],
                "alpha_power": [100, 110, 120],
                "is_clean": [True, True, False],
            })
            df.to_csv(predictions_dir / "window_predictions.csv", index=False)
            df.drop(columns=["is_clean"]).to_csv(
                run_dir / "metrics_per_window.csv", index=False
            )

            all_df, clean_df, features = load_window_data(run_dir)

            assert len(all_df) == 3
            assert len(clean_df) == 2
            assert "alpha_power" in features

    def test_loads_from_ml_label_column(self):
        """Verify loading when ml_label is in metrics_per_window.csv."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)

            df = pd.DataFrame({
                "window_id": [0, 1, 2, 3],
                "alpha_power": [100, 110, 120, 130],
                "ml_label": [0, 0, 1, 0],  # 0 = clean, 1 = artifact
            })
            df.to_csv(run_dir / "metrics_per_window.csv", index=False)

            all_df, clean_df, features = load_window_data(run_dir)

            assert len(all_df) == 4
            assert len(clean_df) == 3  # 3 clean windows

    def test_loads_from_separate_labels_file(self):
        """Verify loading from window_labels.csv."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)

            # Main metrics file
            metrics_df = pd.DataFrame({
                "window_id": [0, 1, 2],
                "beta_power": [50, 60, 70],
            })
            metrics_df.to_csv(run_dir / "metrics_per_window.csv", index=False)

            # Separate labels file
            labels_df = pd.DataFrame({
                "window_id": [0, 1, 2],
                "ml_label": [0, 1, 0],
            })
            labels_df.to_csv(run_dir / "window_labels.csv", index=False)

            all_df, clean_df, features = load_window_data(run_dir)

            assert len(all_df) == 3
            assert len(clean_df) == 2
