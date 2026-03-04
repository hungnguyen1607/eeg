# before/after analysis for artifact rejection

from pathlib import Path
from typing import Optional
import warnings

import numpy as np
import pandas as pd
from scipy import stats


ANALYSIS_FEATURES = [
    "alpha_power",
    "beta_power",
    "theta_power",
    "delta_power",
    "gamma_power",
    "rms_uv",
    "ptp_uv",
    "theta_beta_ratio",
    "alpha_theta_ratio",
]


def load_window_data(
    run_dir: Path,
    metrics_filename: str = "metrics_per_window.csv",
    predictions_filename: str = "predictions/window_predictions.csv",
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    # loads window features and labels, returns all_windows, clean_windows, and available features
    run_dir = Path(run_dir)

    # Load main metrics file
    metrics_path = run_dir / metrics_filename
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    df = pd.read_csv(metrics_path)

    # Try to get clean labels
    is_clean = None

    # Option 1: Check for predictions file with is_clean column
    predictions_path = run_dir / predictions_filename
    if predictions_path.exists():
        pred_df = pd.read_csv(predictions_path)
        if "is_clean" in pred_df.columns:
            # Use predictions file as the main data source (has all columns + labels)
            df = pred_df
            is_clean = df["is_clean"].astype(bool)

    # Option 2: Check for ml_label in metrics file
    if is_clean is None and "ml_label" in df.columns:
        # ml_label: 0 = clean, 1 = artifact
        is_clean = df["ml_label"] == 0

    # Option 3: Check for separate window_labels.csv
    if is_clean is None:
        labels_path = run_dir / "window_labels.csv"
        if labels_path.exists():
            labels_df = pd.read_csv(labels_path)
            if "window_id" in labels_df.columns and "ml_label" in labels_df.columns:
                # Merge labels with main dataframe
                df = df.merge(labels_df[["window_id", "ml_label"]], on="window_id", how="left")
                is_clean = df["ml_label"] == 0

    if is_clean is None:
        raise ValueError(
            "No labels found. Expected one of:\n"
            "  - predictions/window_predictions.csv with 'is_clean' column\n"
            "  - metrics_per_window.csv with 'ml_label' column\n"
            "  - window_labels.csv with 'window_id' and 'ml_label' columns"
        )

    # Find available features
    available_features = [f for f in ANALYSIS_FEATURES if f in df.columns]
    missing_features = [f for f in ANALYSIS_FEATURES if f not in df.columns]

    if missing_features:
        warnings.warn(f"Missing features (will be skipped): {missing_features}")

    if not available_features:
        raise ValueError(f"No analysis features found in data. Expected: {ANALYSIS_FEATURES}")

    # Split into all and clean
    all_windows = df.copy()
    clean_windows = df[is_clean].copy()

    return all_windows, clean_windows, available_features


def _safe_variance_reduction(var_before: float, var_after: float) -> float:
    # calc variance reduction %, handles edge cases
    if var_before == 0 or np.isnan(var_before):
        return 0.0 if var_after == 0 else np.nan
    return (1 - var_after / var_before) * 100


def _safe_percent_change(before: float, after: float) -> float:
    # calc percent change, handles near-zero
    if abs(before) < 1e-10:
        return 0.0 if abs(after) < 1e-10 else np.nan
    return ((after - before) / abs(before)) * 100


def _compute_effect_size(before: np.ndarray, after: np.ndarray, paired: bool) -> float:
    # cohen's d effect size
    before = before[~np.isnan(before)]
    after = after[~np.isnan(after)]

    if len(before) < 2 or len(after) < 2:
        return np.nan

    if paired and len(before) == len(after):
        diff = after - before
        if np.std(diff) < 1e-10:
            return 0.0
        return np.mean(diff) / np.std(diff, ddof=1)
    else:
        # Pooled standard deviation
        n1, n2 = len(before), len(after)
        var1, var2 = np.var(before, ddof=1), np.var(after, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        if pooled_std < 1e-10:
            return 0.0
        return (np.mean(after) - np.mean(before)) / pooled_std


def _select_and_run_test(
    before: np.ndarray,
    after: np.ndarray,
    paired: bool
) -> tuple[str, float]:
    # picks and runs the right stat test (parametric or non-parametric)
    before = before[~np.isnan(before)]
    after = after[~np.isnan(after)]

    if len(before) < 3 or len(after) < 3:
        return "insufficient_data", np.nan

    # Check normality (use Shapiro-Wilk for small samples)
    try:
        if len(before) <= 5000:
            _, p_normal_before = stats.shapiro(before[:min(len(before), 5000)])
            _, p_normal_after = stats.shapiro(after[:min(len(after), 5000)])
        else:
            # For large samples, use D'Agostino-Pearson
            _, p_normal_before = stats.normaltest(before)
            _, p_normal_after = stats.normaltest(after)

        is_normal = p_normal_before > 0.05 and p_normal_after > 0.05
    except Exception:
        is_normal = False

    # Select and run test
    if paired and len(before) == len(after):
        if is_normal:
            stat, p_value = stats.ttest_rel(before, after)
            test_name = "paired_t_test"
        else:
            stat, p_value = stats.wilcoxon(before, after, zero_method='wilcox')
            test_name = "wilcoxon"
    else:
        if is_normal:
            stat, p_value = stats.ttest_ind(before, after)
            test_name = "unpaired_t_test"
        else:
            stat, p_value = stats.mannwhitneyu(before, after, alternative='two-sided')
            test_name = "mann_whitney_u"

    return test_name, p_value


def compute_before_after(
    all_windows: pd.DataFrame,
    clean_windows: pd.DataFrame,
    features: list[str],
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    # compares before (all windows) vs after (clean only) stats
    results = []

    # Determine groups
    if group_col and group_col in all_windows.columns:
        groups = all_windows[group_col].unique().tolist()
        groups.append("overall")
    else:
        groups = ["overall"]

    for group in groups:
        if group == "overall":
            df_before = all_windows
            df_after = clean_windows
        else:
            df_before = all_windows[all_windows[group_col] == group]
            df_after = clean_windows[clean_windows[group_col] == group]

        n_before = len(df_before)
        n_after = len(df_after)

        # Check if we can do paired analysis
        # Paired if same window_ids exist in both and same count
        can_pair = False
        if "window_id" in df_before.columns and "window_id" in df_after.columns:
            before_ids = set(df_before["window_id"])
            after_ids = set(df_after["window_id"])
            can_pair = after_ids.issubset(before_ids)

        for feature in features:
            if feature not in df_before.columns:
                continue

            before_vals = df_before[feature].dropna().values
            after_vals = df_after[feature].dropna().values

            if len(before_vals) == 0:
                continue

            # Compute basic stats
            mean_before = np.mean(before_vals)
            mean_after = np.mean(after_vals) if len(after_vals) > 0 else np.nan
            var_before = np.var(before_vals, ddof=1)
            var_after = np.var(after_vals, ddof=1) if len(after_vals) > 0 else np.nan
            std_before = np.std(before_vals, ddof=1)
            std_after = np.std(after_vals, ddof=1) if len(after_vals) > 0 else np.nan

            # Compute derived metrics
            var_reduction = _safe_variance_reduction(var_before, var_after)
            pct_change_mean = _safe_percent_change(mean_before, mean_after)

            # Statistical test
            if can_pair and len(after_vals) > 0:
                # Get matched pairs by window_id
                merged = df_before[["window_id", feature]].merge(
                    df_after[["window_id", feature]],
                    on="window_id",
                    suffixes=("_before", "_after")
                )
                before_paired = merged[f"{feature}_before"].values
                after_paired = merged[f"{feature}_after"].values
                test_name, p_value = _select_and_run_test(before_paired, after_paired, paired=True)
                effect_size = _compute_effect_size(before_paired, after_paired, paired=True)
            elif len(after_vals) > 0:
                test_name, p_value = _select_and_run_test(before_vals, after_vals, paired=False)
                effect_size = _compute_effect_size(before_vals, after_vals, paired=False)
            else:
                test_name = "no_clean_windows"
                p_value = np.nan
                effect_size = np.nan

            results.append({
                "feature": feature,
                "group": group,
                "mean_before": mean_before,
                "mean_after": mean_after,
                "var_before": var_before,
                "var_after": var_after,
                "std_before": std_before,
                "std_after": std_after,
                "variance_reduction_percent": var_reduction,
                "percent_change_mean": pct_change_mean,
                "test_name": test_name,
                "p_value": p_value,
                "effect_size": effect_size,
                "n_windows_before": n_before,
                "n_windows_after": n_after,
            })

    return pd.DataFrame(results)


def compute_trends(
    clean_windows: pd.DataFrame,
    features: list[str],
    time_col: Optional[str] = None,
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    # calculates temporal trends for clean windows
    results = []

    # Determine time column
    if time_col is None:
        if "start_time_s" in clean_windows.columns:
            time_col = "start_time_s"
            slope_units = "per_minute"
            time_scale = 60.0  # Convert slope to per-minute
        elif "time_mid_sec" in clean_windows.columns:
            time_col = "time_mid_sec"
            slope_units = "per_minute"
            time_scale = 60.0
        elif "window_id" in clean_windows.columns:
            time_col = "window_id"
            slope_units = "per_window"
            time_scale = 1.0
        else:
            # Use index
            clean_windows = clean_windows.reset_index(drop=True)
            clean_windows["_index"] = clean_windows.index
            time_col = "_index"
            slope_units = "per_window"
            time_scale = 1.0
    else:
        if "sec" in time_col.lower() or "time" in time_col.lower():
            slope_units = "per_minute"
            time_scale = 60.0
        else:
            slope_units = "per_window"
            time_scale = 1.0

    # Determine groups
    if group_col and group_col in clean_windows.columns:
        groups = clean_windows[group_col].unique().tolist()
        groups.append("overall")
    else:
        groups = ["overall"]

    for group in groups:
        if group == "overall":
            df = clean_windows
        else:
            df = clean_windows[clean_windows[group_col] == group]

        n_clean = len(df)

        if n_clean < 3:
            # Not enough data for trend analysis
            for feature in features:
                results.append({
                    "feature": feature,
                    "group": group,
                    "slope": np.nan,
                    "slope_units": slope_units,
                    "spearman_rho": np.nan,
                    "spearman_p": np.nan,
                    "n_windows_clean": n_clean,
                })
            continue

        time_vals = df[time_col].values

        for feature in features:
            if feature not in df.columns:
                continue

            feat_vals = df[feature].values

            # Remove NaN pairs
            valid_mask = ~(np.isnan(time_vals) | np.isnan(feat_vals))
            t = time_vals[valid_mask]
            y = feat_vals[valid_mask]

            if len(t) < 3:
                results.append({
                    "feature": feature,
                    "group": group,
                    "slope": np.nan,
                    "slope_units": slope_units,
                    "spearman_rho": np.nan,
                    "spearman_p": np.nan,
                    "n_windows_clean": len(t),
                })
                continue

            # Linear regression for slope
            try:
                slope, intercept = np.polyfit(t, y, 1)
                slope_scaled = slope * time_scale  # Convert to per-minute or per-window
            except Exception:
                slope_scaled = np.nan

            # Spearman correlation
            try:
                spearman_rho, spearman_p = stats.spearmanr(t, y)
            except Exception:
                spearman_rho, spearman_p = np.nan, np.nan

            results.append({
                "feature": feature,
                "group": group,
                "slope": slope_scaled,
                "slope_units": slope_units,
                "spearman_rho": spearman_rho,
                "spearman_p": spearman_p,
                "n_windows_clean": len(t),
            })

    return pd.DataFrame(results)


def write_summaries(
    before_after_df: pd.DataFrame,
    trend_df: pd.DataFrame,
    output_dir: Path,
    n_total: int,
    n_clean: int,
    n_rejected: int,
) -> dict[str, Path]:
    # writes summary csvs and txt to output dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # 1. before_after_summary.csv
    ba_path = output_dir / "before_after_summary.csv"
    before_after_df.to_csv(ba_path, index=False)
    paths["before_after_summary"] = ba_path

    # 2. trend_summary.csv
    trend_path = output_dir / "trend_summary.csv"
    trend_df.to_csv(trend_path, index=False)
    paths["trend_summary"] = trend_path

    # 3. improvement_summary.txt
    txt_path = output_dir / "improvement_summary.txt"

    # Filter to overall group for summary
    ba_overall = before_after_df[before_after_df["group"] == "overall"].copy()
    trend_overall = trend_df[trend_df["group"] == "overall"].copy()

    lines = []
    lines.append("=" * 70)
    lines.append("IMPACT ANALYSIS SUMMARY")
    lines.append("=" * 70)
    lines.append("")

    # Window counts
    lines.append("WINDOW STATISTICS")
    lines.append("-" * 40)
    lines.append(f"  Total windows:    {n_total}")
    lines.append(f"  Clean windows:    {n_clean} ({100*n_clean/n_total:.1f}%)")
    lines.append(f"  Rejected windows: {n_rejected} ({100*n_rejected/n_total:.1f}%)")
    lines.append("")

    # Top 5 by variance reduction
    lines.append("TOP 5 FEATURES BY VARIANCE REDUCTION")
    lines.append("-" * 40)
    if len(ba_overall) > 0:
        top5 = ba_overall.nlargest(5, "variance_reduction_percent")
        for _, row in top5.iterrows():
            vr = row["variance_reduction_percent"]
            if np.isfinite(vr):
                lines.append(f"  {row['feature']:20s}: {vr:+.1f}% variance reduction")
    else:
        lines.append("  (no data)")
    lines.append("")

    # Significant before/after improvements
    lines.append("SIGNIFICANT BEFORE/AFTER CHANGES (p < 0.05)")
    lines.append("-" * 40)
    if len(ba_overall) > 0:
        significant = ba_overall[ba_overall["p_value"] < 0.05]
        if len(significant) > 0:
            for _, row in significant.iterrows():
                direction = "decreased" if row["percent_change_mean"] < 0 else "increased"
                lines.append(
                    f"  {row['feature']:20s}: mean {direction} by {abs(row['percent_change_mean']):.1f}% "
                    f"(p={row['p_value']:.4f}, d={row['effect_size']:.2f})"
                )
        else:
            lines.append("  (no significant changes)")
    else:
        lines.append("  (no data)")
    lines.append("")

    # Significant trends
    lines.append("SIGNIFICANT TEMPORAL TRENDS (p < 0.05)")
    lines.append("-" * 40)
    if len(trend_overall) > 0:
        sig_trends = trend_overall[trend_overall["spearman_p"] < 0.05]
        if len(sig_trends) > 0:
            for _, row in sig_trends.iterrows():
                direction = "increasing" if row["spearman_rho"] > 0 else "decreasing"
                lines.append(
                    f"  {row['feature']:20s}: {direction} (rho={row['spearman_rho']:.3f}, "
                    f"p={row['spearman_p']:.4f}, slope={row['slope']:.4f} {row['slope_units']})"
                )
        else:
            lines.append("  (no significant trends)")
    else:
        lines.append("  (no data)")
    lines.append("")
    lines.append("=" * 70)

    with open(txt_path, "w") as f:
        f.write("\n".join(lines))

    paths["improvement_summary"] = txt_path

    return paths


def run_impact_analysis(
    run_dir: Path,
    output_subdir: Optional[str] = None,
) -> dict:
    # runs the full before/after + trend analysis
    run_dir = Path(run_dir)
    output_dir = run_dir / output_subdir if output_subdir else run_dir

    # Load data
    all_windows, clean_windows, features = load_window_data(run_dir)

    n_total = len(all_windows)
    n_clean = len(clean_windows)
    n_rejected = n_total - n_clean

    # Detect grouping column
    group_col = None
    if "recording_id" in all_windows.columns:
        if all_windows["recording_id"].nunique() > 1:
            group_col = "recording_id"

    # Compute before/after
    before_after_df = compute_before_after(
        all_windows, clean_windows, features, group_col=group_col
    )

    # Compute trends
    trend_df = compute_trends(
        clean_windows, features, group_col=group_col
    )

    # Write summaries
    paths = write_summaries(
        before_after_df, trend_df, output_dir,
        n_total, n_clean, n_rejected
    )

    return {
        "n_total": n_total,
        "n_clean": n_clean,
        "n_rejected": n_rejected,
        "features_analyzed": features,
        "before_after": before_after_df,
        "trends": trend_df,
        "output_paths": paths,
    }
