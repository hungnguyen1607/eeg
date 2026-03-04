# metrics for before/after ml artifact rejection

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


# Key features to track for stability improvement
KEY_FEATURES = [
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

# Extreme value thresholds
EXTREME_THRESHOLDS = {
    "ptp_uv": 150.0,      # Peak-to-peak > 150 uV is extreme
    "rms_uv": 75.0,       # RMS > 75 uV is extreme
    "kurtosis": 10.0,     # Kurtosis > 10 indicates spikes
}


class ImprovementError(Exception):
    # improvement calc failed
    pass


def compute_variance_reduction(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    features: Optional[list[str]] = None,
) -> dict:
    # computes variance reduction for features
    if features is None:
        # Auto-detect available features
        features = [f for f in KEY_FEATURES if f in df_before.columns]

    if len(features) == 0:
        # Try alternative column names
        alt_map = {
            "alpha_power": "bandpower_alpha",
            "beta_power": "bandpower_beta",
            "theta_power": "bandpower_theta",
            "delta_power": "bandpower_delta",
            "gamma_power": "bandpower_gamma",
            "rms_uv": "rms",
            "ptp_uv": "peak_to_peak",
        }
        for primary, alt in alt_map.items():
            if alt in df_before.columns:
                features.append(alt)

    if len(features) == 0:
        raise ImprovementError(
            f"No key features found. Available: {list(df_before.columns)}"
        )

    reductions = {}
    for feat in features:
        if feat not in df_before.columns:
            continue

        var_before = df_before[feat].var()
        var_after = df_after[feat].var() if len(df_after) > 1 else 0

        # Compute percent reduction
        if var_before > 0:
            reduction_pct = (var_before - var_after) / var_before * 100
        else:
            reduction_pct = 0.0

        reductions[feat] = {
            "variance_before": float(var_before),
            "variance_after": float(var_after),
            "reduction_pct": float(reduction_pct),
        }

    # Compute median reduction across features
    reduction_values = [r["reduction_pct"] for r in reductions.values()]
    median_reduction = float(np.median(reduction_values)) if reduction_values else 0

    return {
        "per_feature": reductions,
        "median_variance_reduction_pct": median_reduction,
        "mean_variance_reduction_pct": float(np.mean(reduction_values)) if reduction_values else 0,
        "n_features_analyzed": len(features),
    }


def compute_extreme_window_reduction(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    thresholds: Optional[dict] = None,
) -> dict:
    # how many extreme/outlier windows got removed
    if thresholds is None:
        thresholds = EXTREME_THRESHOLDS.copy()

    results = {}

    for metric, threshold in thresholds.items():
        # Check alternative column names
        col = metric
        if metric not in df_before.columns:
            alt_names = {
                "ptp_uv": "peak_to_peak",
                "rms_uv": "rms",
            }
            if metric in alt_names and alt_names[metric] in df_before.columns:
                col = alt_names[metric]
            else:
                continue

        # Count extreme windows before
        extreme_before = int(np.sum(np.abs(df_before[col]) > threshold))

        # Count extreme windows after
        if col in df_after.columns and len(df_after) > 0:
            extreme_after = int(np.sum(np.abs(df_after[col]) > threshold))
        else:
            extreme_after = 0

        # Compute reduction
        if extreme_before > 0:
            reduction_pct = (extreme_before - extreme_after) / extreme_before * 100
        else:
            reduction_pct = 0.0

        results[metric] = {
            "threshold": threshold,
            "extreme_before": extreme_before,
            "extreme_after": extreme_after,
            "reduction_pct": float(reduction_pct),
        }

    # Overall extreme reduction
    total_extreme_before = sum(r["extreme_before"] for r in results.values())
    total_extreme_after = sum(r["extreme_after"] for r in results.values())

    if total_extreme_before > 0:
        overall_reduction = (total_extreme_before - total_extreme_after) / total_extreme_before * 100
    else:
        overall_reduction = 0.0

    return {
        "per_metric": results,
        "total_extreme_before": total_extreme_before,
        "total_extreme_after": total_extreme_after,
        "overall_extreme_reduction_pct": float(overall_reduction),
    }


def compute_signal_quality_improvement(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
) -> dict:
    # signal quality improvements (line noise, bandpower consistency)
    results = {}

    # Line noise reduction
    if "line_noise_power" in df_before.columns:
        mean_before = float(df_before["line_noise_power"].mean())
        mean_after = float(df_after["line_noise_power"].mean()) if len(df_after) > 0 else 0

        if mean_before > 0:
            reduction = (mean_before - mean_after) / mean_before * 100
        else:
            reduction = 0.0

        results["line_noise_reduction_pct"] = reduction
        results["line_noise_mean_before"] = mean_before
        results["line_noise_mean_after"] = mean_after

    # Bandpower consistency (lower CV = more consistent)
    for band in ["alpha_power", "beta_power", "bandpower_alpha", "bandpower_beta"]:
        if band in df_before.columns:
            cv_before = df_before[band].std() / max(df_before[band].mean(), 1e-10)
            cv_after = df_after[band].std() / max(df_after[band].mean(), 1e-10) if len(df_after) > 1 else 0

            results[f"{band}_cv_before"] = float(cv_before)
            results[f"{band}_cv_after"] = float(cv_after)
            results[f"{band}_cv_improvement_pct"] = float((cv_before - cv_after) / max(cv_before, 1e-10) * 100)

    return results


def compute_improvement_metrics(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    output_dir: Optional[str | Path] = None,
) -> dict:
    # computes all the improvement metrics
    n_before = len(df_before)
    n_after = len(df_after)
    n_rejected = n_before - n_after

    # Compute all metrics
    variance_metrics = compute_variance_reduction(df_before, df_after)
    extreme_metrics = compute_extreme_window_reduction(df_before, df_after)
    quality_metrics = compute_signal_quality_improvement(df_before, df_after)

    metrics = {
        "window_counts": {
            "total_windows": n_before,
            "clean_windows": n_after,
            "rejected_windows": n_rejected,
            "rejection_rate": n_rejected / max(n_before, 1),
        },
        "variance_reduction": variance_metrics,
        "extreme_window_reduction": extreme_metrics,
        "signal_quality": quality_metrics,
        # Summary metrics for easy access
        "variance_reduction_pct": variance_metrics["median_variance_reduction_pct"],
        "extreme_window_reduction_pct": extreme_metrics["overall_extreme_reduction_pct"],
    }

    # Save to file if output_dir provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = output_dir / "improvement_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        metrics["metrics_path"] = str(metrics_path)

    return metrics


def generate_improvement_summary(metrics: dict) -> str:
    # makes a readable summary of improvements
    counts = metrics["window_counts"]
    var_reduction = metrics["variance_reduction_pct"]
    extreme_reduction = metrics["extreme_window_reduction_pct"]

    lines = [
        "=" * 60,
        "ML ARTIFACT REJECTION - IMPROVEMENT SUMMARY",
        "=" * 60,
        "",
        f"Windows processed: {counts['total_windows']}",
        f"Clean windows: {counts['clean_windows']} ({100 - counts['rejection_rate']*100:.1f}%)",
        f"Rejected: {counts['rejected_windows']} ({counts['rejection_rate']*100:.1f}%)",
        "",
        "QUALITY IMPROVEMENTS:",
        f"  Feature variance reduction: {var_reduction:.1f}%",
        f"  Extreme window reduction: {extreme_reduction:.1f}%",
    ]

    # Add per-feature details
    per_feat = metrics["variance_reduction"].get("per_feature", {})
    if per_feat:
        lines.append("")
        lines.append("Per-feature variance reduction:")
        for feat, data in per_feat.items():
            lines.append(f"  {feat}: {data['reduction_pct']:.1f}%")

    lines.append("=" * 60)

    return "\n".join(lines)
