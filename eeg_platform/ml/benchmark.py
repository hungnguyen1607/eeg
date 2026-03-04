# benchmarks processing throughput

import json
import time
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    # holds benchmark results
    n_windows: int
    n_samples: int
    duration_sec: float
    runtime_sec: float
    windows_per_sec: float
    samples_per_sec: float
    realtime_factor: float


class BenchmarkError(Exception):
    # benchmark failed
    pass


def run_benchmark(
    data: np.ndarray,
    sfreq: float,
    window_sec: float = 2.0,
    overlap: float = 0.5,
    n_iterations: int = 3,
    include_ml: bool = True,
    model_dir: Optional[str | Path] = None,
) -> dict:
    # benchmarks the full processing pipeline
    from ..preprocess.windowing import segment_into_windows
    from ..preprocess.clean import clean_signal
    from .featureset import compute_window_features

    n_channels, n_samples = data.shape
    duration_sec = n_samples / sfreq

    timings = {
        "preprocess": [],
        "windowing": [],
        "feature_extraction": [],
        "ml_prediction": [],
        "total": [],
    }

    for _ in range(n_iterations):
        total_start = time.perf_counter()

        # 1. Preprocessing
        t0 = time.perf_counter()
        cleaned = clean_signal(data, sfreq)
        timings["preprocess"].append(time.perf_counter() - t0)

        # 2. Windowing
        t0 = time.perf_counter()
        windows, window_times = segment_into_windows(
            cleaned, sfreq, window_sec=window_sec, overlap=overlap
        )
        timings["windowing"].append(time.perf_counter() - t0)

        # 3. Feature extraction
        t0 = time.perf_counter()
        features_list = []
        for window in windows:
            feats = compute_window_features(window, sfreq)
            features_list.append(feats)
        timings["feature_extraction"].append(time.perf_counter() - t0)

        # 4. ML prediction (if enabled)
        if include_ml and model_dir is not None:
            from .predict_artifact import predict_artifacts
            from .featureset import select_ml_features
            import pandas as pd

            df = pd.DataFrame(features_list)
            try:
                X, _ = select_ml_features(df)
                t0 = time.perf_counter()
                _, _ = predict_artifacts(X, model_dir)
                timings["ml_prediction"].append(time.perf_counter() - t0)
            except Exception:
                timings["ml_prediction"].append(0.0)
        else:
            timings["ml_prediction"].append(0.0)

        timings["total"].append(time.perf_counter() - total_start)

    # Compute statistics
    n_windows = len(windows) if 'windows' in dir() else 0

    avg_timings = {k: float(np.mean(v)) for k, v in timings.items()}
    std_timings = {k: float(np.std(v)) for k, v in timings.items()}

    total_runtime = avg_timings["total"]

    # Compute throughput metrics
    windows_per_sec = n_windows / max(total_runtime, 1e-10)
    samples_per_sec = n_samples / max(total_runtime, 1e-10)
    realtime_factor = duration_sec / max(total_runtime, 1e-10)

    result = {
        "input": {
            "n_channels": n_channels,
            "n_samples": n_samples,
            "duration_sec": float(duration_sec),
            "sfreq": sfreq,
            "window_sec": window_sec,
            "overlap": overlap,
        },
        "output": {
            "n_windows": n_windows,
        },
        "timing_mean_sec": avg_timings,
        "timing_std_sec": std_timings,
        "throughput": {
            "windows_per_sec": float(windows_per_sec),
            "samples_per_sec": float(samples_per_sec),
            "realtime_factor": float(realtime_factor),
        },
        "n_iterations": n_iterations,
        "include_ml": include_ml,
    }

    return result


def benchmark_recordings(
    recording_paths: list[str | Path],
    sfreq: float,
    output_dir: str | Path,
    window_sec: float = 2.0,
    overlap: float = 0.5,
    model_dir: Optional[str | Path] = None,
) -> dict:
    # benchmarks across multiple recordings
    from ..io.load_csv import load_eeg_csv

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    total_windows = 0
    total_samples = 0
    total_duration = 0.0
    total_runtime = 0.0

    for path in recording_paths:
        try:
            result = load_eeg_csv(str(path), sfreq=sfreq)
            data = result["data"]

            bench = run_benchmark(
                data, sfreq,
                window_sec=window_sec,
                overlap=overlap,
                n_iterations=1,
                include_ml=model_dir is not None,
                model_dir=model_dir,
            )

            all_results.append({
                "path": str(path),
                "result": bench,
            })

            total_windows += bench["output"]["n_windows"]
            total_samples += bench["input"]["n_samples"]
            total_duration += bench["input"]["duration_sec"]
            total_runtime += bench["timing_mean_sec"]["total"]

        except Exception as e:
            all_results.append({
                "path": str(path),
                "error": str(e),
            })

    # Aggregate metrics
    aggregate = {
        "n_recordings": len(recording_paths),
        "total_windows": total_windows,
        "total_samples": total_samples,
        "total_duration_sec": float(total_duration),
        "total_runtime_sec": float(total_runtime),
        "aggregate_windows_per_sec": total_windows / max(total_runtime, 1e-10),
        "aggregate_realtime_factor": total_duration / max(total_runtime, 1e-10),
        "per_recording": all_results,
    }

    # Save results
    metrics_path = output_dir / "performance_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(aggregate, f, indent=2)

    aggregate["metrics_path"] = str(metrics_path)

    return aggregate


def generate_benchmark_summary(metrics: dict) -> str:
    # makes a readable summary of benchmark results
    lines = [
        "=" * 60,
        "THROUGHPUT BENCHMARK",
        "=" * 60,
    ]

    if "throughput" in metrics:
        # Single recording result
        t = metrics["throughput"]
        inp = metrics["input"]

        lines.extend([
            f"Input: {inp['n_channels']} channels, {inp['n_samples']} samples",
            f"Duration: {inp['duration_sec']:.2f} sec",
            f"Windows: {metrics['output']['n_windows']}",
            "",
            "Throughput:",
            f"  Windows/sec: {t['windows_per_sec']:.1f}",
            f"  Samples/sec: {t['samples_per_sec']:.0f}",
            f"  Realtime factor: {t['realtime_factor']:.1f}x",
            "",
            "Timing breakdown:",
        ])

        for stage, time_sec in metrics["timing_mean_sec"].items():
            lines.append(f"  {stage}: {time_sec*1000:.1f} ms")

    else:
        # Aggregate result
        lines.extend([
            f"Recordings: {metrics['n_recordings']}",
            f"Total windows: {metrics['total_windows']}",
            f"Total duration: {metrics['total_duration_sec']:.1f} sec",
            f"Total runtime: {metrics['total_runtime_sec']:.2f} sec",
            "",
            f"Windows/sec: {metrics['aggregate_windows_per_sec']:.1f}",
            f"Realtime factor: {metrics['aggregate_realtime_factor']:.1f}x",
        ])

    lines.append("=" * 60)

    return "\n".join(lines)
