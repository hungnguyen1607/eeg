# cli for eeg platform

import argparse
import sys
from pathlib import Path

import numpy as np

from . import __version__


def create_parser() -> argparse.ArgumentParser:
    # sets up the arg parser
    parser = argparse.ArgumentParser(
        prog="eeg-platform",
        description="EEG signal processing platform",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Demo command - one command does everything
    demo_parser = subparsers.add_parser(
        "demo", help="Generate synthetic EEG, process it, and show results (all-in-one)"
    )
    demo_parser.add_argument(
        "--duration", type=float, default=10.0,
        help="Duration of recording in seconds (default: 10)"
    )
    demo_parser.add_argument(
        "--outdir", type=Path, default=Path("outputs/demo"),
        help="Output directory (default: outputs/demo)"
    )

    # Validate command - check if data is valid EEG
    validate_parser = subparsers.add_parser(
        "validate", help="validate EEG data format"
    )
    validate_parser.add_argument("input", type=Path, help="Input CSV file")
    validate_parser.add_argument(
        "--strict", action="store_true",
        help="Fail on any non-standard columns"
    )

    # Inspect command
    inspect_parser = subparsers.add_parser("inspect", help="Inspect EEG CSV file")
    inspect_parser.add_argument("input", type=Path, help="Input CSV file")
    inspect_parser.add_argument(
        "--sfreq",
        type=float,
        default=None,
        help="Sampling frequency in Hz (required if no time column)",
    )

    # Plot-raw command
    plot_raw_parser = subparsers.add_parser("plot-raw", help="Plot raw EEG time series")
    plot_raw_parser.add_argument("input", type=Path, help="Input CSV file")
    plot_raw_parser.add_argument(
        "--sfreq",
        type=float,
        default=None,
        help="Sampling frequency in Hz (required if no time column)",
    )
    plot_raw_parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output image path (e.g., outputs/raw.png)",
    )
    plot_raw_parser.add_argument(
        "--max-seconds",
        type=float,
        default=10.0,
        help="Maximum seconds to plot (default: 10)",
    )
    plot_raw_parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Scale in microvolts for channel spacing (auto if not set)",
    )

    # Clean-and-plot command
    clean_plot_parser = subparsers.add_parser(
        "clean-and-plot", help="Clean EEG data and plot  (to run pls use   python -m eeg_platform.cli clean-and-plot)"
    )
    clean_plot_parser.add_argument("input", type=Path, help="Input CSV file")
    clean_plot_parser.add_argument(
        "--sfreq",
        type=float,
        default=None,
        help="Sampling frequency in Hz (required if no time column)",
    )
    clean_plot_parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output image path (e.g., outputs/cleaned.png)",
    )
    clean_plot_parser.add_argument(
        "--l-freq",
        type=float,
        default=1.0,
        help="Low cutoff frequency for bandpass (default: 1 Hz)",
    )
    clean_plot_parser.add_argument(
        "--h-freq",
        type=float,
        default=45.0,
        help="High cutoff frequency for bandpass (default: 45 Hz)",
    )
    clean_plot_parser.add_argument(
        "--notch",
        type=float,
        default=60.0,
        help="Notch filter frequency (default: 60 Hz, use 0 to disable)",
    )
    clean_plot_parser.add_argument(
        "--max-seconds",
        type=float,
        default=10.0,
        help="Maximum seconds to plot (default: 10)",
    )
    clean_plot_parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Scale in microvolts for channel spacing (auto if not set)",
    )

    # Features-spectral command
    features_spectral_parser = subparsers.add_parser(
        "features-spectral", help="Extract spectral features and save to JSON"
    )
    features_spectral_parser.add_argument("input", type=Path, help="Input CSV file")
    features_spectral_parser.add_argument(
        "--sfreq",
        type=float,
        default=None,
        help="Sampling frequency in Hz (required if no time column)",
    )
    features_spectral_parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output JSON path (e.g., outputs/spectral.json)",
    )
    features_spectral_parser.add_argument(
        "--nperseg",
        type=int,
        default=512,
        help="Segment length for Welch PSD (default: 512)",
    )

    # Features-time command
    features_time_parser = subparsers.add_parser(
        "features-time", help="Extract time-domain features and save to JSON"
    )
    features_time_parser.add_argument("input", type=Path, help="Input CSV file")
    features_time_parser.add_argument(
        "--sfreq",
        type=float,
        default=None,
        help="Sampling frequency in Hz (required if no time column)",
    )
    features_time_parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output JSON path (e.g., outputs/time_features.json)",
    )

    # Run command (end-to-end pipeline)
    run_parser = subparsers.add_parser(
        "run", help="Run full EEG processing pipeline"
    )
    run_parser.add_argument("input", type=Path, help="Input CSV file")
    run_parser.add_argument(
        "--sfreq",
        type=float,
        default=None,
        help="Sampling frequency in Hz (required if no time column)",
    )
    run_parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Output directory (e.g., outputs/run1)",
    )
    run_parser.add_argument(
        "--l-freq",
        type=float,
        default=1.0,
        help="Low cutoff frequency for bandpass (default: 1 Hz)",
    )
    run_parser.add_argument(
        "--h-freq",
        type=float,
        default=45.0,
        help="High cutoff frequency for bandpass (default: 45 Hz)",
    )
    run_parser.add_argument(
        "--notch",
        type=float,
        default=60.0,
        help="Notch filter frequency (default: 60 Hz, use 0 to disable)",
    )
    run_parser.add_argument(
        "--max-seconds",
        type=float,
        default=10.0,
        help="Maximum seconds to plot (default: 10)",
    )

    # Process command (legacy)
    process_parser = subparsers.add_parser("process", help="Process EEG data")
    process_parser.add_argument("input", type=Path, help="Input CSV file")
    process_parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("outputs"),
        help="Output directory",
    )
    process_parser.add_argument(
        "--sfreq",
        type=float,
        default=256.0,
        help="Sampling frequency in Hz",
    )
    process_parser.add_argument(
        "--lowcut",
        type=float,
        default=1.0,
        help="Low cutoff frequency for bandpass filter",
    )
    process_parser.add_argument(
        "--highcut",
        type=float,
        default=40.0,
        help="High cutoff frequency for bandpass filter",
    )

    # Info command (legacy, kept for compatibility)
    info_parser = subparsers.add_parser("info", help="Show info about EEG file")
    info_parser.add_argument("input", type=Path, help="Input CSV file")

    # ========== ML Commands ==========

    # run-windowed: Run pipeline with windowing for ML
    run_windowed_parser = subparsers.add_parser(
        "run-windowed", help="Run pipeline with window segmentation"
    )
    run_windowed_parser.add_argument("input", type=Path, help="Input CSV file")
    run_windowed_parser.add_argument(
        "--sfreq", type=float, required=True, help="Sampling frequency in Hz"
    )
    run_windowed_parser.add_argument(
        "--outdir", type=Path, required=True, help="Output directory"
    )
    run_windowed_parser.add_argument(
        "--window-sec", type=float, default=2.0, help="Window length in seconds"
    )
    run_windowed_parser.add_argument(
        "--overlap", type=float, default=0.5, help="Overlap fraction (0-1)"
    )

    # ml-train: Train artifact classifier
    ml_train_parser = subparsers.add_parser(
        "ml-train", help="Train ML artifact classifier"
    )
    ml_train_parser.add_argument(
        "features", type=Path, help="Path to metrics_per_window.csv"
    )
    ml_train_parser.add_argument(
        "--outdir", type=Path, required=True, help="Output directory for model"
    )
    ml_train_parser.add_argument(
        "--labels", type=Path, default=None, help="Manual labels CSV (optional)"
    )
    ml_train_parser.add_argument(
        "--model-type", choices=["auto", "logreg", "rf"], default="auto",
        help="Model type (default: auto)"
    )

    # ml-eval: Evaluate trained model
    ml_eval_parser = subparsers.add_parser(
        "ml-eval", help="Evaluate trained artifact model"
    )
    ml_eval_parser.add_argument(
        "model_dir", type=Path, help="Directory with trained model"
    )
    ml_eval_parser.add_argument(
        "features", type=Path, help="Path to test features CSV"
    )
    ml_eval_parser.add_argument(
        "--labels", type=Path, default=None, help="Labels CSV (optional)"
    )
    ml_eval_parser.add_argument(
        "--threshold", type=float, default=0.5, help="Classification threshold"
    )

    # ml-predict: Generate artifact predictions
    ml_predict_parser = subparsers.add_parser(
        "ml-predict", help="Predict artifacts for windows"
    )
    ml_predict_parser.add_argument(
        "features", type=Path, help="Path to metrics_per_window.csv"
    )
    ml_predict_parser.add_argument(
        "model_dir", type=Path, help="Directory with trained model"
    )
    ml_predict_parser.add_argument(
        "--outdir", type=Path, required=True, help="Output directory"
    )
    ml_predict_parser.add_argument(
        "--threshold", type=float, default=0.5, help="Classification threshold"
    )

    # ml-benchmark: Benchmark throughput
    ml_benchmark_parser = subparsers.add_parser(
        "ml-benchmark", help="Benchmark processing throughput"
    )
    ml_benchmark_parser.add_argument(
        "input", type=Path, help="Input CSV file"
    )
    ml_benchmark_parser.add_argument(
        "--sfreq", type=float, required=True, help="Sampling frequency"
    )
    ml_benchmark_parser.add_argument(
        "--outdir", type=Path, required=True, help="Output directory"
    )
    ml_benchmark_parser.add_argument(
        "--model-dir", type=Path, default=None, help="Model dir (optional)"
    )
    ml_benchmark_parser.add_argument(
        "--iterations", type=int, default=3, help="Timing iterations"
    )

    # report-impact: Before/After + Trend analysis
    report_impact_parser = subparsers.add_parser(
        "report-impact", help="Analyze before/after impact and temporal trends"
    )
    report_impact_parser.add_argument(
        "--run-dir", type=Path, required=True,
        help="Directory containing metrics_per_window.csv and predictions"
    )

    # ========== Streaming Commands ==========

    # stream-boards: List available boards
    stream_boards_parser = subparsers.add_parser(
        "stream-boards", help="List available EEG boards"
    )

    # stream-test: Quick test with synthetic board
    stream_test_parser = subparsers.add_parser(
        "stream-test", help="Test streaming with synthetic board (no hardware)"
    )
    stream_test_parser.add_argument(
        "--duration", type=float, default=5.0,
        help="Duration in seconds (default: 5)"
    )
    stream_test_parser.add_argument(
        "--outdir", type=Path, default=None,
        help="Output directory (optional, prints stats if not provided)"
    )

    # stream-record: Record from a board
    stream_record_parser = subparsers.add_parser(
        "stream-record", help="Record EEG data from a device"
    )
    stream_record_parser.add_argument(
        "--board", type=str, default="synthetic",
        help="Board name (default: synthetic). Use 'stream-boards' to list options."
    )
    stream_record_parser.add_argument(
        "--duration", type=float, required=True,
        help="Recording duration in seconds"
    )
    stream_record_parser.add_argument(
        "--outdir", type=Path, default=Path("recordings"),
        help="Output directory (default: recordings/)"
    )
    stream_record_parser.add_argument(
        "--serial-port", type=str, default=None,
        help="Serial port (for OpenBCI Cyton, e.g., COM3 or /dev/ttyUSB0)"
    )
    stream_record_parser.add_argument(
        "--mac-address", type=str, default=None,
        help="Bluetooth MAC address (for Muse, Ganglion)"
    )
    stream_record_parser.add_argument(
        "--filename", type=str, default=None,
        help="Output filename (default: auto-generated with timestamp)"
    )

    # stream-live: Live view of streaming data
    stream_live_parser = subparsers.add_parser(
        "stream-live", help="Live view of streaming data (prints stats)"
    )
    stream_live_parser.add_argument(
        "--board", type=str, default="synthetic",
        help="Board name (default: synthetic)"
    )
    stream_live_parser.add_argument(
        "--duration", type=float, default=10.0,
        help="Duration in seconds (default: 10)"
    )
    stream_live_parser.add_argument(
        "--serial-port", type=str, default=None,
        help="Serial port (for OpenBCI Cyton)"
    )
    stream_live_parser.add_argument(
        "--mac-address", type=str, default=None,
        help="Bluetooth MAC address (for Muse, Ganglion)"
    )

    return parser


def cmd_demo(args: argparse.Namespace) -> int:
    # all-in-one demo - generate data, process it, show results
    import time
    from .streaming import BrainFlowClient, BoardConfig
    from .preprocess.clean import clean_signal
    from .features.spectral import compute_psd_welch, bandpower, dominant_frequency, EEG_BANDS
    from .features.time_domain import extract_time_features
    from .viz.plots import plot_timeseries, plot_psd
    from .export.save import save_json, save_csv_rows

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    duration = args.duration

    print("=" * 60)
    print("  EEG PLATFORM DEMO - All-in-One")
    print("=" * 60)

    # Step 1: Generate synthetic EEG data
    print(f"\n[1/5] Generating {duration}s of synthetic EEG data...")
    config = BoardConfig.synthetic()

    try:
        client = BrainFlowClient(config)
        client.start()
        time.sleep(duration)
        result = client.get_data()
        client.release()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    data = result["data"]
    ch_names = result["ch_names"]
    sfreq = result["sfreq"]
    time_arr = result["time"]

    print(f"    Channels: {len(ch_names)}")
    print(f"    Samples: {data.shape[1]}")
    print(f"    Sample rate: {sfreq} Hz")

    # Step 2: Plot raw signal
    print(f"\n[2/5] Plotting raw signal...")
    raw_path = outdir / "raw.png"
    plot_timeseries(time_arr, data, ch_names, raw_path, max_seconds=min(5, duration))
    print(f"    Saved: {raw_path}")

    # Step 3: Clean the signal
    print(f"\n[3/5] Cleaning signal (filtering noise)...")
    cleaned = clean_signal(data, sfreq, bandpass=(1.0, 45.0), notch=60.0)

    cleaned_path = outdir / "cleaned.png"
    plot_timeseries(time_arr, cleaned, ch_names, cleaned_path, max_seconds=min(5, duration))
    print(f"    Saved: {cleaned_path}")

    # Step 4: Extract features
    print(f"\n[4/5] Analyzing brain wave patterns...")
    nperseg = min(256, data.shape[1])
    freqs, psd = compute_psd_welch(cleaned, sfreq, nperseg=nperseg)
    dom_freqs = dominant_frequency(psd, freqs, fmin=1.0, fmax=45.0)

    # Band powers
    bandpowers = {}
    for band_name, band_range in EEG_BANDS.items():
        bandpowers[band_name] = bandpower(psd, freqs, band_range)

    # Time features
    time_features = extract_time_features(cleaned)

    # Plot PSD
    psd_path = outdir / "psd.png"
    plot_psd(freqs, psd.T, ch_names, title="Power Spectral Density", save_path=psd_path)
    print(f"    Saved: {psd_path}")

    # Step 5: Save results
    print(f"\n[5/5] Saving results...")

    # Build metrics
    metrics = {
        "meta": {
            "source": "synthetic",
            "duration_sec": duration,
            "sfreq": sfreq,
            "n_channels": len(ch_names),
            "n_samples": data.shape[1],
        },
        "channels": {},
    }

    csv_rows = []
    for i, ch_name in enumerate(ch_names):
        ch_metrics = {"dominant_freq_hz": float(dom_freqs[i])}
        for band_name in EEG_BANDS:
            ch_metrics[f"bandpower_{band_name}"] = float(bandpowers[band_name][i])
        for feat_name, feat_values in time_features.items():
            ch_metrics[feat_name] = float(feat_values[i])
        metrics["channels"][ch_name] = ch_metrics
        csv_rows.append({"channel": ch_name, **ch_metrics})

    json_path = outdir / "metrics.json"
    save_json(json_path, metrics)
    print(f"    Saved: {json_path}")

    csv_path = outdir / "metrics.csv"
    save_csv_rows(csv_path, csv_rows)
    print(f"    Saved: {csv_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)

    print(f"\n  Brain Wave Power (averaged across channels):")
    print(f"  " + "-" * 40)
    for band_name in EEG_BANDS:
        avg_power = float(np.mean(bandpowers[band_name]))
        bar = "#" * min(30, int(avg_power / 100))
        print(f"    {band_name:8s}: {avg_power:8.1f} {bar}")

    print(f"\n  Output files created in: {outdir.resolve()}")
    print(f"    - raw.png      (original signal)")
    print(f"    - cleaned.png  (filtered signal)")
    print(f"    - psd.png      (frequency analysis)")
    print(f"    - metrics.json (all features)")
    print(f"    - metrics.csv  (spreadsheet format)")

    print(f"\n  Open the folder:")
    print(f"    explorer {outdir.resolve()}")
    print("=" * 60)

    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    # validates eeg data format
    import pandas as pd
    from .io.validation import validate_eeg_data, print_validation_report

    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        return 1

    columns = df.columns.tolist()
    n_samples = len(df)

    # Assume 250 Hz if we can't determine
    sfreq = 250.0
    if "time" in [c.lower() for c in columns]:
        time_col = [c for c in columns if c.lower() == "time"][0]
        time_data = pd.to_numeric(df[time_col], errors="coerce")
        if len(time_data) >= 2:
            dt = time_data.iloc[1] - time_data.iloc[0]
            if dt > 0:
                sfreq = 1.0 / dt

    result = validate_eeg_data(columns, n_samples, sfreq, strict=args.strict)
    print(print_validation_report(result))

    return 0 if result.is_valid else 1


def cmd_inspect(args: argparse.Namespace) -> int:
    # inspect eeg csv - shows meta, channels, and sample data
    from .io.load_csv import load_eeg_csv, EEGLoadError

    try:
        result = load_eeg_csv(str(args.input), sfreq=args.sfreq)
    except (FileNotFoundError, EEGLoadError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    data = result["data"]
    ch_names = result["ch_names"]
    time = result["time"]
    sfreq = result["sfreq"]
    meta = result["meta"]

    # Print meta
    print("=== Meta ===")
    print(f"  filepath:   {meta['filepath']}")
    print(f"  n_channels: {meta['n_channels']}")
    print(f"  n_samples:  {meta['n_samples']}")
    print(f"  sfreq:      {sfreq:.2f} Hz")
    print(f"  duration:   {meta['n_samples'] / sfreq:.2f} s")

    # Print first 3 channel names
    print("\n=== Channels (first 3) ===")
    for i, name in enumerate(ch_names[:3]):
        print(f"  [{i}] {name}")
    if len(ch_names) > 3:
        print(f"  ... and {len(ch_names) - 3} more")

    # Print first 5 samples
    print("\n=== Data (first 5 samples) ===")
    print(f"  {'time':>10}", end="")
    for name in ch_names[:3]:
        print(f"  {name:>12}", end="")
    print()

    for i in range(min(5, data.shape[1])):
        print(f"  {time[i]:>10.4f}", end="")
        for ch_idx in range(min(3, len(ch_names))):
            print(f"  {data[ch_idx, i]:>12.4f}", end="")
        print()

    return 0


def cmd_plot_raw(args: argparse.Namespace) -> int:
    # plots raw eeg time series
    from .io.load_csv import load_eeg_csv, EEGLoadError
    from .viz.plots import plot_timeseries

    try:
        result = load_eeg_csv(str(args.input), sfreq=args.sfreq)
    except (FileNotFoundError, EEGLoadError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    data = result["data"]
    ch_names = result["ch_names"]
    time = result["time"]
    sfreq = result["sfreq"]
    meta = result["meta"]

    print(f"Plotting: {args.input}")
    print(f"  Channels: {len(ch_names)}")
    print(f"  Samples: {meta['n_samples']}")
    print(f"  Duration: {meta['n_samples'] / sfreq:.2f} s")
    print(f"  Max seconds to plot: {args.max_seconds}")

    plot_timeseries(
        time=time,
        data=data,
        ch_names=ch_names,
        out_path=args.out,
        max_seconds=args.max_seconds,
        scale_uv=args.scale,
    )

    print(f"Saved: {args.out}")
    return 0


def cmd_clean_and_plot(args: argparse.Namespace) -> int:
    # cleans eeg data and plots
    from .io.load_csv import load_eeg_csv, EEGLoadError
    from .preprocess.clean import clean_signal
    from .preprocess.filters import FilterError
    from .viz.plots import plot_timeseries

    try:
        result = load_eeg_csv(str(args.input), sfreq=args.sfreq)
    except (FileNotFoundError, EEGLoadError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    data = result["data"]
    ch_names = result["ch_names"]
    time = result["time"]
    sfreq = result["sfreq"]
    meta = result["meta"]

    print(f"Loading: {args.input}")
    print(f"  Channels: {len(ch_names)}")
    print(f"  Samples: {meta['n_samples']}")
    print(f"  sfreq: {sfreq:.2f} Hz")

    # Prepare notch frequency (0 means disabled)
    notch_freq = args.notch if args.notch > 0 else None

    print(f"Cleaning signal...")
    print(f"  Bandpass: {args.l_freq}-{args.h_freq} Hz")
    print(f"  Notch: {notch_freq} Hz" if notch_freq else "  Notch: disabled")

    try:
        cleaned = clean_signal(
            data,
            sfreq,
            bandpass=(args.l_freq, args.h_freq),
            notch=notch_freq,
        )
    except FilterError as e:
        print(f"Filter error: {e}", file=sys.stderr)
        return 1

    print(f"Plotting cleaned signal...")
    plot_timeseries(
        time=time,
        data=cleaned,
        ch_names=ch_names,
        out_path=args.out,
        max_seconds=args.max_seconds,
        scale_uv=args.scale,
    )

    print(f"Saved: {args.out}")
    return 0


def cmd_features_spectral(args: argparse.Namespace) -> int:
    # extracts spectral features and saves to json
    import json
    from .io.load_csv import load_eeg_csv, EEGLoadError
    from .features.spectral import (
        compute_psd_welch,
        bandpower,
        dominant_frequency,
        EEG_BANDS,
    )

    try:
        result = load_eeg_csv(str(args.input), sfreq=args.sfreq)
    except (FileNotFoundError, EEGLoadError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    data = result["data"]
    ch_names = result["ch_names"]
    sfreq = result["sfreq"]
    meta = result["meta"]


    print(f"Loading: {args.input}")
    print(f"  Channels: {len(ch_names)}")
    print(f"  Samples: {meta['n_samples']}")
    print(f"  sfreq: {sfreq:.2f} Hz")

    print(f"Computing PSD (nperseg={args.nperseg})...")
    freqs, psd = compute_psd_welch(data, sfreq, nperseg=args.nperseg)

    print("Extracting spectral features...")
    dom_freqs = dominant_frequency(psd, freqs, fmin=1.0, fmax=45.0)

    # Compute band powers
    bandpowers = {}
    for band_name, band_range in EEG_BANDS.items():
        bandpowers[band_name] = bandpower(psd, freqs, band_range)

    # Build output structure
    output = {
        "meta": {
            "filepath": meta["filepath"],
            "sfreq": sfreq,
            "n_channels": meta["n_channels"],
            "n_samples": meta["n_samples"],
            "nperseg": args.nperseg,
        },
        "channels": {},
    }

    for i, ch_name in enumerate(ch_names):
        ch_data = {
            "dominant_freq_hz": float(dom_freqs[i]),
            "bandpowers": {
                band: float(bandpowers[band][i])
                for band in EEG_BANDS
            },
        }
        output["channels"][ch_name] = ch_data

    # Validate all values are finite
    all_finite = True
    for ch_name, ch_data in output["channels"].items():
        if not np.isfinite(ch_data["dominant_freq_hz"]):
            print(f"Warning: {ch_name} has non-finite dominant_freq", file=sys.stderr)
            all_finite = False
        for band, power in ch_data["bandpowers"].items():
            if not np.isfinite(power):
                print(f"Warning: {ch_name} {band} has non-finite power", file=sys.stderr)
                all_finite = False

    # Save JSON
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved: {args.out}")

    if not all_finite:
        return 1
    return 0


def cmd_features_time(args: argparse.Namespace) -> int:
    # extracts time-domain features and saves to json
    import json
    from .io.load_csv import load_eeg_csv, EEGLoadError
    from .features.time_domain import extract_time_features

    try:
        result = load_eeg_csv(str(args.input), sfreq=args.sfreq)
    except (FileNotFoundError, EEGLoadError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    data = result["data"]
    ch_names = result["ch_names"]
    sfreq = result["sfreq"]
    meta = result["meta"]

    print(f"Loading: {args.input}")
    print(f"  Channels: {len(ch_names)}")
    print(f"  Samples: {meta['n_samples']}")
    print(f"  sfreq: {sfreq:.2f} Hz")

    print("Extracting time-domain features...")
    features = extract_time_features(data)

    # Build output structure
    output = {
        "meta": {
            "filepath": meta["filepath"],
            "sfreq": sfreq,
            "n_channels": meta["n_channels"],
            "n_samples": meta["n_samples"],
        },
        "channels": {},
    }

    for i, ch_name in enumerate(ch_names):
        ch_data = {
            feat_name: float(feat_values[i])
            for feat_name, feat_values in features.items()
        }
        output["channels"][ch_name] = ch_data

    # Validate all values are finite
    all_finite = True
    for ch_name, ch_data in output["channels"].items():
        for feat_name, feat_value in ch_data.items():
            if not np.isfinite(feat_value):
                print(f"Warning: {ch_name} {feat_name} is non-finite", file=sys.stderr)
                all_finite = False

    # Save JSON
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved: {args.out}")

    if not all_finite:
        return 1
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    # runs full eeg processing pipeline
    from .io.load_csv import load_eeg_csv, EEGLoadError
    from .preprocess.clean import clean_signal
    from .preprocess.filters import FilterError
    from .features.spectral import (
        compute_psd_welch,
        bandpower,
        dominant_frequency,
        EEG_BANDS,
    )
    from .features.time_domain import extract_time_features
    from .viz.plots import plot_timeseries, plot_psd
    from .export.save import save_json, save_csv_rows

    # Create output directory
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    print(f"=== Loading: {args.input} ===")
    try:
        result = load_eeg_csv(str(args.input), sfreq=args.sfreq)
    except (FileNotFoundError, EEGLoadError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    data = result["data"]
    ch_names = result["ch_names"]
    time = result["time"]
    sfreq = result["sfreq"]
    meta = result["meta"]

    print(f"  Channels: {len(ch_names)}")
    print(f"  Samples: {meta['n_samples']}")
    print(f"  sfreq: {sfreq:.2f} Hz")
    print(f"  Duration: {meta['n_samples'] / sfreq:.2f} s")

    # 2. Plot raw signal
    print("\n=== Plotting raw signal ===")
    raw_plot_path = outdir / "raw.png"
    plot_timeseries(
        time=time,
        data=data,
        ch_names=ch_names,
        out_path=raw_plot_path,
        max_seconds=args.max_seconds,
    )
    print(f"  Saved: {raw_plot_path}")

    # 3. Clean signal
    print("\n=== Cleaning signal ===")
    notch_freq = args.notch if args.notch > 0 else None
    print(f"  Bandpass: {args.l_freq}-{args.h_freq} Hz")
    print(f"  Notch: {notch_freq} Hz" if notch_freq else "  Notch: disabled")

    try:
        cleaned = clean_signal(
            data,
            sfreq,
            bandpass=(args.l_freq, args.h_freq),
            notch=notch_freq,
        )
    except FilterError as e:
        print(f"Filter error: {e}", file=sys.stderr)
        return 1

    # 4. Plot cleaned signal
    print("\n=== Plotting cleaned signal ===")
    cleaned_plot_path = outdir / "cleaned.png"
    plot_timeseries(
        time=time,
        data=cleaned,
        ch_names=ch_names,
        out_path=cleaned_plot_path,
        max_seconds=args.max_seconds,
    )
    print(f"  Saved: {cleaned_plot_path}")

    # 5. Compute spectral features
    print("\n=== Computing spectral features ===")
    nperseg = min(512, meta["n_samples"])
    freqs, psd = compute_psd_welch(cleaned, sfreq, nperseg=nperseg)
    dom_freqs = dominant_frequency(psd, freqs, fmin=1.0, fmax=45.0)

    bandpowers = {}
    for band_name, band_range in EEG_BANDS.items():
        bandpowers[band_name] = bandpower(psd, freqs, band_range)
        print(f"  {band_name}: computed")

    # 6. Plot PSD
    print("\n=== Plotting PSD ===")
    psd_plot_path = outdir / "psd.png"
    plot_psd(
        freqs=freqs,
        psd=psd.T,  # plot_psd expects (n_freqs, n_channels)
        channels=ch_names,
        title="Power Spectral Density",
        save_path=psd_plot_path,
    )
    print(f"  Saved: {psd_plot_path}")

    # 7. Compute time-domain features
    print("\n=== Computing time-domain features ===")
    time_features = extract_time_features(cleaned)
    for feat_name in time_features:
        print(f"  {feat_name}: computed")

    # 8. Build metrics structure
    print("\n=== Building metrics ===")
    metrics = {
        "meta": {
            "input_file": str(args.input.resolve()),
            "sfreq": sfreq,
            "n_channels": meta["n_channels"],
            "n_samples": meta["n_samples"],
            "duration_s": meta["n_samples"] / sfreq,
            "preprocessing": {
                "bandpass": [args.l_freq, args.h_freq],
                "notch": notch_freq,
            },
        },
        "channels": {},
    }

    # Build per-channel metrics
    csv_rows = []
    for i, ch_name in enumerate(ch_names):
        ch_metrics = {
            "dominant_freq_hz": float(dom_freqs[i]),
        }

        # Add band powers
        for band_name in EEG_BANDS:
            ch_metrics[f"bandpower_{band_name}"] = float(bandpowers[band_name][i])

        # Add time-domain features
        for feat_name, feat_values in time_features.items():
            ch_metrics[feat_name] = float(feat_values[i])

        metrics["channels"][ch_name] = ch_metrics

        # Build CSV row
        csv_row = {"channel": ch_name}
        csv_row.update(ch_metrics)
        csv_rows.append(csv_row)

    # 9. Save metrics.json
    print("\n=== Saving outputs ===")
    json_path = outdir / "metrics.json"
    save_json(json_path, metrics)
    print(f"  Saved: {json_path}")

    # 10. Save metrics.csv
    csv_path = outdir / "metrics.csv"
    save_csv_rows(csv_path, csv_rows)
    print(f"  Saved: {csv_path}")

    # Validate all values are finite
    all_finite = True
    for ch_name, ch_data in metrics["channels"].items():
        for feat_name, feat_value in ch_data.items():
            if not np.isfinite(feat_value):
                print(f"Warning: {ch_name} {feat_name} is non-finite", file=sys.stderr)
                all_finite = False

    print(f"\n=== Done ===")
    print(f"Output directory: {outdir}")
    print(f"Files created:")
    print(f"  - raw.png")
    print(f"  - cleaned.png")
    print(f"  - psd.png")
    print(f"  - metrics.json")
    print(f"  - metrics.csv")

    if not all_finite:
        return 1
    return 0


def cmd_process(args: argparse.Namespace) -> int:
    # processes eeg data
    from .io.load_csv import load_eeg_csv, EEGLoadError
    from .preprocess.clean import clean_signal
    from .features.spectral import compute_band_power, EEG_BANDS
    from .features.time_domain import compute_rms
    from .export.save import save_csv
    import pandas as pd

    try:
        result = load_eeg_csv(str(args.input), sfreq=args.sfreq)
    except (FileNotFoundError, EEGLoadError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    data = result["data"]
    ch_names = result["ch_names"]
    sfreq = result["sfreq"]

    print(f"Loading: {args.input}")
    print(f"Channels: {ch_names}")
    print(f"Samples: {data.shape[1]}")

    print("Preprocessing...")
    cleaned = clean_signal(data, sfreq, bandpass=(args.lowcut, args.highcut))

    print("Extracting features...")
    # Transpose for feature functions that expect (samples, channels)
    data_t = cleaned.T
    rms = compute_rms(data_t)
    for i, ch in enumerate(ch_names):
        print(f"  {ch} RMS: {rms[i]:.4f}")

    print("Computing band powers...")
    for band_name in EEG_BANDS:
        power = compute_band_power(data_t, sfreq, band_name)
        print(f"  {band_name}: {power}")

    args.output.mkdir(parents=True, exist_ok=True)
    output_file = args.output / "processed.csv"
    df_out = pd.DataFrame(cleaned.T, columns=ch_names)
    save_csv(df_out, output_file)
    print(f"Saved: {output_file}")

    return 0


def cmd_info(args: argparse.Namespace) -> int:
    # shows eeg file info (legacy)
    from .io.load_csv import load_eeg_csv, EEGLoadError

    try:
        result = load_eeg_csv(str(args.input))
    except EEGLoadError:
        # Try with default sfreq
        try:
            result = load_eeg_csv(str(args.input), sfreq=256.0)
        except (FileNotFoundError, EEGLoadError) as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    print(f"File: {args.input}")
    print(f"Channels: {result['ch_names']}")
    print(f"Samples: {result['meta']['n_samples']}")
    print(f"Sampling rate: {result['sfreq']:.1f} Hz")
    print(f"Duration: {result['meta']['n_samples'] / result['sfreq']:.2f} s")

    return 0


def cmd_run_windowed(args: argparse.Namespace) -> int:
    # runs pipeline with window segmentation
    from .io.load_csv import load_eeg_csv, EEGLoadError
    from .preprocess.clean import clean_signal
    from .ml.featureset import extract_all_window_features
    from .export.save import save_csv_rows

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"=== Loading: {args.input} ===")
    try:
        result = load_eeg_csv(str(args.input), sfreq=args.sfreq)
    except (FileNotFoundError, EEGLoadError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    data = result["data"]
    ch_names = result["ch_names"]
    sfreq = result["sfreq"]

    print(f"  Channels: {len(ch_names)}")
    print(f"  Samples: {data.shape[1]}")

    print("\n=== Cleaning signal ===")
    cleaned = clean_signal(data, sfreq)

    print("\n=== Extracting window features ===")
    print(f"  Window: {args.window_sec}s, Overlap: {args.overlap}")

    df = extract_all_window_features(
        cleaned, sfreq,
        window_sec=args.window_sec,
        overlap=args.overlap,
        ch_names=ch_names,
        recording_id=str(args.input.stem),
    )

    print(f"  Windows extracted: {len(df)}")

    # Save features
    csv_path = outdir / "metrics_per_window.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n=== Saved: {csv_path} ===")

    return 0


def cmd_ml_train(args: argparse.Namespace) -> int:
    # trains ml artifact classifier
    import pandas as pd
    from .ml.featureset import load_window_features, select_ml_features
    from .ml.artifact_labels import generate_weak_labels, load_artifact_labels, save_weak_labels
    from .ml.train_artifact import train_artifact_classifier

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"=== Loading features: {args.features} ===")
    try:
        df = load_window_features(args.features)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"  Windows: {len(df)}")

    # Get labels
    if args.labels and args.labels.exists():
        print(f"=== Loading manual labels: {args.labels} ===")
        labels = load_artifact_labels(args.labels, df)
    else:
        print("=== Generating weak labels ===")
        labels = generate_weak_labels(df)
        weak_path = outdir / "weak_labels.csv"
        save_weak_labels(df, labels, weak_path)
        print(f"  Saved: {weak_path}")

    n_clean = int(np.sum(labels == 0))
    n_artifact = int(np.sum(labels == 1))
    print(f"  Clean: {n_clean}, Artifact: {n_artifact}")

    # Select features
    print("\n=== Selecting features ===")
    X, feature_names = select_ml_features(df)
    print(f"  Features: {len(feature_names)}")

    # Train
    print("\n=== Training models ===")
    result = train_artifact_classifier(
        X, labels, feature_names, outdir,
        model_type=args.model_type,
    )

    meta = result["meta"]
    print(f"\n=== Results ===")
    print(f"  Best model: {meta['best_model']}")
    print(f"  Test F1: {meta['best_test_f1']:.3f}")
    print(f"  Model saved: {result['model_path']}")

    return 0


def cmd_ml_eval(args: argparse.Namespace) -> int:
    # evaluates trained artifact model
    import pandas as pd
    from .ml.featureset import load_window_features, select_ml_features
    from .ml.artifact_labels import generate_weak_labels, load_artifact_labels
    from .ml.train_artifact import load_trained_model
    from .ml.eval_artifact import evaluate_artifact_model, evaluate_with_report

    print(f"=== Loading model: {args.model_dir} ===")
    try:
        model, scaler, meta = load_trained_model(args.model_dir)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"=== Loading features: {args.features} ===")
    df = load_window_features(args.features)

    # Get labels
    if args.labels and args.labels.exists():
        labels = load_artifact_labels(args.labels, df)
    else:
        labels = generate_weak_labels(df)

    # Select features using model's feature list
    feature_names = meta.get("feature_names")
    X, _ = select_ml_features(df, feature_names)

    print("\n=== Evaluation ===")
    metrics = evaluate_artifact_model(
        model, scaler, X, labels,
        output_dir=args.model_dir,
        threshold=args.threshold,
    )

    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall: {metrics['recall']:.3f}")
    print(f"  F1: {metrics['f1']:.3f}")

    if "roc_auc" in metrics:
        print(f"  ROC-AUC: {metrics['roc_auc']:.3f}")

    print("\n" + evaluate_with_report(model, scaler, X, labels))

    return 0


def cmd_ml_predict(args: argparse.Namespace) -> int:
    # generates artifact predictions
    import pandas as pd
    from .ml.featureset import load_window_features
    from .ml.predict_artifact import generate_window_predictions
    from .ml.improvement import compute_improvement_metrics, generate_improvement_summary

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"=== Loading features: {args.features} ===")
    df = load_window_features(args.features)
    print(f"  Windows: {len(df)}")

    print(f"\n=== Generating predictions ===")
    summary = generate_window_predictions(
        df, args.model_dir, outdir,
        threshold=args.threshold,
    )

    print(f"  Clean: {summary['n_clean_windows']}")
    print(f"  Artifact: {summary['n_artifact_windows']}")
    print(f"  Saved: {summary['predictions_path']}")

    # Compute improvement metrics
    print(f"\n=== Computing improvement metrics ===")
    pred_df = pd.read_csv(summary["predictions_path"])
    df_clean = pred_df[pred_df["is_clean"] == True]

    improvement = compute_improvement_metrics(df, df_clean, outdir)

    print(generate_improvement_summary(improvement))

    return 0


def cmd_ml_benchmark(args: argparse.Namespace) -> int:
    # benchmarks processing throughput
    from .io.load_csv import load_eeg_csv, EEGLoadError
    from .ml.benchmark import run_benchmark, generate_benchmark_summary

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"=== Loading: {args.input} ===")
    try:
        result = load_eeg_csv(str(args.input), sfreq=args.sfreq)
    except (FileNotFoundError, EEGLoadError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    data = result["data"]
    sfreq = result["sfreq"]

    print(f"\n=== Running benchmark ({args.iterations} iterations) ===")
    metrics = run_benchmark(
        data, sfreq,
        n_iterations=args.iterations,
        include_ml=args.model_dir is not None,
        model_dir=args.model_dir,
    )

    # Save results
    import json
    metrics_path = outdir / "performance_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(generate_benchmark_summary(metrics))
    print(f"\nSaved: {metrics_path}")

    return 0


def cmd_report_impact(args: argparse.Namespace) -> int:
    # analyzes before/after impact and temporal trends
    from .analysis.impact import run_impact_analysis

    run_dir = args.run_dir

    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}", file=sys.stderr)
        return 1

    print(f"=== Impact Analysis: {run_dir} ===")

    try:
        result = run_impact_analysis(run_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"\n  Total windows:    {result['n_total']}")
    print(f"  Clean windows:    {result['n_clean']}")
    print(f"  Rejected windows: {result['n_rejected']}")
    print(f"  Features analyzed: {len(result['features_analyzed'])}")

    print(f"\n=== Output Files ===")
    for name, path in result['output_paths'].items():
        print(f"  {name}: {path}")

    # Print the summary
    summary_path = result['output_paths'].get('improvement_summary')
    if summary_path and summary_path.exists():
        print()
        with open(summary_path) as f:
            print(f.read())

    return 0


def cmd_stream_boards(args: argparse.Namespace) -> int:
    # lists available eeg boards
    from .streaming import list_boards

    boards = list_boards()

    print("=== Available EEG Boards ===\n")
    print(f"{'Name':<15} {'Channels':<10} {'Sample Rate':<12} {'Description'}")
    print("-" * 70)

    for board in boards:
        print(
            f"{board['key']:<15} "
            f"{board['channels']:<10} "
            f"{board['sfreq']} Hz{'':<5} "
            f"{board['description']}"
        )

    print("\n=== Connection Requirements ===")
    print(f"{'Board':<15} {'Serial Port':<15} {'MAC Address'}")
    print("-" * 45)

    for board in boards:
        serial = "Required" if board['requires_serial'] else "-"
        mac = "Required" if board['requires_mac'] else "-"
        print(f"{board['key']:<15} {serial:<15} {mac}")

    print("\n=== Usage Examples ===")
    print("  # Test with synthetic board (no hardware)")
    print("  eeg-platform stream-test --duration 5")
    print("")
    print("  # Record from synthetic board")
    print("  eeg-platform stream-record --board synthetic --duration 30")
    print("")
    print("  # Record from OpenBCI Cyton")
    print("  eeg-platform stream-record --board cyton --serial-port COM3 --duration 60")
    print("")
    print("  # Record from Muse 2")
    print("  eeg-platform stream-record --board muse_2 --duration 60")

    return 0


def cmd_stream_test(args: argparse.Namespace) -> int:
    # tests streaming with synthetic board
    import time
    from .streaming import BrainFlowClient, BoardConfig, StreamingError

    print("=== Streaming Test (Synthetic Board) ===")
    print(f"Duration: {args.duration}s\n")

    config = BoardConfig.synthetic()

    try:
        client = BrainFlowClient(config)
        print(f"Board: {config.board_name}")
        print(f"Sample Rate: {client.sfreq} Hz")
        print(f"Channels: {client.n_channels}")
        print(f"Channel Names: {', '.join(client.channel_names)}")
        print("")

        print("Starting stream...")
        client.start()

        start_time = time.time()
        total_samples = 0

        while time.time() - start_time < args.duration:
            time.sleep(0.5)
            count = client.get_data_count()
            elapsed = time.time() - start_time
            print(f"  {elapsed:.1f}s: {count} samples in buffer", end="\r")

        # Get all data
        data = client.get_data()
        total_samples = data["data"].shape[1]

        print(f"\n\nStream stopped.")
        print(f"  Total samples: {total_samples}")
        print(f"  Duration: {total_samples / client.sfreq:.2f}s")
        print(f"  Data shape: {data['data'].shape}")

        # Print sample statistics
        print(f"\n=== Sample Statistics ===")
        for i, ch_name in enumerate(data["ch_names"][:4]):  # First 4 channels
            ch_data = data["data"][i, :]
            print(f"  {ch_name}: mean={np.mean(ch_data):.2f}, std={np.std(ch_data):.2f}")

        if len(data["ch_names"]) > 4:
            print(f"  ... and {len(data['ch_names']) - 4} more channels")

        client.release()

        # Save if output directory provided
        if args.outdir:
            from .streaming import record_session
            args.outdir.mkdir(parents=True, exist_ok=True)
            print(f"\nRecording to {args.outdir}...")
            path = record_session(
                board_name="synthetic",
                duration_sec=args.duration,
                output_dir=args.outdir,
            )
            print(f"Saved: {path}")

        print("\nTest completed successfully!")
        return 0

    except StreamingError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


def cmd_stream_record(args: argparse.Namespace) -> int:
    # records eeg data from a device
    from .streaming import record_session, StreamingError

    print(f"=== Recording from {args.board} ===")
    print(f"Duration: {args.duration}s")
    print(f"Output: {args.outdir}/")
    print("")

    try:
        path = record_session(
            board_name=args.board,
            duration_sec=args.duration,
            output_dir=args.outdir,
            serial_port=args.serial_port,
            mac_address=args.mac_address,
            verbose=True,
        )

        print(f"\nRecording complete!")
        print(f"  CSV: {path}")
        print(f"  Metadata: {path.with_suffix('.json')}")

        # Show quick stats
        import pandas as pd
        df = pd.read_csv(path)
        n_samples = len(df)
        n_channels = len(df.columns) - 1  # Exclude time column
        duration = df["time"].iloc[-1] if "time" in df.columns else n_samples / 250

        print(f"\n=== Recording Summary ===")
        print(f"  Samples: {n_samples}")
        print(f"  Channels: {n_channels}")
        print(f"  Duration: {duration:.2f}s")

        return 0

    except StreamingError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


def cmd_stream_live(args: argparse.Namespace) -> int:
    # live view of streaming data
    import time
    from .streaming import BrainFlowClient, BoardConfig, StreamingError

    print(f"=== Live Stream: {args.board} ===")
    print(f"Duration: {args.duration}s")
    print("")

    # Build config
    kwargs = {}
    if args.serial_port:
        kwargs["serial_port"] = args.serial_port
    if args.mac_address:
        kwargs["mac_address"] = args.mac_address

    try:
        config = BoardConfig.from_name(args.board, **kwargs)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    try:
        client = BrainFlowClient(config)

        print(f"Board: {config.board_name}")
        print(f"Sample Rate: {client.sfreq} Hz")
        print(f"Channels: {client.n_channels}")
        print("")

        client.start()
        print("Streaming... (Ctrl+C to stop)\n")

        sfreq = client.sfreq
        start_time = time.time()
        update_interval = 1.0  # Update every second

        while time.time() - start_time < args.duration:
            time.sleep(update_interval)

            # Get recent data (last second)
            n_samples = int(sfreq * update_interval)
            if client.get_data_count() >= n_samples:
                data = client.get_data(n_samples)
                eeg = data["data"]

                elapsed = time.time() - start_time
                print(f"[{elapsed:6.1f}s] ", end="")

                # Print RMS for each channel (first 4)
                for i, ch_name in enumerate(data["ch_names"][:4]):
                    rms = np.sqrt(np.mean(eeg[i, :] ** 2))
                    print(f"{ch_name}:{rms:6.1f}uV  ", end="")

                print("")

        client.release()
        print("\nStream ended.")
        return 0

    except StreamingError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n\nStopped by user.")
        if client:
            client.release()
        return 0
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    # main entry point
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "demo":
        return cmd_demo(args)
    elif args.command == "validate":
        return cmd_validate(args)
    elif args.command == "inspect":
        return cmd_inspect(args)
    elif args.command == "plot-raw":
        return cmd_plot_raw(args)
    elif args.command == "clean-and-plot":
        return cmd_clean_and_plot(args)
    elif args.command == "features-spectral":
        return cmd_features_spectral(args)
    elif args.command == "features-time":
        return cmd_features_time(args)
    elif args.command == "run":
        return cmd_run(args)
    elif args.command == "process":
        return cmd_process(args)
    elif args.command == "info":
        return cmd_info(args)
    # ML commands
    elif args.command == "run-windowed":
        return cmd_run_windowed(args)
    elif args.command == "ml-train":
        return cmd_ml_train(args)
    elif args.command == "ml-eval":
        return cmd_ml_eval(args)
    elif args.command == "ml-predict":
        return cmd_ml_predict(args)
    elif args.command == "ml-benchmark":
        return cmd_ml_benchmark(args)
    elif args.command == "report-impact":
        return cmd_report_impact(args)
    # Streaming commands
    elif args.command == "stream-boards":
        return cmd_stream_boards(args)
    elif args.command == "stream-test":
        return cmd_stream_test(args)
    elif args.command == "stream-record":
        return cmd_stream_record(args)
    elif args.command == "stream-live":
        return cmd_stream_live(args)

    return 0


if __name__ == "__main__": #
    sys.exit(main())
