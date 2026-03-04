# predicts artifacts and assesses window quality

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

try:
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class PredictionError(Exception):
    # prediction failed
    pass


def predict_artifacts(
    X: np.ndarray,
    model_dir: str | Path,
    threshold: float = 0.5,
    feature_names: Optional[list[str]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    # predicts artifact probability for each window
    if not SKLEARN_AVAILABLE:
        raise PredictionError("scikit-learn/joblib not installed")

    model_dir = Path(model_dir)
    model_path = model_dir / "model.joblib"
    scaler_path = model_dir / "scaler.joblib"
    meta_path = model_dir / "model_meta.json"

    if not model_path.exists():
        raise PredictionError(f"Model not found: {model_path}")

    # Load model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    # Validate feature count
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        expected_features = meta.get("n_features")
        if expected_features and X.shape[1] != expected_features:
            raise PredictionError(
                f"Feature count mismatch: expected {expected_features}, "
                f"got {X.shape[1]}"
            )

    # Scale features
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X

    # Get probabilities
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_scaled)[:, 1]
    else:
        # Fall back to decision function or hard predictions
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X_scaled)
            # Sigmoid to convert to probability-like score
            probabilities = 1 / (1 + np.exp(-scores))
        else:
            probabilities = model.predict(X_scaled).astype(float)

    # Apply threshold
    predictions = (probabilities >= threshold).astype(int)

    return probabilities, predictions


def generate_window_predictions(
    df: pd.DataFrame,
    model_dir: str | Path,
    output_dir: str | Path,
    threshold: float = 0.5,
    feature_columns: Optional[list[str]] = None,
) -> dict:
    # generates and saves predictions for all windows
    from .featureset import select_ml_features

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model metadata to get expected features
    model_dir = Path(model_dir)
    meta_path = model_dir / "model_meta.json"

    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        feature_columns = meta.get("feature_names", feature_columns)

    # Select features
    X, used_features = select_ml_features(df, feature_columns)

    # Get predictions
    probabilities, predictions = predict_artifacts(
        X, model_dir, threshold=threshold
    )

    # Build predictions DataFrame
    pred_df = df.copy()
    pred_df["artifact_probability"] = probabilities
    pred_df["artifact_prediction"] = predictions
    pred_df["is_clean"] = predictions == 0

    # Save predictions CSV
    pred_path = output_dir / "window_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    # Compute quality summary
    n_total = len(predictions)
    n_clean = int(np.sum(predictions == 0))
    n_artifact = int(np.sum(predictions == 1))

    summary = {
        "threshold": threshold,
        "n_total_windows": n_total,
        "n_clean_windows": n_clean,
        "n_artifact_windows": n_artifact,
        "clean_rate": n_clean / max(n_total, 1),
        "artifact_rate": n_artifact / max(n_total, 1),
        "mean_artifact_probability": float(np.mean(probabilities)),
        "median_artifact_probability": float(np.median(probabilities)),
        "model_dir": str(model_dir),
        "predictions_path": str(pred_path),
    }

    # Save quality summary
    summary_path = output_dir / "quality_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    summary["summary_path"] = str(summary_path)

    return summary


def filter_clean_windows(
    df: pd.DataFrame,
    predictions: np.ndarray,
) -> pd.DataFrame:
    # keeps only clean windows from df
    clean_mask = predictions == 0
    return df[clean_mask].reset_index(drop=True)


def get_clean_window_indices(
    predictions: np.ndarray,
) -> np.ndarray:
    # gets indices of clean windows
    return np.where(predictions == 0)[0]
