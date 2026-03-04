# evaluates artifact detection models

import json
import numpy as np
from pathlib import Path
from typing import Optional

try:
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        classification_report,
        roc_auc_score,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class EvalError(Exception):
    # eval went wrong
    pass


def evaluate_artifact_model(
    model,
    scaler,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: Optional[str | Path] = None,
    threshold: float = 0.5,
) -> dict:
    # evaluates model on test set, returns metrics dict
    if not SKLEARN_AVAILABLE:
        raise EvalError("scikit-learn not installed")

    if len(X_test) == 0:
        raise EvalError("Empty test set")

    # Scale features
    if scaler is not None:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test

    # Get predictions
    y_pred = model.predict(X_test_scaled)

    # Get probabilities if available
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        # Apply custom threshold
        y_pred_thresh = (y_prob >= threshold).astype(int)
    else:
        y_pred_thresh = y_pred

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred_thresh)
    precision = precision_score(y_test, y_pred_thresh, zero_division=0)
    recall = recall_score(y_test, y_pred_thresh, zero_division=0)
    f1 = f1_score(y_test, y_pred_thresh, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_thresh)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "threshold": threshold,
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "n_test_samples": len(y_test),
        "n_positive": int(np.sum(y_test == 1)),
        "n_negative": int(np.sum(y_test == 0)),
    }

    # ROC-AUC if probabilities available
    if y_prob is not None and len(np.unique(y_test)) == 2:
        try:
            roc_auc = roc_auc_score(y_test, y_prob)
            metrics["roc_auc"] = float(roc_auc)
        except ValueError:
            pass

    # Save metrics
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = output_dir / "model_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        metrics["metrics_path"] = str(metrics_path)

    return metrics


def evaluate_with_report(
    model,
    scaler,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> str:
    # generates sklearn classification report string
    if not SKLEARN_AVAILABLE:
        raise EvalError("scikit-learn not installed")

    if scaler is not None:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test

    y_pred = model.predict(X_test_scaled)

    report = classification_report(
        y_test, y_pred,
        target_names=["clean", "artifact"],
        zero_division=0
    )

    return report


def compute_threshold_sweep(
    model,
    scaler,
    X_test: np.ndarray,
    y_test: np.ndarray,
    thresholds: Optional[list[float]] = None,
) -> list[dict]:
    # tries different thresholds and computes metrics for each
    if not SKLEARN_AVAILABLE:
        raise EvalError("scikit-learn not installed")

    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    if scaler is not None:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test

    if not hasattr(model, "predict_proba"):
        raise EvalError("Model does not support predict_proba")

    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    results = []
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        results.append({
            "threshold": thresh,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        })

    return results
