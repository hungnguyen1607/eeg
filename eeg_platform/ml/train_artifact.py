# trains artifact classifier models

import json
import numpy as np
from pathlib import Path
from typing import Optional
from datetime import datetime

try:
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class TrainingError(Exception):
    # training failed
    pass


def check_sklearn():
    # makes sure sklearn is installed
    if not SKLEARN_AVAILABLE:
        raise TrainingError(
            "scikit-learn not installed. Run: pip install scikit-learn"
        )


def train_artifact_classifier(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    output_dir: str | Path,
    model_type: str = "auto",
    test_size: float = 0.2,
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict:
    # trains artifact classifier, saves model and returns metrics
    check_sklearn()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate inputs
    if len(X) < 10:
        raise TrainingError(f"Need at least 10 samples, got {len(X)}")

    if len(np.unique(y)) < 2:
        raise TrainingError("Need both classes (0 and 1) in training data")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define models
    models = {}
    if model_type in ("auto", "logreg"):
        models["logreg"] = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            class_weight="balanced",
        )
    if model_type in ("auto", "rf"):
        models["rf"] = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        )

    if not models:
        raise TrainingError(f"Unknown model_type: {model_type}")

    # Train and evaluate each model
    results = {}
    best_model_name = None
    best_f1 = -1

    for name, model in models.items():
        # Cross-validation on training set
        cv_scores = cross_val_score(
            model, X_train_scaled, y_train,
            cv=min(cv_folds, len(X_train) // 2),
            scoring="f1"
        )

        # Fit on full training set
        model.fit(X_train_scaled, y_train)

        # Evaluate on test set
        y_pred = model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, zero_division=0)

        results[name] = {
            "cv_f1_mean": float(np.mean(cv_scores)),
            "cv_f1_std": float(np.std(cv_scores)),
            "test_accuracy": float(test_accuracy),
            "test_f1": float(test_f1),
        }

        if test_f1 > best_f1:
            best_f1 = test_f1
            best_model_name = name

    # Save best model
    best_model = models[best_model_name]
    model_path = output_dir / "model.joblib"
    scaler_path = output_dir / "scaler.joblib"

    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)

    # Save metadata
    meta = {
        "model_type": best_model_name,
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "class_distribution": {
            "train": {
                "clean": int(np.sum(y_train == 0)),
                "artifact": int(np.sum(y_train == 1)),
            },
            "test": {
                "clean": int(np.sum(y_test == 0)),
                "artifact": int(np.sum(y_test == 1)),
            },
        },
        "model_results": results,
        "best_model": best_model_name,
        "best_test_f1": float(best_f1),
        "training_timestamp": datetime.now().isoformat(),
        "random_state": random_state,
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
    }

    meta_path = output_dir / "model_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "model": best_model,
        "scaler": scaler,
        "model_path": model_path,
        "scaler_path": scaler_path,
        "meta_path": meta_path,
        "meta": meta,
        "X_test": X_test,
        "y_test": y_test,
    }


def load_trained_model(
    model_dir: str | Path,
) -> tuple:
    # loads trained model and scaler from dir
    check_sklearn()

    model_dir = Path(model_dir)
    model_path = model_dir / "model.joblib"
    scaler_path = model_dir / "scaler.joblib"
    meta_path = model_dir / "model_meta.json"

    if not model_path.exists():
        raise TrainingError(f"Model not found: {model_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    return model, scaler, meta
