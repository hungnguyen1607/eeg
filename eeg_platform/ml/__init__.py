# ml stuff for artifact detection and quality checks

# Core modules (no sklearn dependency)
from .featureset import load_window_features, select_ml_features, ML_FEATURE_COLUMNS
from .artifact_labels import generate_weak_labels, load_artifact_labels
from .improvement import compute_improvement_metrics
from .benchmark import run_benchmark

# Modules requiring sklearn
try:
    from .train_artifact import train_artifact_classifier
    from .eval_artifact import evaluate_artifact_model
    from .predict_artifact import predict_artifacts
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    train_artifact_classifier = None
    evaluate_artifact_model = None
    predict_artifacts = None

__all__ = [
    "load_window_features",
    "select_ml_features",
    "ML_FEATURE_COLUMNS",
    "generate_weak_labels",
    "load_artifact_labels",
    "train_artifact_classifier",
    "evaluate_artifact_model",
    "predict_artifacts",
    "compute_improvement_metrics",
    "run_benchmark",
    "SKLEARN_AVAILABLE",
]
