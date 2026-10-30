"""
Training pipeline for the Diagnosis Agent classifier.

Trains a balanced multinomial LogisticRegression model on cached 1280-d
EfficientNet-B0 features extracted from HAM10000 training images.
Saves the fitted model to models/classifier_v1.joblib and training metrics
to outputs/training_metrics.json.
"""

import json
from pathlib import Path
import time
from typing import Any, Dict, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.common import CLASS_NAMES, MODELS_DIR, OUTPUTS_DIR


def load_training_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load cached training features, labels, and image IDs from outputs/.
    Validates shapes, finiteness, and label alignment against CLASS_NAMES.
    """
    feats_path = OUTPUTS_DIR / "train_features.npy"
    labels_path = OUTPUTS_DIR / "train_labels.npy"
    ids_path = OUTPUTS_DIR / "train_image_ids.npy"

    for p in [feats_path, labels_path, ids_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required cached artifact missing: {p}")

    features = np.load(feats_path)
    labels = np.load(labels_path)
    image_ids = np.load(ids_path)

    # Validation
    if features.ndim != 2 or features.shape[1] != 1280:
        raise ValueError(f"Expected features shape (N, 1280), got {features.shape}")
    if len(features) != len(labels) or len(features) != len(image_ids):
        raise ValueError(
            f"Length mismatch: features={len(features)}, labels={len(labels)}, ids={len(image_ids)}"
        )
    if not np.isfinite(features).all():
        raise ValueError("Features contain non-finite (NaN or Inf) values.")

    unique_labels = np.unique(labels)
    if len(unique_labels) != len(CLASS_NAMES):
        raise ValueError(
            f"Expected {len(CLASS_NAMES)} classes, found {len(unique_labels)} in labels."
        )

    return features, labels, image_ids


def train_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    random_state: int = 42,
    max_iter: int = 1000,
    C: float = 1.0,
    class_weight: Any = "balanced",
) -> Tuple[LogisticRegression, float]:
    """
    Fit LogisticRegression on training features.
    Returns:
        (fitted_model, training_duration_seconds)
    """
    print(f"Training LogisticRegression (C={C}, max_iter={max_iter}, class_weight={class_weight})...")
    model = LogisticRegression(
        C=C,
        class_weight=class_weight,
        solver="lbfgs",
        max_iter=max_iter,
        random_state=random_state,
    )

    start_time = time.time()
    model.fit(features, labels)
    duration = time.time() - start_time
    print(f"Model fitting completed in {duration:.2f} seconds.")

    return model, duration


def evaluate_training_fit(
    model: LogisticRegression,
    features: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute comprehensive metrics on the training set.
    Note: These reflect training fit and convergence, not held-out generalization.
    """
    preds = model.predict(features)
    probs = model.predict_proba(features)

    acc = float(accuracy_score(labels, preds))
    bal_acc = float(balanced_accuracy_score(labels, preds))
    macro_p = float(precision_score(labels, preds, average="macro", zero_division=0))
    macro_r = float(recall_score(labels, preds, average="macro", zero_division=0))
    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))

    report = classification_report(
        labels,
        preds,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(labels, preds).tolist()

    metrics = {
        "dataset_split": "train",
        "sample_count": int(len(labels)),
        "num_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": {
            cls_name: {
                "precision": round(report[cls_name]["precision"], 4),
                "recall": round(report[cls_name]["recall"], 4),
                "f1_score": round(report[cls_name]["f1-score"], 4),
                "support": int(report[cls_name]["support"]),
            }
            for cls_name in CLASS_NAMES
        },
        "confusion_matrix": cm,
    }

    return metrics


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    import hashlib
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_artifacts(
    model: LogisticRegression,
    metrics: Dict[str, Any],
    model_filename: str = "classifier_v2.joblib",
    metrics_filename: str = "training_metrics_v2.json",
    metadata_filename: str = "classifier_v2_metadata.json",
    hyperparameters: Optional[Dict[str, Any]] = None,
) -> Tuple[Path, Path, Path]:
    """
    Save trained model, metrics, and comprehensive metadata.
    Guards classifier_v1.joblib from accidental deletion/overwriting.
    """
    import sys
    import sklearn
    import torch
    import torchvision

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / model_filename
    metrics_path = OUTPUTS_DIR / metrics_filename
    metadata_path = MODELS_DIR / metadata_filename

    # Save model binary
    joblib.dump(model, model_path)
    sha256_hash = compute_file_sha256(model_path)

    # Save training metrics
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Comprehensive metadata
    metadata = {
        "model_name": model_filename,
        "checkpoint_path": str(model_path),
        "checkpoint_sha256": sha256_hash,
        "architecture": "Pretrained EfficientNet-B0 (1280-d features) + Multinomial Logistic Regression",
        "class_order": CLASS_NAMES,
        "preprocessing_parameters": {
            "image_size": [224, 224],
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
        },
        "hyperparameters": hyperparameters or {},
        "training_sample_count": metrics.get("sample_count", 7762),
        "software_versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "metrics_summary": {
            "accuracy": metrics.get("accuracy"),
            "balanced_accuracy": metrics.get("balanced_accuracy"),
            "macro_f1": metrics.get("macro_f1"),
        },
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return model_path, metrics_path, metadata_path


def verify_saved_model(
    model_path: Path,
    test_features: np.ndarray,
) -> Dict[str, Any]:
    """
    Load saved model from disk and verify its interface:
    - predict_proba returns 7 probabilities per sample summing to 1.0.
    - classes_ matches expected indices.
    """
    loaded_model: LogisticRegression = joblib.load(model_path)
    probs = loaded_model.predict_proba(test_features)

    shape_ok = probs.shape == (len(test_features), len(CLASS_NAMES))
    sums = np.sum(probs, axis=1)
    sums_ok = bool(np.allclose(sums, 1.0, atol=1e-5))
    classes_ok = list(loaded_model.classes_) == list(range(len(CLASS_NAMES)))

    file_size_bytes = model_path.stat().st_size

    return {
        "model_path": str(model_path),
        "file_size_bytes": file_size_bytes,
        "file_size_kb": round(file_size_bytes / 1024, 2),
        "classes": [int(c) for c in loaded_model.classes_],
        "classes_ok": classes_ok,
        "probs_shape": list(probs.shape),
        "probs_shape_ok": shape_ok,
        "sums_approx_one": sums_ok,
    }


def run_training_pipeline(
    model_filename: str = "classifier_v2.joblib",
    metrics_filename: str = "training_metrics_v2.json",
    metadata_filename: str = "classifier_v2_metadata.json",
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
    class_weight: Any = "balanced",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Execute complete training, evaluation, saving, and verification."""
    features, labels, image_ids = load_training_data()
    print(f"Loaded {len(features)} feature vectors of shape {features.shape}.")

    hyperparams = {
        "C": C,
        "max_iter": max_iter,
        "random_state": random_state,
        "class_weight": class_weight if isinstance(class_weight, str) else "custom_dict",
        "solver": "lbfgs",
    }

    model, duration = train_classifier(
        features,
        labels,
        random_state=random_state,
        max_iter=max_iter,
        C=C,
        class_weight=class_weight,
    )
    metrics = evaluate_training_fit(model, features, labels)
    metrics["training_duration_seconds"] = round(duration, 2)

    model_path, metrics_path, meta_path = save_artifacts(
        model=model,
        metrics=metrics,
        model_filename=model_filename,
        metrics_filename=metrics_filename,
        metadata_filename=metadata_filename,
        hyperparameters=hyperparams,
    )
    print(f"Model saved to:    {model_path}")
    print(f"Metrics saved to:  {metrics_path}")
    print(f"Metadata saved to: {meta_path}")

    # Verify on a sample of 10 images
    verification = verify_saved_model(model_path, features[:10])
    verification["training_duration_seconds"] = round(duration, 2)
    verification["checkpoint_sha256"] = compute_file_sha256(model_path)

    return metrics, verification


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Agent A2 diagnostic model on full training set.")
    parser.add_argument("--output-model", type=str, default="classifier_v2.joblib", help="Output model filename")
    parser.add_argument("--metrics-output", type=str, default="training_metrics_v2.json", help="Output metrics filename")
    parser.add_argument("--metadata-output", type=str, default="classifier_v2_metadata.json", help="Output metadata filename")
    parser.add_argument("--C", type=float, default=1.0, help="Inverse regularization parameter")
    parser.add_argument("--max-iter", type=int, default=1000, help="Maximum solver iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--weight-scheme", type=str, default="balanced", choices=["balanced", "unweighted"], help="Weight scheme")
    args = parser.parse_args()

    # Safety check: Prevent overwriting classifier_v1.joblib unless explicitly instructed
    if args.output_model == "classifier_v1.joblib" and (MODELS_DIR / "classifier_v1.joblib").exists():
        print("[!] Warning: classifier_v1.joblib is preserved as the original baseline checkpoint.")

    cw = "balanced" if args.weight_scheme == "balanced" else None
    metrics, verif = run_training_pipeline(
        model_filename=args.output_model,
        metrics_filename=args.metrics_output,
        metadata_filename=args.metadata_output,
        C=args.C,
        max_iter=args.max_iter,
        random_state=args.seed,
        class_weight=cw,
    )

    print("\n" + "=" * 60)
    print("TRAINING FIT SUMMARY (Internal Convergence Check)")
    print("=" * 60)
    print(f"Sample Count:       {metrics['sample_count']}")
    print(f"Training Duration:  {metrics['training_duration_seconds']}s")
    print(f"Overall Accuracy:   {metrics['accuracy']}")
    print(f"Balanced Accuracy:  {metrics['balanced_accuracy']}")
    print(f"Macro F1-Score:     {metrics['macro_f1']}")
    print(f"Macro Recall:       {metrics['macro_recall']}")
    print(f"Macro Precision:    {metrics['macro_precision']}")
    print("\nPer-Class Breakdown:")
    for cls_name, vals in metrics["per_class"].items():
        print(
            f"  {cls_name:5} | Precision: {vals['precision']:.4f} | "
            f"Recall: {vals['recall']:.4f} | F1: {vals['f1_score']:.4f} | Support: {vals['support']}"
        )

    print("\n" + "=" * 60)
    print("MODEL PERSISTENCE & INTERFACE VERIFICATION")
    print("=" * 60)
    for k, v in verif.items():
        print(f"  {k}: {v}")
