"""
src/evaluate.py — Evaluation Engine for Agentic DermaGuard.

This module evaluates validation predictions against ground truth labels for the
seven-class skin lesion diagnosis benchmark (HAM10000 / ISIC standard).

Key Responsibilities:
1. Validate incoming CSV schema against strict 18-column contract.
2. Compute primary evaluation metrics:
   - Overall Accuracy
   - Balanced Accuracy (macro-recall, accounting for severe class imbalance)
   - Macro F1-score
   - Per-class precision, recall, F1-score, and support
3. Compute quality assessment summary (PASS/WARN/FAIL distribution and top reasons).
4. Generate labeled Confusion Matrix visualization (`outputs/confusion_matrix.png`).
5. Export contract-compliant metrics payload (`outputs/metrics.json`).
6. Display naive majority-class baseline and quality-stratified performance in console.

Author: Utsavi (Senior Developer — Evaluation Track)
"""

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# Ensure matplotlib uses a safe cache directory in restricted/sandboxed environments
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib_cache"))
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt

# ==============================================================================
# AUTHORITATIVE CONTRACT DEFINITIONS (TEAM_PLAN_v3.md / PLAN_utsavi.md)
# ==============================================================================

CLASS_NAMES: List[str] = [
    "akiec",  # 0: Actinic keratoses / intraepithelial carcinoma
    "bcc",    # 1: Basal cell carcinoma
    "bkl",    # 2: Benign keratosis-like lesions
    "df",     # 3: Dermatofibroma
    "mel",    # 4: Melanoma
    "nv",     # 5: Melanocytic nevi
    "vasc",   # 6: Vascular lesions
]

PROB_COLUMNS: List[str] = [f"prob_{c}" for c in CLASS_NAMES]

REQUIRED_COLUMNS: List[str] = [
    "image_id",
    "true_label",
    "pred_class",
    "confidence",
    *PROB_COLUMNS,
    "q_status",
    "q_score",
    "q_reasons",
    "q_blur",
    "q_illumination",
    "q_glare",
    "q_obstruction",
]

VALID_QUALITY_STATUSES = {"PASS", "WARN", "FAIL"}

NON_NULLABLE_COLUMNS: List[str] = [
    "image_id",
    "true_label",
    "pred_class",
    "confidence",
    *PROB_COLUMNS,
    "q_status",
    "q_score",
    "q_blur",
    "q_illumination",
    "q_glare",
    "q_obstruction",
]

NUMERIC_COLUMNS: List[str] = [
    "confidence",
    *PROB_COLUMNS,
    "q_score",
    "q_blur",
    "q_illumination",
    "q_glare",
    "q_obstruction",
]


# ==============================================================================
# INPUT VALIDATION
# ==============================================================================

def validate_results_dataframe(df: pd.DataFrame, source_name: str = "input CSV") -> None:
    """
    Validate that the results DataFrame strictly adheres to the 18-column contract.

    Enforces:
    1. Exact 18 columns in the exact order (rejects missing, unexpected extra, or reordered columns).
    2. Non-empty DataFrame (rejects 0-row datasets).
    3. Rejects missing (NaN/None) or blank/whitespace values in all non-nullable columns.
       (q_reasons is allowed to be empty or NaN for PASS images).
    4. Validates that numeric columns are numeric, finite (no NaN, no Inf), and in [0.0, 1.0].
    5. Validates that true_label and pred_class belong to the 7 canonical diagnostic classes.
    6. Validates probability normalization: row sums must equal 1.0 within 1e-3.
    7. Validates confidence equals the maximum class probability.
    8. Validates prediction consistency: pred_class must match argmax of probabilities according to CLASS_NAMES.
    9. Validates quality status values belong to {"PASS", "WARN", "FAIL"}.

    Args:
        df: DataFrame loaded from results CSV.
        source_name: Human-readable name/path of input for error messages.

    Raises:
        ValueError: If any contract rule or data integrity constraint is violated.
    """
    if len(df) == 0:
        raise ValueError(f"Contract violation in {source_name}: file contains 0 rows.")

    # 1. Exact 18 columns, no missing, no unexpected extra, exact order preserved
    actual_cols = list(df.columns)
    if actual_cols != REQUIRED_COLUMNS:
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in actual_cols]
        extra_cols = [c for c in actual_cols if c not in REQUIRED_COLUMNS]
        if missing_cols and extra_cols:
            raise ValueError(
                f"Contract violation in {source_name}: missing required column(s) {missing_cols} "
                f"and found unexpected extra column(s) {extra_cols}."
            )
        if missing_cols:
            raise ValueError(
                f"Contract violation in {source_name}: missing required column(s): {missing_cols}"
            )
        if extra_cols:
            raise ValueError(
                f"Contract violation in {source_name}: unexpected extra column(s) found: {extra_cols}"
            )
        # Same columns, but incorrect order
        raise ValueError(
            f"Contract violation in {source_name}: column order mismatch.\n"
            f"Expected: {REQUIRED_COLUMNS}\n"
            f"Found:    {actual_cols}"
        )

    # 2. Rejection of missing / blank values in non-nullable columns (allows empty q_reasons)
    for col in NON_NULLABLE_COLUMNS:
        if df[col].isna().any():
            nan_count = int(df[col].isna().sum())
            first_nan_idx = df.index[df[col].isna()][0]
            raise ValueError(
                f"Contract violation in {source_name}: column '{col}' contains {nan_count} missing (NaN/None) value(s), first at row {first_nan_idx}."
            )

        # Check for blank / empty strings
        if df[col].dtype == object or (len(df) > 0 and isinstance(df[col].iloc[0], str)):
            str_series = df[col].astype(str).str.strip()
            blank_mask = str_series.eq("") | str_series.str.lower().isin(["nan", "none", "null"])
            if blank_mask.any():
                blank_count = int(blank_mask.sum())
                first_blank_idx = df.index[blank_mask][0]
                raise ValueError(
                    f"Contract violation in {source_name}: column '{col}' contains {blank_count} blank/empty value(s), first at row {first_blank_idx}."
                )

    # 3. Numeric validation: must be numeric, finite, and in [0.0, 1.0]
    for col in NUMERIC_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Contract violation: column '{col}' must be numeric.")

        if np.isinf(df[col]).any():
            first_inf_idx = df.index[np.isinf(df[col])][0]
            raise ValueError(
                f"Contract violation: column '{col}' contains infinite values at row {first_inf_idx}."
            )

        out_of_bounds = (df[col] < -1e-5) | (df[col] > 1.0 + 1e-5)
        if out_of_bounds.any():
            first_bad_idx = df.index[out_of_bounds][0]
            bad_val = df.loc[first_bad_idx, col]
            raise ValueError(
                f"Contract violation: values in column '{col}' must be in [0.0, 1.0]. "
                f"Found {bad_val} at row {first_bad_idx}."
            )

    # 4. Class label validity: true_label and pred_class must be among CLASS_NAMES
    invalid_true = set(df["true_label"].unique()) - set(CLASS_NAMES)
    if invalid_true:
        raise ValueError(
            f"Contract violation in {source_name}: invalid true_label values found: {sorted(list(invalid_true))}. "
            f"Must be one of {CLASS_NAMES}."
        )

    invalid_pred = set(df["pred_class"].unique()) - set(CLASS_NAMES)
    if invalid_pred:
        raise ValueError(
            f"Contract violation in {source_name}: invalid pred_class values found: {sorted(list(invalid_pred))}. "
            f"Must be one of {CLASS_NAMES}."
        )

    # 5. Probability sum-to-one validation
    prob_sums = df[PROB_COLUMNS].sum(axis=1)
    invalid_sums = ~np.isclose(prob_sums, 1.0, atol=1e-3)
    if invalid_sums.any():
        num_invalid = int(invalid_sums.sum())
        first_bad_idx = df.index[invalid_sums][0]
        bad_sum = prob_sums.loc[first_bad_idx]
        raise ValueError(
            f"Contract violation: {num_invalid} rows have probability sums != 1.0 (e.g. row {first_bad_idx} sum = {bad_sum:.4f})."
        )

    # 6. Confidence column matches max predicted probability
    max_probs = df[PROB_COLUMNS].max(axis=1)
    confidence_diff = (df["confidence"] - max_probs).abs()
    if (confidence_diff > 1e-3).any():
        bad_conf_idx = df.index[confidence_diff > 1e-3][0]
        raise ValueError(
            f"Contract violation: confidence does not match max probability at row {bad_conf_idx} "
            f"(confidence={df.loc[bad_conf_idx, 'confidence']}, max_prob={max_probs.loc[bad_conf_idx]})."
        )

    # 7. Prediction consistency: pred_class must match argmax of probabilities using CLASS_NAMES order
    prob_values = df[PROB_COLUMNS].to_numpy()
    argmax_indices = np.argmax(prob_values, axis=1)
    expected_pred_classes = [CLASS_NAMES[i] for i in argmax_indices]
    mismatch_mask = df["pred_class"].to_numpy() != np.array(expected_pred_classes)
    if mismatch_mask.any():
        bad_pred_idx = df.index[mismatch_mask][0]
        actual_pred = df.loc[bad_pred_idx, "pred_class"]
        expected_pred = expected_pred_classes[bad_pred_idx]
        raise ValueError(
            f"Contract violation: pred_class inconsistent with probability argmax at row {bad_pred_idx}: "
            f"pred_class='{actual_pred}', expected='{expected_pred}' "
            f"(probabilities: {dict(zip(CLASS_NAMES, prob_values[bad_pred_idx]))})."
        )

    # 8. Quality statuses validity
    invalid_q = set(df["q_status"].unique()) - VALID_QUALITY_STATUSES
    if invalid_q:
        raise ValueError(
            f"Contract violation: invalid q_status values found: {sorted(list(invalid_q))}. "
            f"Must be subset of {VALID_QUALITY_STATUSES}."
        )


# ==============================================================================
# METRICS COMPUTATION
# ==============================================================================

def compute_quality_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Summarize image quality statuses and aggregate frequent failure/warning reasons.

    Args:
        df: Validated results DataFrame.

    Returns:
        Dict conforming to metrics.json quality_summary contract:
        {"PASS": int, "WARN": int, "FAIL": int, "top_reasons": list[str]}
    """
    counts = df["q_status"].value_counts().to_dict()
    pass_count = int(counts.get("PASS", 0))
    warn_count = int(counts.get("WARN", 0))
    fail_count = int(counts.get("FAIL", 0))

    # Parse reason strings (semicolon or comma delimited)
    all_reasons: List[str] = []
    for raw in df["q_reasons"].dropna():
        raw_str = str(raw).strip()
        if not raw_str or raw_str.lower() in {"none", "nan", "pass", "ok"}:
            continue
        # Split on semicolon or comma
        parts = [p.strip() for p in raw_str.replace(";", ",").split(",") if p.strip()]
        all_reasons.extend(parts)

    reason_counts = Counter(all_reasons)
    # List top reasons sorted by descending frequency
    top_reasons = [reason for reason, _ in reason_counts.most_common(5)]

    return {
        "PASS": pass_count,
        "WARN": warn_count,
        "FAIL": fail_count,
        "top_reasons": top_reasons,
    }


def compute_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute full evaluation metrics payload strictly adhering to metrics.json contract.

    Args:
        df: Validated results DataFrame.

    Returns:
        Dict matching exact metrics.json structure:
        {
          "accuracy": float,
          "balanced_accuracy": float,
          "macro_f1": float,
          "per_class": {cls: {"precision": float, "recall": float, "f1": float, "support": int}},
          "quality_summary": {"PASS": int, "WARN": int, "FAIL": int, "top_reasons": list[str]}
        }
    """
    y_true = df["true_label"].astype(str).tolist()
    y_pred = df["pred_class"].astype(str).tolist()

    # Core high-level metrics
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=CLASS_NAMES, average="macro", zero_division=0))

    # Per-class precision, recall, F1-score, and support
    report = classification_report(
        y_true,
        y_pred,
        labels=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    per_class: Dict[str, Dict[str, Any]] = {}
    for cls in CLASS_NAMES:
        cls_data = report.get(cls, {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0})
        per_class[cls] = {
            "precision": round(float(cls_data["precision"]), 4),
            "recall": round(float(cls_data["recall"]), 4),
            "f1": round(float(cls_data["f1-score"]), 4),
            "support": int(cls_data["support"]),
        }

    quality_summary = compute_quality_summary(df)

    # EXACT CONTRACT: Do NOT add any extra top-level keys!
    return {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "quality_summary": quality_summary,
    }


# ==============================================================================
# VISUALIZATION & REPORTING
# ==============================================================================

def save_confusion_matrix_plot(
    df: pd.DataFrame,
    output_path: Path,
    title_suffix: str = "",
) -> None:
    """
    Generate and save a labeled confusion matrix PNG.

    Args:
        df: Validated results DataFrame.
        output_path: File destination for the PNG.
        title_suffix: Optional label (e.g. "[SYNTHETIC DATA]").
    """
    y_true = df["true_label"].astype(str).tolist()
    y_pred = df["pred_class"].astype(str).tolist()

    cm = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.85)

    # Annotate matrix values in cells
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            val = cm[i, j]
            # Use white text for dark cells, dark text for light cells
            thresh = cm.max() / 2.0
            color = "white" if val > thresh else "black"
            ax.text(j, i, f"{val:d}", ha="center", va="center", color=color, fontsize=10)

    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    # Configure axes
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, fontsize=10, rotation=45, ha="left")
    ax.set_yticklabels(CLASS_NAMES, fontsize=10)

    title = "Confusion Matrix — Agentic DermaGuard"
    if title_suffix:
        title += f" {title_suffix}"
    ax.set_title(title, pad=20, fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Class", labelpad=10, fontsize=11, fontweight="bold")
    ax.set_ylabel("True Ground Truth", labelpad=10, fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def print_console_summary(
    df: pd.DataFrame,
    metrics: Dict[str, Any],
    is_synthetic: bool = False,
) -> None:
    """
    Print an informative, executive-ready evaluation report to stdout.

    Args:
        df: Results DataFrame.
        metrics: Computed metrics dictionary.
        is_synthetic: True if evaluating synthetic/fake test data.
    """
    total_samples = len(df)
    most_common_true, count_most_common = Counter(df["true_label"]).most_common(1)[0]
    naive_baseline = count_most_common / total_samples

    banner = " [SYNTHETIC TEST BENCHMARK] " if is_synthetic else " [VALIDATION EVALUATION] "
    print("\n" + "=" * 76)
    print(f"{banner:^76}")
    print("=" * 76)
    print(f"Total Validation Samples Evaluated: {total_samples:,}")
    print(f"Primary Benchmark Metrics:")
    print(f"  - Overall Accuracy:        {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"  - Balanced Accuracy:       {metrics['balanced_accuracy']:.4f}  ({metrics['balanced_accuracy']*100:.2f}%)")
    print(f"  - Macro F1-Score:          {metrics['macro_f1']:.4f}")
    print(f"  - Naive Majority Baseline: {naive_baseline:.4f}  (predicting '{most_common_true}' always)")

    if abs(metrics["accuracy"] - naive_baseline) < 0.05 and metrics["balanced_accuracy"] < 0.30:
        print("    [!] WARNING: Model accuracy approximates the majority baseline; evaluate balanced accuracy closely.")

    print("\n" + "-" * 76)
    print("PER-CLASS CLASSIFICATION REPORT (7 HAM10000 DIAGNOSTIC CLASSES)")
    print("-" * 76)
    print(f"{'Class':<8} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Support':>12}")
    print("-" * 76)
    for cls in CLASS_NAMES:
        c = metrics["per_class"][cls]
        print(f"{cls:<8} {c['precision']:>12.4f} {c['recall']:>12.4f} {c['f1']:>12.4f} {c['support']:>12d}")
    print("-" * 76)

    # Highlight Melanoma recall
    mel_recall = metrics["per_class"]["mel"]["recall"]
    print(f"  Critical Clinical Metric: Melanoma (mel) Recall = {mel_recall:.4f} ({mel_recall*100:.2f}%)")

    # Quality breakdown
    qs = metrics["quality_summary"]
    print("\n" + "-" * 76)
    print("IMAGE QUALITY AGENT SUMMARY")
    print("-" * 76)
    print(f"  PASS: {qs['PASS']:>5d} ({qs['PASS']/total_samples*100:>5.1f}%)")
    print(f"  WARN: {qs['WARN']:>5d} ({qs['WARN']/total_samples*100:>5.1f}%)")
    print(f"  FAIL: {qs['FAIL']:>5d} ({qs['FAIL']/total_samples*100:>5.1f}%)")
    if qs["top_reasons"]:
        print(f"  Top Quality Warning/Failure Reasons: {', '.join(qs['top_reasons'])}")

    # Bonus: Accuracy stratified by quality status
    print("\n" + "-" * 76)
    print("STRATIFIED ACCURACY BY IMAGE QUALITY STATUS")
    print("-" * 76)
    for status in ["PASS", "WARN", "FAIL"]:
        subset = df[df["q_status"] == status]
        if len(subset) > 0:
            sub_acc = accuracy_score(subset["true_label"], subset["pred_class"])
            print(f"  Status {status:<4} (N={len(subset):>4d}): Accuracy = {sub_acc:.4f} ({sub_acc*100:.2f}%)")
        else:
            print(f"  Status {status:<4} (N=   0): N/A")
    print("=" * 76 + "\n")


# ==============================================================================
# PIPELINE EXECUTION
# ==============================================================================

def evaluate_predictions(
    input_path: Path,
    output_metrics_path: Path = Path("outputs/metrics.json"),
    output_matrix_path: Path = Path("outputs/confusion_matrix.png"),
) -> Dict[str, Any]:
    """
    Run the end-to-end evaluation pipeline on a results CSV.

    Args:
        input_path: Path to val_results.csv or fake_val_results.csv.
        output_metrics_path: Destination path for metrics.json.
        output_matrix_path: Destination path for confusion_matrix.png.

    Returns:
        Contract-compliant metrics dictionary.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Results file not found at: '{input_path.resolve()}'.\n"
            f"If testing before model run, generate synthetic data first using:\n"
            f"  python tests/test_contracts.py"
        )

    df = pd.read_csv(input_path)
    validate_results_dataframe(df, source_name=input_path.name)

    is_synthetic = "fake" in input_path.name.lower() or "synth" in input_path.name.lower()

    # 1. Compute metrics
    metrics = compute_metrics(df)

    # 2. Save metrics.json
    output_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # 3. Generate and save confusion matrix
    save_confusion_matrix_plot(
        df=df,
        output_path=output_matrix_path,
        title_suffix="[SYNTHETIC TEST DATA]" if is_synthetic else "",
    )

    # 4. Print console summary
    print_console_summary(df, metrics, is_synthetic=is_synthetic)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agentic DermaGuard — Diagnostic Evaluation and Contract Verification Engine"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="outputs/val_results.csv",
        help="Path to results CSV (default: outputs/val_results.csv)",
    )
    parser.add_argument(
        "--output-metrics",
        type=str,
        default="outputs/metrics.json",
        help="Path to output metrics.json (default: outputs/metrics.json)",
    )
    parser.add_argument(
        "--output-confusion-matrix",
        type=str,
        default="outputs/confusion_matrix.png",
        help="Path to output confusion matrix PNG (default: outputs/confusion_matrix.png)",
    )

    args = parser.parse_args()

    try:
        evaluate_predictions(
            input_path=Path(args.input),
            output_metrics_path=Path(args.output_metrics),
            output_matrix_path=Path(args.output_confusion_matrix),
        )
    except Exception as exc:
        print(f"\n[ERROR in evaluate.py]: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
