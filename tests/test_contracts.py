"""
tests/test_contracts.py — Contract Verification and Synthetic Benchmark Generator.

This module enforces the data and metric contracts specified in TEAM_PLAN_v3.md and PLAN_utsavi.md:
1. Synthetic CSV generator (`generate_fake_val_results`) producing 2,000 realistic rows.
2. 18-column contract tests on validation results CSV.
3. Probability distribution and confidence invariant tests.
4. Schema contract tests on `outputs/metrics.json` (exact top-level keys, no extra keys).
5. Evaluation engine error handling and boundary condition tests.

Author: Utsavi (Senior Developer — Evaluation Track)
"""

import json
import sys
from pathlib import Path
from typing import List

# Ensure project root is in sys.path when running script directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    CLASS_NAMES,
    PROB_COLUMNS,
    REQUIRED_COLUMNS,
    VALID_QUALITY_STATUSES,
    compute_metrics,
    evaluate_predictions,
    validate_results_dataframe,
)

# Marker separating synthetic benchmark from real production models
SYNTHETIC_CSV_PATH = Path("outputs/fake_val_results.csv")
REAL_CSV_PATH = Path("outputs/val_results.csv")


# ==============================================================================
# SYNTHETIC DATA GENERATOR
# ==============================================================================

def generate_fake_val_results(
    output_path: Path = SYNTHETIC_CSV_PATH,
    n_samples: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate exactly n_samples of synthetic validation results mimicking HAM10000.

    Requirements:
    - Exactly 18 contract columns.
    - Exactly 7 diagnostic classes in canonical order.
    - ~67% true labels are 'nv' (1,340 samples out of 2,000).
    - Probabilities strictly sum to 1.0 per row.
    - Confidence matches max predicted probability.
    - Valid quality statuses (PASS, WARN, FAIL) with plausible checks and reasons.
    - Fixed random seed for deterministic reproducibility.

    Args:
        output_path: File destination for the synthetic CSV.
        n_samples: Total number of rows (default: 2,000).
        seed: Random seed (default: 42).

    Returns:
        Generated pandas DataFrame.
    """
    rng = np.random.RandomState(seed)

    # Class distribution: exactly 67.0% nv (1340), plus all other 6 classes represented
    # mel: 220 (11%), bkl: 220 (11%), bcc: 100 (5%), akiec: 60 (3%), df: 30 (1.5%), vasc: 30 (1.5%)
    class_counts = {
        "akiec": 60,
        "bcc": 100,
        "bkl": 220,
        "df": 30,
        "mel": 220,
        "nv": 1340,
        "vasc": 30,
    }
    assert sum(class_counts.values()) == n_samples, "Class counts must sum to n_samples"

    true_labels: List[str] = []
    for cls in CLASS_NAMES:
        true_labels.extend([cls] * class_counts[cls])

    # Shuffle true labels deterministically
    shuffled_indices = rng.permutation(n_samples)
    true_labels = [true_labels[i] for i in shuffled_indices]

    # Generate probabilities and predictions
    prob_matrix = np.zeros((n_samples, len(CLASS_NAMES)), dtype=np.float64)
    pred_classes: List[str] = []
    confidences: List[float] = []

    for idx, true_cls in enumerate(true_labels):
        true_idx = CLASS_NAMES.index(true_cls)

        # Base noise logits
        logits = rng.normal(loc=0.0, scale=0.4, size=len(CLASS_NAMES))

        # Model accuracy simulation (~78% correct prediction with realistic class confusion)
        is_correct = rng.rand() < 0.80
        if is_correct:
            logits[true_idx] += rng.uniform(1.8, 3.2)
        else:
            # Plausible error (confuse mel with nv/bkl, or akiec with bcc)
            if true_cls == "mel":
                confuse_idx = CLASS_NAMES.index(rng.choice(["nv", "bkl"]))
            elif true_cls == "akiec":
                confuse_idx = CLASS_NAMES.index("bcc")
            else:
                confuse_idx = rng.choice([i for i in range(len(CLASS_NAMES)) if i != true_idx])
            logits[confuse_idx] += rng.uniform(1.5, 2.5)

        # Softmax transformation
        exp_logits = np.exp(logits - np.max(logits))
        raw_probs = exp_logits / np.sum(exp_logits)

        # Round to 4 decimal places and adjust max probability to guarantee exact sum == 1.0000
        rounded = np.round(raw_probs, 4)
        diff = np.round(1.0 - np.sum(rounded), 4)
        max_idx = int(np.argmax(rounded))
        rounded[max_idx] += diff
        rounded = np.clip(rounded, 0.0, 1.0)
        # Ensure sum is strictly 1.0
        rounded[max_idx] += round(1.0 - float(np.sum(rounded)), 4)

        prob_matrix[idx] = rounded
        pred_cls = CLASS_NAMES[int(np.argmax(rounded))]
        pred_classes.append(pred_cls)
        confidences.append(float(rounded[max_idx]))

    # Quality agent simulation
    # 82% PASS, 14% WARN, 4% FAIL
    q_statuses: List[str] = []
    q_scores: List[float] = []
    q_reasons: List[str] = []
    q_blur: List[float] = []
    q_illumination: List[float] = []
    q_glare: List[float] = []
    q_obstruction: List[float] = []

    warn_reasons = ["slight_blur", "uneven_illumination", "mild_glare", "partial_hair_obstruction"]
    fail_reasons = ["severe_blur", "severe_glare", "low_illumination", "heavy_marker_obstruction"]

    for _ in range(n_samples):
        r = rng.rand()
        if r < 0.82:
            q_statuses.append("PASS")
            q_scores.append(round(rng.uniform(0.85, 0.99), 3))
            q_reasons.append("")
            q_blur.append(round(rng.uniform(0.85, 0.99), 3))
            q_illumination.append(round(rng.uniform(0.85, 0.99), 3))
            q_glare.append(round(rng.uniform(0.85, 0.99), 3))
            q_obstruction.append(round(rng.uniform(0.88, 0.99), 3))
        elif r < 0.96:
            q_statuses.append("WARN")
            q_scores.append(round(rng.uniform(0.60, 0.82), 3))
            reason = rng.choice(warn_reasons)
            q_reasons.append(reason)
            q_blur.append(round(rng.uniform(0.50, 0.70) if "blur" in reason else rng.uniform(0.80, 0.95), 3))
            q_illumination.append(round(rng.uniform(0.50, 0.70) if "illumination" in reason else rng.uniform(0.80, 0.95), 3))
            q_glare.append(round(rng.uniform(0.50, 0.70) if "glare" in reason else rng.uniform(0.80, 0.95), 3))
            q_obstruction.append(round(rng.uniform(0.50, 0.70) if "obstruction" in reason else rng.uniform(0.80, 0.95), 3))
        else:
            q_statuses.append("FAIL")
            q_scores.append(round(rng.uniform(0.20, 0.55), 3))
            reason = rng.choice(fail_reasons)
            q_reasons.append(reason)
            q_blur.append(round(rng.uniform(0.15, 0.40) if "blur" in reason else rng.uniform(0.60, 0.85), 3))
            q_illumination.append(round(rng.uniform(0.15, 0.40) if "illumination" in reason else rng.uniform(0.60, 0.85), 3))
            q_glare.append(round(rng.uniform(0.15, 0.40) if "glare" in reason else rng.uniform(0.60, 0.85), 3))
            q_obstruction.append(round(rng.uniform(0.15, 0.40) if "obstruction" in reason else rng.uniform(0.60, 0.85), 3))

    # Assemble DataFrame with synthetic image IDs
    image_ids = [f"SYNTH_ISIC_{i+1:06d}" for i in range(n_samples)]

    data = {
        "image_id": image_ids,
        "true_label": true_labels,
        "pred_class": pred_classes,
        "confidence": confidences,
    }
    for idx, cls in enumerate(CLASS_NAMES):
        data[f"prob_{cls}"] = prob_matrix[:, idx]

    data.update({
        "q_status": q_statuses,
        "q_score": q_scores,
        "q_reasons": q_reasons,
        "q_blur": q_blur,
        "q_illumination": q_illumination,
        "q_glare": q_glare,
        "q_obstruction": q_obstruction,
    })

    df = pd.DataFrame(data)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


# ==============================================================================
# PYTEST FIXTURES
# ==============================================================================

@pytest.fixture(scope="session")
def fake_df() -> pd.DataFrame:
    """Fixture providing the synthetic validation DataFrame, generating if missing."""
    if not SYNTHETIC_CSV_PATH.exists():
        return generate_fake_val_results(SYNTHETIC_CSV_PATH, n_samples=2000, seed=42)
    return pd.read_csv(SYNTHETIC_CSV_PATH)


# ==============================================================================
# CONTRACT TESTS: CSV DATA STRUCTURE
# ==============================================================================

def test_synthetic_csv_file_exists():
    """Verify synthetic validation CSV file exists on disk."""
    if not SYNTHETIC_CSV_PATH.exists():
        generate_fake_val_results(SYNTHETIC_CSV_PATH)
    assert SYNTHETIC_CSV_PATH.exists(), f"Missing synthetic test file: {SYNTHETIC_CSV_PATH}"


def test_synthetic_csv_exact_18_columns(fake_df: pd.DataFrame):
    """Verify CSV contains exactly the 18 required contract columns in order."""
    assert list(fake_df.columns) == REQUIRED_COLUMNS, (
        f"Columns mismatch!\nExpected: {REQUIRED_COLUMNS}\nGot: {list(fake_df.columns)}"
    )


def test_synthetic_csv_exactly_2000_rows(fake_df: pd.DataFrame):
    """Verify synthetic dataset contains exactly 2,000 rows as specified in TEAM_PLAN_v3.md."""
    assert len(fake_df) == 2000, f"Expected exactly 2,000 rows, found {len(fake_df)}"


def test_synthetic_csv_all_seven_classes_present_and_nv_ratio(fake_df: pd.DataFrame):
    """Verify all 7 classes are present and 'nv' represents approximately 67% of samples."""
    true_classes = set(fake_df["true_label"].unique())
    pred_classes = set(fake_df["pred_class"].unique())

    # All 7 classes must exist in ground truth
    assert true_classes == set(CLASS_NAMES), f"Mismatch in true labels: {true_classes ^ set(CLASS_NAMES)}"
    assert pred_classes.issubset(set(CLASS_NAMES)), f"Invalid predicted classes found: {pred_classes - set(CLASS_NAMES)}"

    # Check realistic ~67% nv class imbalance
    nv_count = (fake_df["true_label"] == "nv").sum()
    nv_ratio = nv_count / len(fake_df)
    assert 0.65 <= nv_ratio <= 0.70, f"nv ratio ({nv_ratio:.2%}) is outside expected ~67% range [65%, 70%]"


def test_synthetic_csv_probabilities_valid_range_and_sums(fake_df: pd.DataFrame):
    """Verify probabilities are strictly in [0.0, 1.0] and sum to 1.0 within 1e-3."""
    for col in PROB_COLUMNS:
        assert pd.api.types.is_numeric_dtype(fake_df[col]), f"Column {col} is not numeric"
        assert (fake_df[col] >= 0.0).all(), f"Negative probabilities detected in {col}"
        assert (fake_df[col] <= 1.0).all(), f"Probabilities > 1.0 detected in {col}"

    row_sums = fake_df[PROB_COLUMNS].sum(axis=1)
    np.testing.assert_allclose(
        row_sums.values,
        1.0,
        atol=1e-3,
        err_msg="Probability rows do not sum to 1.0 within tolerance",
    )


def test_synthetic_csv_confidence_matches_max_probability(fake_df: pd.DataFrame):
    """Verify confidence equals the maximum predicted probability in each row."""
    max_probs = fake_df[PROB_COLUMNS].max(axis=1)
    np.testing.assert_allclose(
        fake_df["confidence"].values,
        max_probs.values,
        atol=1e-3,
        err_msg="Confidence does not match max class probability",
    )

    # Verify pred_class corresponds to argmax of probability columns
    computed_pred_indices = np.argmax(fake_df[PROB_COLUMNS].values, axis=1)
    computed_pred_classes = [CLASS_NAMES[i] for i in computed_pred_indices]
    assert (fake_df["pred_class"].values == computed_pred_classes).all(), (
        "pred_class does not match the class corresponding to highest probability"
    )


def test_synthetic_csv_quality_contract(fake_df: pd.DataFrame):
    """Verify quality assessment columns follow the contract."""
    assert set(fake_df["q_status"].unique()).issubset(VALID_QUALITY_STATUSES)
    assert (fake_df["q_score"] >= 0.0).all() and (fake_df["q_score"] <= 1.0).all()

    for col in ["q_blur", "q_illumination", "q_glare", "q_obstruction"]:
        assert pd.api.types.is_numeric_dtype(fake_df[col])
        assert (fake_df[col] >= 0.0).all() and (fake_df[col] <= 1.0).all()


# ==============================================================================
# CONTRACT TESTS: EVALUATION ENGINE & METRICS SCHEMA
# ==============================================================================

def test_evaluate_engine_full_run(tmp_path: Path, fake_df: pd.DataFrame):
    """Run evaluate_predictions on synthetic data and verify outputs."""
    metrics_path = tmp_path / "metrics.json"
    matrix_path = tmp_path / "confusion_matrix.png"

    metrics = evaluate_predictions(
        input_path=SYNTHETIC_CSV_PATH,
        output_metrics_path=metrics_path,
        output_matrix_path=matrix_path,
    )

    # Check files were created
    assert metrics_path.exists(), "metrics.json was not created"
    assert matrix_path.exists(), "confusion_matrix.png was not created"
    assert matrix_path.stat().st_size > 1000, "confusion_matrix.png is too small or empty"

    # EXACT Contract: Top-level keys must match exactly
    expected_top_keys = {"accuracy", "balanced_accuracy", "macro_f1", "per_class", "quality_summary"}
    assert set(metrics.keys()) == expected_top_keys, (
        f"metrics.json key mismatch!\nExpected: {expected_top_keys}\nGot: {set(metrics.keys())}"
    )

    # Numerical validity
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["balanced_accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0

    # per_class contract
    assert list(metrics["per_class"].keys()) == CLASS_NAMES, "per_class must contain all 7 classes in order"
    for cls in CLASS_NAMES:
        cls_data = metrics["per_class"][cls]
        assert set(cls_data.keys()) == {"precision", "recall", "f1", "support"}
        assert 0.0 <= cls_data["precision"] <= 1.0
        assert 0.0 <= cls_data["recall"] <= 1.0
        assert 0.0 <= cls_data["f1"] <= 1.0
        assert cls_data["support"] >= 0

    # quality_summary contract
    qs = metrics["quality_summary"]
    assert set(qs.keys()) == {"PASS", "WARN", "FAIL", "top_reasons"}
    assert isinstance(qs["PASS"], int)
    assert isinstance(qs["WARN"], int)
    assert isinstance(qs["FAIL"], int)
    assert isinstance(qs["top_reasons"], list)
    assert qs["PASS"] + qs["WARN"] + qs["FAIL"] == len(fake_df)


def test_evaluate_rejects_missing_column(tmp_path: Path, fake_df: pd.DataFrame):
    """Verify validator catches and raises ValueError when contract columns are missing."""
    corrupt_df = fake_df.drop(columns=["pred_class"])
    corrupt_csv = tmp_path / "corrupt_missing_col.csv"
    corrupt_df.to_csv(corrupt_csv, index=False)

    with pytest.raises(ValueError, match="missing required column"):
        validate_results_dataframe(corrupt_df, source_name="corrupt_missing_col.csv")


def test_evaluate_rejects_extra_csv_column(fake_df: pd.DataFrame):
    """Verify validator rejects unexpected extra columns."""
    corrupt_df = fake_df.copy()
    corrupt_df["unexpected_extra_col"] = "test"
    with pytest.raises(ValueError, match="unexpected extra column"):
        validate_results_dataframe(corrupt_df, source_name="corrupt_extra_col.csv")


def test_evaluate_rejects_column_order_mismatch(fake_df: pd.DataFrame):
    """Verify validator rejects columns that do not follow the exact contract order."""
    reversed_cols = list(reversed(REQUIRED_COLUMNS))
    corrupt_df = fake_df[reversed_cols].copy()
    with pytest.raises(ValueError, match="column order mismatch"):
        validate_results_dataframe(corrupt_df, source_name="corrupt_order.csv")


def test_evaluate_rejects_missing_true_label(fake_df: pd.DataFrame):
    """Verify validator rejects missing or blank true_label."""
    corrupt_nan = fake_df.copy()
    corrupt_nan.loc[0, "true_label"] = np.nan
    with pytest.raises(ValueError, match="column 'true_label' contains .* missing"):
        validate_results_dataframe(corrupt_nan, source_name="corrupt_nan_true_label.csv")

    corrupt_blank = fake_df.copy()
    corrupt_blank.loc[0, "true_label"] = "   "
    with pytest.raises(ValueError, match="column 'true_label' contains .* blank/empty"):
        validate_results_dataframe(corrupt_blank, source_name="corrupt_blank_true_label.csv")


def test_evaluate_rejects_missing_q_status(fake_df: pd.DataFrame):
    """Verify validator rejects missing or blank q_status."""
    corrupt_nan = fake_df.copy()
    corrupt_nan.loc[0, "q_status"] = np.nan
    with pytest.raises(ValueError, match="column 'q_status' contains .* missing"):
        validate_results_dataframe(corrupt_nan, source_name="corrupt_nan_q_status.csv")

    corrupt_blank = fake_df.copy()
    corrupt_blank.loc[0, "q_status"] = ""
    with pytest.raises(ValueError, match="column 'q_status' contains .* blank/empty"):
        validate_results_dataframe(corrupt_blank, source_name="corrupt_blank_q_status.csv")


def test_evaluate_rejects_confidence_outside_range(fake_df: pd.DataFrame):
    """Verify validator rejects confidence values outside [0.0, 1.0]."""
    corrupt_high = fake_df.copy()
    corrupt_high.loc[0, "confidence"] = 1.25
    with pytest.raises(ValueError, match="values in column 'confidence' must be in"):
        validate_results_dataframe(corrupt_high, source_name="corrupt_high_confidence.csv")

    corrupt_low = fake_df.copy()
    corrupt_low.loc[0, "confidence"] = -0.05
    with pytest.raises(ValueError, match="values in column 'confidence' must be in"):
        validate_results_dataframe(corrupt_low, source_name="corrupt_low_confidence.csv")


def test_evaluate_rejects_q_score_outside_range(fake_df: pd.DataFrame):
    """Verify validator rejects q_score values outside [0.0, 1.0]."""
    corrupt_high = fake_df.copy()
    corrupt_high.loc[0, "q_score"] = 1.05
    with pytest.raises(ValueError, match="values in column 'q_score' must be in"):
        validate_results_dataframe(corrupt_high, source_name="corrupt_high_q_score.csv")

    corrupt_low = fake_df.copy()
    corrupt_low.loc[0, "q_score"] = -0.1
    with pytest.raises(ValueError, match="values in column 'q_score' must be in"):
        validate_results_dataframe(corrupt_low, source_name="corrupt_low_q_score.csv")


def test_evaluate_rejects_nan_probability(fake_df: pd.DataFrame):
    """Verify validator rejects NaN probability values."""
    corrupt_df = fake_df.copy()
    corrupt_df.loc[0, "prob_nv"] = np.nan
    with pytest.raises(ValueError, match="column 'prob_nv' contains .* missing"):
        validate_results_dataframe(corrupt_df, source_name="corrupt_nan_prob.csv")


def test_evaluate_rejects_pred_class_argmax_inconsistency(fake_df: pd.DataFrame):
    """Verify validator rejects pred_class inconsistent with probability argmax."""
    corrupt_df = fake_df.copy()
    nv_idx = corrupt_df.index[corrupt_df["pred_class"] == "nv"][0]
    corrupt_df.loc[nv_idx, "pred_class"] = "mel"
    with pytest.raises(ValueError, match="pred_class inconsistent with probability argmax"):
        validate_results_dataframe(corrupt_df, source_name="corrupt_pred_mismatch.csv")


def test_evaluate_allows_empty_q_reasons(fake_df: pd.DataFrame):
    """Verify validator allows empty strings and NaN in q_reasons for PASS images."""
    valid_df = fake_df.copy()
    valid_df.loc[0, "q_reasons"] = ""
    valid_df.loc[1, "q_reasons"] = np.nan
    validate_results_dataframe(valid_df, source_name="valid_empty_reasons.csv")


def test_evaluate_rejects_invalid_class_labels(tmp_path: Path, fake_df: pd.DataFrame):
    """Verify validator catches invalid class names."""
    corrupt_df = fake_df.copy()
    corrupt_df.loc[0, "true_label"] = "invalid_lesion_type"

    with pytest.raises(ValueError, match="invalid true_label"):
        validate_results_dataframe(corrupt_df, source_name="corrupt_invalid_label.csv")


def test_evaluate_rejects_unnormalized_probabilities(tmp_path: Path, fake_df: pd.DataFrame):
    """Verify validator catches probability rows that do not sum to 1.0."""
    corrupt_df = fake_df.copy()
    corrupt_df.loc[0, "prob_nv"] = 0.99  # breaks probability sum

    with pytest.raises(ValueError, match="probability sums != 1.0"):
        validate_results_dataframe(corrupt_df, source_name="corrupt_bad_sum.csv")


# ==============================================================================
# MAIN STANDALONE EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print("[*] Generating synthetic validation dataset (outputs/fake_val_results.csv)...")
    df = generate_fake_val_results()
    print(f"[+] Successfully created {len(df):,} synthetic rows at {SYNTHETIC_CSV_PATH}")
    print(f"    True label counts: {dict(df['true_label'].value_counts())}")
    print(f"    Quality statuses: {dict(df['q_status'].value_counts())}")
