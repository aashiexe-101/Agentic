"""Unit and contract tests for dataset loading, path security, and lesion grouping.

Module: tests/test_data.py
Author: Aashi (Data Pipeline Owner)
"""

import sys
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (
    CLASS_LABELS,
    IDX_TO_LABEL,
    LABEL_TO_IDX,
    REQUIRED_COLUMNS,
    VALID_SPLITS,
    _resolve_data_dir,
    _resolve_split,
    get_image_path,
    split_metadata_by_lesion,
    validate_image_id,
    verify_split_leakage,
)

# Optional third-party imports
try:
    import pandas as pd
except ImportError:
    pd = None


def test_constants_and_class_ordering():
    """Verify exact 7 canonical classes and bidirectional mappings."""
    expected = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    assert CLASS_LABELS == expected, f"Expected {expected}, got {CLASS_LABELS}"
    assert len(CLASS_LABELS) == 7

    for idx, label in enumerate(CLASS_LABELS):
        assert LABEL_TO_IDX[label] == idx
        assert IDX_TO_LABEL[idx] == label


def test_split_resolution_and_aliases():
    """Verify split resolution accepts 'train', 'val', and alias 'eval'."""
    assert _resolve_split("train") == "train"
    assert _resolve_split("val") == "val"
    assert _resolve_split("eval") == "val"
    assert _resolve_split("VAL") == "val"
    assert _resolve_split("EVAL") == "val"

    try:
        _resolve_split("invalid_split")
        assert False, "Expected ValueError for invalid_split"
    except ValueError:
        pass


def test_validate_image_id_security():
    """Verify image_id security validation rules."""
    # Valid IDs
    assert validate_image_id("ISIC_0024306") == "ISIC_0024306"
    assert validate_image_id("ISIC_0024306.jpg") == "ISIC_0024306"
    assert validate_image_id("ISIC_0024306.jpeg") == "ISIC_0024306"
    assert validate_image_id("ISIC_0024306.JPG") == "ISIC_0024306"
    assert validate_image_id("sample-img_99") == "sample-img_99"

    # Non-strings must raise ValueError
    for bad in [12345, 0, 3.14, float("nan"), None, [], {}, object()]:
        try:
            validate_image_id(bad)
            assert False, f"Expected ValueError for non-string {bad!r}"
        except ValueError:
            pass

    # Empty / whitespace
    for bad in ["", "   ", "\t\n"]:
        try:
            validate_image_id(bad)
            assert False, f"Expected ValueError for empty {bad!r}"
        except ValueError:
            pass

    # Path traversal / separators
    for bad in ["../secret", "a/b", "c\\d", "img:name", "null\x00byte"]:
        try:
            validate_image_id(bad)
            assert False, f"Expected ValueError for traversal/separator {bad!r}"
        except ValueError:
            pass


def test_split_by_lesion_zero_leakage():
    """Verify lesion-grouped internal splitting guarantees 0 lesion overlap."""
    if pd is None:
        return

    # Create synthetic dataset with shared lesion IDs
    records = []
    # 10 lesions, each with 1 to 4 images
    for l_idx in range(10):
        lesion_id = f"HAM_L_{l_idx:04d}"
        for img_idx in range(3):
            records.append({
                "image_id": f"IMG_{l_idx}_{img_idx}",
                "lesion_id": lesion_id,
                "dx": CLASS_LABELS[l_idx % 7],
                "age": 45.0,
                "sex": "female",
                "localization": "back",
            })

    df = pd.DataFrame(records)
    train_df, val_df = split_metadata_by_lesion(df, val_fraction=0.3, seed=42)

    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(train_df) + len(val_df) == len(df)

    # Invariant: 0 overlapping lesions
    train_lesions = set(train_df["lesion_id"])
    val_lesions = set(val_df["lesion_id"])
    overlap = train_lesions.intersection(val_lesions)
    assert len(overlap) == 0, f"Found overlapping lesions: {overlap}"

    # Verify leakage helper
    leakage = verify_split_leakage(train_df, val_df)
    assert leakage["is_leak_free"] is True
    assert leakage["lesion_overlap_count"] == 0
    assert leakage["image_overlap_count"] == 0


def test_verify_split_leakage_detector():
    """Verify leakage detector correctly flags overlapping images and lesions."""
    if pd is None:
        return

    train_data = pd.DataFrame([
        {"image_id": "IMG_01", "lesion_id": "L_01"},
        {"image_id": "IMG_02", "lesion_id": "L_02"},
    ])
    # Leaking image and lesion
    val_data = pd.DataFrame([
        {"image_id": "IMG_02", "lesion_id": "L_02"},
        {"image_id": "IMG_03", "lesion_id": "L_03"},
    ])

    report = verify_split_leakage(train_data, val_data)
    assert report["is_leak_free"] is False
    assert report["image_overlap_count"] == 1
    assert report["lesion_overlap_count"] == 1
    assert "IMG_02" in report["overlapping_images"]
    assert "L_02" in report["overlapping_lesions"]


def test_real_dataset_counts_and_zero_overlap():
    """Verify physical dataset integrity on disk if present."""
    data_dir = PROJECT_ROOT / "data"
    train_csv = data_dir / "train" / "metadata.csv"
    val_csv = data_dir / "val" / "metadata.csv"

    if not (train_csv.is_file() and val_csv.is_file()):
        return  # Skip if datasets not extracted in test environment

    if pd is None:
        return

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    # 1. Exact counts
    assert len(train_df) == 7762, f"Expected 7,762 train rows, found {len(train_df)}"
    assert len(val_df) == 1942, f"Expected 1,942 val rows, found {len(val_df)}"

    # 2. Schema check
    for col in REQUIRED_COLUMNS:
        assert col in train_df.columns
        assert col in val_df.columns

    # 3. Label check
    train_labels = set(train_df["dx"].str.strip().str.lower().unique())
    val_labels = set(val_df["dx"].str.strip().str.lower().unique())
    assert train_labels == set(CLASS_LABELS)
    assert val_labels == set(CLASS_LABELS)

    # 4. Zero duplicate image IDs
    assert len(train_df["image_id"].unique()) == 7762
    assert len(val_df["image_id"].unique()) == 1942

    # 5. Zero leakage between train and val
    leakage = verify_split_leakage(train_df, val_df)
    assert leakage["is_leak_free"] is True
    assert leakage["image_overlap_count"] == 0
    assert leakage["lesion_overlap_count"] == 0


def run_all_tests():
    """Execute all tests sequentially for direct script execution."""
    print("=" * 65)
    print("  Running test_data.py Unit & Integration Suite")
    print("=" * 65)

    test_constants_and_class_ordering()
    print("  [PASS] test_constants_and_class_ordering")

    test_split_resolution_and_aliases()
    print("  [PASS] test_split_resolution_and_aliases")

    test_validate_image_id_security()
    print("  [PASS] test_validate_image_id_security")

    test_split_by_lesion_zero_leakage()
    print("  [PASS] test_split_by_lesion_zero_leakage")

    test_verify_split_leakage_detector()
    print("  [PASS] test_verify_split_leakage_detector")

    test_real_dataset_counts_and_zero_overlap()
    print("  [PASS] test_real_dataset_counts_and_zero_overlap")

    print("\n" + "=" * 65)
    print("  ALL DATA PIPELINE TESTS PASSED!")
    print("=" * 65)


if __name__ == "__main__":
    run_all_tests()
