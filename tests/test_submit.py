"""Unit and contract tests for the Hidden-Test Submission Integration (src/submit.py).

Module: tests/test_submit.py
Author: Aashi (Submission Integration Owner)
"""

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.submit import (
    SUBMISSION_COLUMNS,
    VALID_DECISIONS,
    VALID_PREDICTED_LABELS,
    normalize_decision,
    submit_to_portal,
    validate_submission_file,
    validate_submission_record,
    write_submission_csv,
)


def test_record_validation_valid():
    """Verify legitimate records are accepted and normalized."""
    rec = validate_submission_record("ISIC_0024306", "mel", "AUTO-TRIAGE")
    assert rec["image_id"] == "ISIC_0024306"
    assert rec["predicted_label"] == "mel"
    assert rec["decision"] == "AUTO-TRIAGE"

    # Variations in case and extension
    rec2 = validate_submission_record("ISIC_0024307.jpg", "NV", "refer")
    assert rec2["image_id"] == "ISIC_0024307"
    assert rec2["predicted_label"] == "nv"
    assert rec2["decision"] == "REFER"

    rec3 = validate_submission_record("ISIC_0024308", "bcc", "re_image")
    assert rec3["decision"] == "RE-IMAGE"


def test_record_validation_invalid_label():
    """Verify invalid diagnostic labels are rejected."""
    for bad_label in ["cancer", "unknown", "carcinoma", "", 123]:
        try:
            validate_submission_record("ISIC_0024306", bad_label, "REFER")
            assert False, f"Expected ValueError for invalid label '{bad_label}'"
        except ValueError:
            pass


def test_record_validation_invalid_decision():
    """Verify invalid decision strings are rejected."""
    for bad_dec in ["ACCEPT", "REJECT", "IGNORE", "DIAGNOSE", ""]:
        try:
            validate_submission_record("ISIC_0024306", "mel", bad_dec)
            assert False, f"Expected ValueError for invalid decision '{bad_dec}'"
        except ValueError:
            pass


def test_record_validation_unsafe_image_id():
    """Verify path traversal or malformed IDs in submission records are rejected."""
    for bad_id in ["../secret", "img/01", "a\\b", 9999]:
        try:
            validate_submission_record(bad_id, "mel", "REFER")
            assert False, f"Expected ValueError for unsafe id '{bad_id}'"
        except ValueError:
            pass


def test_write_and_validate_submission_csv():
    """Verify end-to-end writing and validation of a submission CSV."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_csv = Path(tmpdir) / "sub.csv"
        records = [
            {"image_id": "ISIC_0000001", "predicted_label": "nv", "decision": "AUTO-TRIAGE"},
            {"image_id": "ISIC_0000002", "predicted_label": "mel", "decision": "REFER"},
            {"image_id": "ISIC_0000003", "predicted_label": "bcc", "decision": "RE-IMAGE"},
        ]

        written_path = write_submission_csv(records, out_csv)
        assert written_path.is_file()

        # Validate back
        parsed = validate_submission_file(written_path)
        assert len(parsed) == 3
        assert parsed[0]["image_id"] == "ISIC_0000001"
        assert parsed[1]["decision"] == "REFER"
        assert parsed[2]["decision"] == "RE-IMAGE"


def test_duplicate_image_id_rejection():
    """Verify submission validator strictly rejects duplicate image IDs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_csv = Path(tmpdir) / "dup.csv"
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write("image_id,predicted_label,decision\n")
            f.write("ISIC_0000001,nv,AUTO-TRIAGE\n")
            f.write("ISIC_0000001,mel,REFER\n")  # duplicate

        try:
            validate_submission_file(out_csv)
            assert False, "Expected ValueError for duplicate image IDs"
        except ValueError as e:
            assert "duplicate" in str(e).lower()


def test_header_format_strictness():
    """Verify validator rejects wrong column order or missing columns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Wrong column order
        out_csv1 = Path(tmpdir) / "bad_order.csv"
        with open(out_csv1, "w", encoding="utf-8") as f:
            f.write("predicted_label,image_id,decision\n")
            f.write("nv,ISIC_0000001,AUTO-TRIAGE\n")

        try:
            validate_submission_file(out_csv1)
            assert False, "Expected ValueError for wrong column order"
        except ValueError as e:
            assert "header" in str(e).lower()

        # Extra column
        out_csv2 = Path(tmpdir) / "extra_col.csv"
        with open(out_csv2, "w", encoding="utf-8") as f:
            f.write("image_id,predicted_label,decision,confidence\n")
            f.write("ISIC_0000001,nv,AUTO-TRIAGE,0.95\n")

        try:
            validate_submission_file(out_csv2)
            assert False, "Expected ValueError for extra column"
        except ValueError as e:
            assert "header" in str(e).lower()


def test_submit_to_portal_safety_gate():
    """Verify safety gate blocks unconfirmed submissions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_csv = Path(tmpdir) / "sub.csv"
        write_submission_csv([
            {"image_id": "ISIC_0000001", "predicted_label": "nv", "decision": "AUTO-TRIAGE"}
        ], out_csv)

        # Without confirmation: must be refused
        res_refused = submit_to_portal(out_csv, confirm=False)
        assert res_refused["status"] == "REFUSED"
        assert res_refused["authorized"] is False

        # With confirmation but missing endpoint: must be blocked safely
        res_blocked = submit_to_portal(out_csv, confirm=True, endpoint_url=None)
        assert res_blocked["status"] == "BLOCKED"
        assert res_blocked["authorized"] is True


def run_all_tests():
    """Execute all tests sequentially for direct script execution."""
    print("=" * 65)
    print("  Running test_submit.py Unit Suite")
    print("=" * 65)

    test_record_validation_valid()
    print("  [PASS] test_record_validation_valid")

    test_record_validation_invalid_label()
    print("  [PASS] test_record_validation_invalid_label")

    test_record_validation_invalid_decision()
    print("  [PASS] test_record_validation_invalid_decision")

    test_record_validation_unsafe_image_id()
    print("  [PASS] test_record_validation_unsafe_image_id")

    test_write_and_validate_submission_csv()
    print("  [PASS] test_write_and_validate_submission_csv")

    test_duplicate_image_id_rejection()
    print("  [PASS] test_duplicate_image_id_rejection")

    test_header_format_strictness()
    print("  [PASS] test_header_format_strictness")

    test_submit_to_portal_safety_gate()
    print("  [PASS] test_submit_to_portal_safety_gate")

    print("\n" + "=" * 65)
    print("  ALL SUBMISSION INTEGRATION TESTS PASSED!")
    print("=" * 65)


if __name__ == "__main__":
    run_all_tests()
