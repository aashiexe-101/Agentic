"""Unit and contract tests for Agent A4 (Context & Demographic Evaluation).

Module: tests/test_context_agent.py
Author: Aashi (Context Agent Owner)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.context_agent import evaluate_context


def test_empty_and_none_metadata():
    """Verify that None or empty metadata preserves missing fields without inventing data."""
    for empty_input in (None, {}, "not a dict", []):
        res = evaluate_context(empty_input)
        assert res["age"] is None
        assert res["sex"] is None
        assert res["localization"] is None
        assert set(res["missing_fields"]) == {"age", "sex", "localization"}
        assert "ALL_CONTEXT_MISSING" in res["context_flags"]
        assert len(res["notes"]) > 0


def test_full_valid_context():
    """Verify full demographic inputs are parsed cleanly."""
    meta = {
        "age": 45.0,
        "sex": "female",
        "localization": "back",
    }
    res = evaluate_context(meta)
    assert res["age"] == 45.0
    assert res["sex"] == "female"
    assert res["localization"] == "back"
    assert len(res["missing_fields"]) == 0
    assert "CONTEXT_COMPLETE" in res["context_flags"]


def test_partial_context():
    """Verify partial context preserves missing fields accurately."""
    meta = {
        "sex": "male",
        "localization": "face",
        # age missing
    }
    res = evaluate_context(meta)
    assert res["age"] is None
    assert res["sex"] == "male"
    assert res["localization"] == "face"
    assert res["missing_fields"] == ["age"]
    assert "INCOMPLETE_CONTEXT" in res["context_flags"]
    assert "HEAD_NECK_SITE" in res["context_flags"]


def test_ground_truth_isolation():
    """Verify context agent strictly quarantines and ignores ground-truth diagnosis keys."""
    meta = {
        "age": 55,
        "sex": "male",
        "localization": "lower extremity",
        "dx": "mel",
        "diagnosis": "malignant melanoma",
        "true_label": "mel",
        "label": "mel",
    }
    res = evaluate_context(meta)
    # Output must strictly adhere to the schema
    expected_keys = {"age", "sex", "localization", "missing_fields", "context_flags", "notes"}
    assert set(res.keys()) == expected_keys

    # None of the forbidden keys should appear in top-level output
    for forbidden in ("dx", "diagnosis", "true_label", "label"):
        assert forbidden not in res

    # The notes must document the quarantine
    assert any("quarantined" in note.lower() for note in res["notes"])


def test_demographic_age_ranges():
    """Verify age boundaries (pediatric, adult, geriatric, invalid)."""
    # Pediatric
    p_res = evaluate_context({"age": 12, "sex": "male", "localization": "arm"})
    assert p_res["age"] == 12.0
    assert "PEDIATRIC_PATIENT" in p_res["context_flags"]

    # Geriatric
    g_res = evaluate_context({"age": 72, "sex": "female", "localization": "chest"})
    assert g_res["age"] == 72.0
    assert "GERIATRIC_PATIENT" in g_res["context_flags"]

    # Invalid negative age
    neg_res = evaluate_context({"age": -5})
    assert neg_res["age"] is None
    assert "age" in neg_res["missing_fields"]

    # Invalid excessive age
    big_res = evaluate_context({"age": 150})
    assert big_res["age"] is None
    assert "age" in big_res["missing_fields"]


def test_sex_normalization():
    """Verify flexible biological sex normalization."""
    assert evaluate_context({"sex": "M"})["sex"] == "male"
    assert evaluate_context({"sex": "Female"})["sex"] == "female"
    assert evaluate_context({"gender": "male"})["sex"] == "male"
    assert evaluate_context({"sex": "other"})["sex"] == "other"
    assert evaluate_context({"sex": "unknown"})["sex"] is None


def test_anatomical_site_normalization():
    """Verify anatomical site normalization and observational flags."""
    assert evaluate_context({"site": "BACK"})["localization"] == "back"
    res_acral = evaluate_context({"localization": "acral"})
    assert res_acral["localization"] == "acral"
    assert "ACRAL_SITE" in res_acral["context_flags"]

    res_genital = evaluate_context({"localization": "genital"})
    assert "GENITAL_SITE" in res_genital["context_flags"]


def run_all_tests():
    """Execute all tests sequentially for direct script execution."""
    print("=" * 65)
    print("  Running test_context_agent.py Unit Suite")
    print("=" * 65)

    test_empty_and_none_metadata()
    print("  [PASS] test_empty_and_none_metadata")

    test_full_valid_context()
    print("  [PASS] test_full_valid_context")

    test_partial_context()
    print("  [PASS] test_partial_context")

    test_ground_truth_isolation()
    print("  [PASS] test_ground_truth_isolation")

    test_demographic_age_ranges()
    print("  [PASS] test_demographic_age_ranges")

    test_sex_normalization()
    print("  [PASS] test_sex_normalization")

    test_anatomical_site_normalization()
    print("  [PASS] test_anatomical_site_normalization")

    print("\n" + "=" * 65)
    print("  ALL CONTEXT AGENT TESTS PASSED!")
    print("=" * 65)


if __name__ == "__main__":
    run_all_tests()
