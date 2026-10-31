"""Unit, contract, and robustness tests for Agent A4 (Context & Demographic Evaluation).

Module: tests/test_context_agent.py
Author: Aashi (Context Agent Owner)

Authoritative References:
- docs/context_rules.md
- src/context_agent.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.context_agent import (
    FORBIDDEN_GROUND_TRUTH_KEYS,
    KNOWN_ANATOMICAL_SITES,
    evaluate_context,
)

EXPECTED_OUTPUT_KEYS = {
    "age",
    "sex",
    "localization",
    "missing_fields",
    "context_flags",
    "notes",
}

COMPLETENESS_FLAGS = {
    "ALL_CONTEXT_MISSING",
    "INCOMPLETE_CONTEXT",
    "CONTEXT_COMPLETE",
}


# ==============================================================================
# PART 1: SPECIFICATION CONTRACT TESTS (docs/context_rules.md)
# ==============================================================================

def test_empty_and_none_metadata():
    """Verify that None, empty, or non-dict metadata preserves missing fields without inventing data."""
    for empty_input in (None, {}, "not a dict", [], 42, False):
        res = evaluate_context(empty_input)
        assert set(res.keys()) == EXPECTED_OUTPUT_KEYS
        assert res["age"] is None
        assert res["sex"] is None
        assert res["localization"] is None
        assert set(res["missing_fields"]) == {"age", "sex", "localization"}
        assert "ALL_CONTEXT_MISSING" in res["context_flags"]
        assert len(res["notes"]) > 0
        assert len(set(res["context_flags"]) & COMPLETENESS_FLAGS) == 1


def test_full_valid_context():
    """Verify full valid demographic inputs are parsed cleanly and output contract is satisfied."""
    meta = {
        "age": 45.0,
        "sex": "female",
        "localization": "back",
    }
    res = evaluate_context(meta)
    assert set(res.keys()) == EXPECTED_OUTPUT_KEYS
    assert res["age"] == 45.0
    assert res["sex"] == "female"
    assert res["localization"] == "back"
    assert len(res["missing_fields"]) == 0
    assert "CONTEXT_COMPLETE" in res["context_flags"]
    assert "INCOMPLETE_CONTEXT" not in res["context_flags"]
    assert "ALL_CONTEXT_MISSING" not in res["context_flags"]
    assert len(res["notes"]) > 0


def test_partial_context():
    """Verify partial context preserves missing fields accurately with observational flags."""
    meta = {
        "sex": "male",
        "localization": "face",
        # age missing
    }
    res = evaluate_context(meta)
    assert set(res.keys()) == EXPECTED_OUTPUT_KEYS
    assert res["age"] is None
    assert res["sex"] == "male"
    assert res["localization"] == "face"
    assert res["missing_fields"] == ["age"]
    assert "INCOMPLETE_CONTEXT" in res["context_flags"]
    assert "HEAD_NECK_SITE" in res["context_flags"]
    assert "CONTEXT_COMPLETE" not in res["context_flags"]


def test_demographic_age_ranges():
    """Verify age boundaries (pediatric, adult, geriatric, invalid)."""
    # Pediatric (<18)
    p_res = evaluate_context({"age": 12, "sex": "male", "localization": "abdomen"})
    assert set(p_res.keys()) == EXPECTED_OUTPUT_KEYS
    assert p_res["age"] == 12.0
    assert "PEDIATRIC_PATIENT" in p_res["context_flags"]
    assert "GERIATRIC_PATIENT" not in p_res["context_flags"]

    # Geriatric (>=65)
    g_res = evaluate_context({"age": 72, "sex": "female", "localization": "chest"})
    assert set(g_res.keys()) == EXPECTED_OUTPUT_KEYS
    assert g_res["age"] == 72.0
    assert "GERIATRIC_PATIENT" in g_res["context_flags"]
    assert "PEDIATRIC_PATIENT" not in g_res["context_flags"]

    # Adult (18 <= age < 65)
    a_res = evaluate_context({"age": 35, "sex": "male", "localization": "trunk"})
    assert set(a_res.keys()) == EXPECTED_OUTPUT_KEYS
    assert a_res["age"] == 35.0
    assert "PEDIATRIC_PATIENT" not in a_res["context_flags"]
    assert "GERIATRIC_PATIENT" not in a_res["context_flags"]

    # Invalid negative age
    neg_res = evaluate_context({"age": -5})
    assert set(neg_res.keys()) == EXPECTED_OUTPUT_KEYS
    assert neg_res["age"] is None
    assert "age" in neg_res["missing_fields"]

    # Invalid excessive age (>125)
    big_res = evaluate_context({"age": 150})
    assert set(big_res.keys()) == EXPECTED_OUTPUT_KEYS
    assert big_res["age"] is None
    assert "age" in big_res["missing_fields"]


def test_age_boundary_exact_limits():
    """Verify exact valid boundaries: age 0 and age 125 are valid; float rounding to 1 decimal."""
    res_0 = evaluate_context({"age": 0})
    assert set(res_0.keys()) == EXPECTED_OUTPUT_KEYS
    assert res_0["age"] == 0.0
    assert "age" not in res_0["missing_fields"]
    assert "PEDIATRIC_PATIENT" in res_0["context_flags"]

    res_125 = evaluate_context({"age": 125})
    assert set(res_125.keys()) == EXPECTED_OUTPUT_KEYS
    assert res_125["age"] == 125.0
    assert "age" not in res_125["missing_fields"]
    assert "GERIATRIC_PATIENT" in res_125["context_flags"]

    # Rounding to 1 decimal place
    res_round = evaluate_context({"age": 45.678})
    assert set(res_round.keys()) == EXPECTED_OUTPUT_KEYS
    assert res_round["age"] == 45.7


def test_age_invalid_ranges_and_non_numerics():
    """Verify out-of-range, non-numeric, NaN, and infinite ages are marked invalid."""
    invalid_cases = [
        -0.1,
        -1,
        125.1,
        200,
        "forty",
        "unknown",
        "?",
        "N/A",
        "",
        float("nan"),
        float("inf"),
        float("-inf"),
        "nan",
        "inf",
        "-inf",
    ]
    for inv in invalid_cases:
        res = evaluate_context({"age": inv})
        assert set(res.keys()) == EXPECTED_OUTPUT_KEYS
        assert res["age"] is None, f"Expected age=None for input {inv!r}, got {res['age']}"
        assert "age" in res["missing_fields"], f"Expected 'age' in missing_fields for {inv!r}"


def test_sex_normalization():
    """Verify biological sex normalization for canonical, extended, and unknown values."""
    assert evaluate_context({"sex": "M"})["sex"] == "male"
    assert evaluate_context({"sex": "Female"})["sex"] == "female"
    assert evaluate_context({"gender": "male"})["sex"] == "male"
    assert evaluate_context({"sex": "other"})["sex"] == "other"
    assert evaluate_context({"sex": "non-binary"})["sex"] == "non-binary"
    assert evaluate_context({"sex": "intersex"})["sex"] == "intersex"
    res_unk = evaluate_context({"sex": "unknown"})
    assert set(res_unk.keys()) == EXPECTED_OUTPUT_KEYS
    assert res_unk["sex"] is None
    assert "sex" in res_unk["missing_fields"]


def test_sex_normalization_case_and_whitespace():
    """Verify case-insensitivity and whitespace stripping for biological sex."""
    for m in ("m", "M", "male", "Male", "MALE", " male ", "  M  \t"):
        res = evaluate_context({"sex": m})
        assert set(res.keys()) == EXPECTED_OUTPUT_KEYS
        assert res["sex"] == "male", f"Failed for {m!r}"

    for f in ("f", "F", "female", "Female", "FEMALE", " female ", " F \n"):
        res = evaluate_context({"sex": f})
        assert set(res.keys()) == EXPECTED_OUTPUT_KEYS
        assert res["sex"] == "female", f"Failed for {f!r}"


def test_sex_normalization_extended_categories():
    """Verify preservation of 'other', 'non-binary', and 'intersex' in lowercase."""
    for val, expected in [
        ("other", "other"),
        ("Other", "other"),
        ("  OTHER  ", "other"),
        ("non-binary", "non-binary"),
        ("Non-Binary", "non-binary"),
        ("intersex", "intersex"),
        ("Intersex", "intersex"),
    ]:
        res = evaluate_context({"sex": val})
        assert set(res.keys()) == EXPECTED_OUTPUT_KEYS
        assert res["sex"] == expected, f"Failed for {val!r}"
        assert "sex" not in res["missing_fields"]


def test_sex_normalization_placeholders_and_unrecognized():
    """Verify unknown, null, blank, and unrecognized sex values become missing."""
    placeholders = [None, "", "   ", "nan", "none", "null", "unknown", "?", "n/a", "not reported"]
    for ph in placeholders:
        res = evaluate_context({"sex": ph})
        assert set(res.keys()) == EXPECTED_OUTPUT_KEYS
        assert res["sex"] == None, f"Expected sex=None for placeholder {ph!r}"
        assert "sex" in res["missing_fields"]

    unrecognized = ["alien", "123", "prefer not to say"]
    for unk in unrecognized:
        res = evaluate_context({"sex": unk})
        assert set(res.keys()) == EXPECTED_OUTPUT_KEYS
        assert res["sex"] is None
        assert "sex" in res["missing_fields"]
        assert any("unconfirmed" in note.lower() for note in res["notes"])


def test_anatomical_site_normalization():
    """Verify anatomical site normalization and observational flags."""
    assert evaluate_context({"site": "BACK"})["localization"] == "back"

    res_acral = evaluate_context({"localization": "acral"})
    assert set(res_acral.keys()) == EXPECTED_OUTPUT_KEYS
    assert res_acral["localization"] == "acral"
    assert "ACRAL_SITE" in res_acral["context_flags"]

    res_genital = evaluate_context({"localization": "genital"})
    assert set(res_genital.keys()) == EXPECTED_OUTPUT_KEYS
    assert "GENITAL_SITE" in res_genital["context_flags"]

    res_head_neck = evaluate_context({"localization": "ear"})
    assert set(res_head_neck.keys()) == EXPECTED_OUTPUT_KEYS
    assert "HEAD_NECK_SITE" in res_head_neck["context_flags"]


def test_anatomical_localization_all_canonical_sites():
    """Verify all canonical anatomical sites specified in docs/context_rules.md normalize correctly."""
    canonical_sites_test = [
        site for site in KNOWN_ANATOMICAL_SITES if site != "unknown"
    ]
    for site in canonical_sites_test:
        res_exact = evaluate_context({"localization": site})
        assert set(res_exact.keys()) == EXPECTED_OUTPUT_KEYS
        assert res_exact["localization"] == site, f"Failed for site {site!r}"
        assert "localization" not in res_exact["missing_fields"]

        transformed = f"  {site.upper()}  "
        res_transform = evaluate_context({"localization": transformed})
        assert set(res_transform.keys()) == EXPECTED_OUTPUT_KEYS
        assert res_transform["localization"] == site, f"Failed for transformed site {transformed!r}"
        assert "localization" not in res_transform["missing_fields"]


def test_anatomical_localization_missing_and_placeholders():
    """Verify placeholder and missing localization values are treated as missing."""
    placeholders = [None, "", "   ", "nan", "none", "null", "not reported", "?"]
    for ph in placeholders:
        res = evaluate_context({"localization": ph})
        assert set(res.keys()) == EXPECTED_OUTPUT_KEYS
        assert res["localization"] is None, f"Expected localization=None for {ph!r}"
        assert "localization" in res["missing_fields"]


def test_alternative_metadata_keys_supported():
    """Verify alternative metadata keys are parsed when primary keys are absent."""
    res_p_age = evaluate_context({"patient_age": 32})
    assert set(res_p_age.keys()) == EXPECTED_OUTPUT_KEYS
    assert res_p_age["age"] == 32.0

    res_p_age_camel = evaluate_context({"patientAge": 28.5})
    assert res_p_age_camel["age"] == 28.5

    res_gender = evaluate_context({"gender": "female"})
    assert res_gender["sex"] == "female"

    res_p_sex = evaluate_context({"patient_sex": "male"})
    assert res_p_sex["sex"] == "male"

    res_p_gender = evaluate_context({"patientGender": "female"})
    assert res_p_gender["sex"] == "female"

    res_site = evaluate_context({"site": "chest"})
    assert res_site["localization"] == "chest"

    res_anat_site = evaluate_context({"anatomical_site": "scalp"})
    assert res_anat_site["localization"] == "scalp"
    assert "HEAD_NECK_SITE" in res_anat_site["context_flags"]

    res_anat_loc = evaluate_context({"anatomical_localization": "foot"})
    assert res_anat_loc["localization"] == "foot"
    assert "ACRAL_SITE" in res_anat_loc["context_flags"]


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
    assert set(res.keys()) == EXPECTED_OUTPUT_KEYS

    for forbidden in FORBIDDEN_GROUND_TRUTH_KEYS:
        assert forbidden not in res

    assert any("strictly quarantined" in note.lower() or "quarantined" in note.lower() for note in res["notes"])
    assert res["age"] == 55.0
    assert res["sex"] == "male"
    assert res["localization"] == "lower extremity"


def test_ground_truth_isolation_comparative():
    """Strengthened Ground-Truth Isolation (Specification Contract):
    Compare evaluations with and without forbidden diagnostic keys on identical demographic data.
    The demographic and flag outputs must be completely identical.
    The only allowed difference is the quarantine audit log in notes.
    """
    clean_meta = {
        "age": 55.0,
        "sex": "male",
        "localization": "lower extremity",
    }
    dirty_meta = {
        **clean_meta,
        "dx": "mel",
        "dx_type": "histopathology",
        "diagnosis": "malignant melanoma",
        "true_label": "mel",
        "ground_truth": "mel",
        "label": "mel",
        "label_name": "melanoma",
        "label_idx": 1,
    }
    res_clean = evaluate_context(clean_meta)
    res_dirty = evaluate_context(dirty_meta)

    assert set(res_clean.keys()) == EXPECTED_OUTPUT_KEYS
    assert set(res_dirty.keys()) == EXPECTED_OUTPUT_KEYS

    # Extractions and classifications must be identical
    assert res_dirty["age"] == res_clean["age"]
    assert res_dirty["sex"] == res_clean["sex"]
    assert res_dirty["localization"] == res_clean["localization"]
    assert res_dirty["missing_fields"] == res_clean["missing_fields"]
    assert res_dirty["context_flags"] == res_clean["context_flags"]

    # Forbidden keys must never appear in dirty output keys
    for forbidden in FORBIDDEN_GROUND_TRUTH_KEYS:
        assert forbidden not in res_dirty

    # Only allowed difference: quarantine audit note in dirty_meta
    assert any("strictly quarantined" in n.lower() or "quarantined" in n.lower() for n in res_dirty["notes"])
    assert not any("quarantined" in n.lower() for n in res_clean["notes"])


def test_ground_truth_quarantine_all_eight_keys():
    """Verify all 8 forbidden keys are quarantined, excluded from output, and trigger quarantine audit note."""
    for forbidden_key in FORBIDDEN_GROUND_TRUTH_KEYS:
        meta = {
            "age": 40.0,
            "sex": "female",
            "localization": "back",
            forbidden_key: "melanoma_ground_truth_value",
        }
        res = evaluate_context(meta)
        assert set(res.keys()) == EXPECTED_OUTPUT_KEYS
        assert forbidden_key not in res
        assert any("strictly quarantined" in n.lower() or "quarantined" in n.lower() for n in res["notes"])
        assert res["age"] == 40.0
        assert res["sex"] == "female"
        assert res["localization"] == "back"


def test_ground_truth_quarantine_case_insensitivity():
    """Verify ground-truth quarantine detects uppercase and mixed-case forbidden keys."""
    mixed_forbidden = {
        "DX": "bcc",
        "Dx_Type": "histopathology",
        "DIAGNOSIS": "basal cell carcinoma",
        "True_Label": "bcc",
        "Ground_Truth": "bcc",
        "LABEL": "bcc",
        "Label_Name": "bcc",
        "LABEL_IDX": 1,
    }
    meta = {"age": 60.0, "sex": "male", "localization": "chest", **mixed_forbidden}
    res = evaluate_context(meta)
    assert set(res.keys()) == EXPECTED_OUTPUT_KEYS

    for k in mixed_forbidden:
        assert k not in res
        assert k.lower() not in res

    assert any("strictly quarantined" in n.lower() or "quarantined" in n.lower() for n in res["notes"])


def test_ground_truth_quarantine_no_influence_on_clinical_evaluation():
    """Verify ground-truth values do not influence age, sex, localization, missing_fields, or context_flags."""
    meta = {
        "age": 8.0,
        "sex": "female",
        "localization": "hand",
        "dx": "mel",
        "diagnosis": "invasive melanoma",
        "true_label": "mel",
    }
    res = evaluate_context(meta)
    assert set(res.keys()) == EXPECTED_OUTPUT_KEYS
    assert res["age"] == 8.0
    assert res["sex"] == "female"
    assert res["localization"] == "hand"
    assert "PEDIATRIC_PATIENT" in res["context_flags"]
    assert "ACRAL_SITE" in res["context_flags"]
    assert "CONTEXT_COMPLETE" in res["context_flags"]

    for flag in res["context_flags"]:
        assert "MEL" not in flag
        assert "CANCER" not in flag
        assert "MALIGNANT" not in flag


def test_completeness_flags_mutually_exclusive():
    """Verify exactly one completeness flag is present in context_flags across all completeness states."""
    test_cases = [
        (None, "ALL_CONTEXT_MISSING"),
        ({}, "ALL_CONTEXT_MISSING"),
        ({"age": None, "sex": None, "localization": None}, "ALL_CONTEXT_MISSING"),
        ({"age": 30}, "INCOMPLETE_CONTEXT"),
        ({"sex": "male"}, "INCOMPLETE_CONTEXT"),
        ({"localization": "back"}, "INCOMPLETE_CONTEXT"),
        ({"age": 30, "sex": "male"}, "INCOMPLETE_CONTEXT"),
        ({"age": 30, "localization": "back"}, "INCOMPLETE_CONTEXT"),
        ({"sex": "female", "localization": "face"}, "INCOMPLETE_CONTEXT"),
        ({"age": 30, "sex": "male", "localization": "back"}, "CONTEXT_COMPLETE"),
    ]
    for meta, expected_flag in test_cases:
        res = evaluate_context(meta)
        assert set(res.keys()) == EXPECTED_OUTPUT_KEYS
        flags = res["context_flags"]
        present_completeness = [f for f in flags if f in COMPLETENESS_FLAGS]
        assert len(present_completeness) == 1, (
            f"Expected exactly 1 completeness flag for {meta!r}, got {present_completeness}"
        )
        assert present_completeness[0] == expected_flag


def test_observational_flags_clinical_definitions():
    """Verify all observational flags trigger under exact clinical criteria without diagnostic prediction."""
    # PEDIATRIC_PATIENT (<18)
    assert "PEDIATRIC_PATIENT" in evaluate_context({"age": 0})["context_flags"]
    assert "PEDIATRIC_PATIENT" in evaluate_context({"age": 17.9})["context_flags"]
    assert "PEDIATRIC_PATIENT" not in evaluate_context({"age": 18.0})["context_flags"]

    # GERIATRIC_PATIENT (>=65)
    assert "GERIATRIC_PATIENT" not in evaluate_context({"age": 64.9})["context_flags"]
    assert "GERIATRIC_PATIENT" in evaluate_context({"age": 65.0})["context_flags"]
    assert "GERIATRIC_PATIENT" in evaluate_context({"age": 95.0})["context_flags"]

    # HEAD_NECK_SITE
    for site in ("face", "scalp", "ear", "neck"):
        res = evaluate_context({"localization": site})
        assert set(res.keys()) == EXPECTED_OUTPUT_KEYS
        assert "HEAD_NECK_SITE" in res["context_flags"]

    # ACRAL_SITE
    for site in ("acral", "hand", "foot"):
        res = evaluate_context({"localization": site})
        assert set(res.keys()) == EXPECTED_OUTPUT_KEYS
        assert "ACRAL_SITE" in res["context_flags"]

    # GENITAL_SITE
    res_gen = evaluate_context({"localization": "genital"})
    assert set(res_gen.keys()) == EXPECTED_OUTPUT_KEYS
    assert "GENITAL_SITE" in res_gen["context_flags"]

    # Neutral site
    neutral_res = evaluate_context({"localization": "back"})
    for flag in ("HEAD_NECK_SITE", "ACRAL_SITE", "GENITAL_SITE"):
        assert flag not in neutral_res["context_flags"]


def test_observational_flags_no_diagnostic_predictions():
    """Verify observational flags are purely descriptive and contain no speculative diagnostic predictions."""
    res = evaluate_context({"age": 75, "sex": "male", "localization": "face"})
    assert set(res.keys()) == EXPECTED_OUTPUT_KEYS
    for flag in res["context_flags"]:
        assert not any(diag in flag for diag in ("MELANOMA", "CARCINOMA", "BENIGN", "MALIGNANT", "NEVUS"))


def test_output_schema_invariants():
    """Verify output dictionary strictly matches the agreed six-field schema with proper types."""
    res = evaluate_context({"age": 50, "sex": "female", "localization": "trunk"})
    assert set(res.keys()) == EXPECTED_OUTPUT_KEYS
    assert isinstance(res["age"], float)
    assert isinstance(res["sex"], str)
    assert isinstance(res["localization"], str)
    assert isinstance(res["missing_fields"], list)
    assert isinstance(res["context_flags"], list)
    assert isinstance(res["notes"], list)

    res_empty = evaluate_context(None)
    assert set(res_empty.keys()) == EXPECTED_OUTPUT_KEYS
    assert res_empty["age"] is None
    assert res_empty["sex"] is None
    assert res_empty["localization"] is None
    assert isinstance(res_empty["missing_fields"], list)
    assert isinstance(res_empty["context_flags"], list)
    assert isinstance(res_empty["notes"], list)


# ==============================================================================
# PART 2: SPECIFICATION AMBIGUITY TESTS (PENDING OWNER CONFIRMATION)
# ==============================================================================

@pytest.mark.skip(
    reason=(
        "UNRESOLVED SPECIFICATION DECISION: docs/context_rules.md lists 'unknown' under "
        "KNOWN_ANATOMICAL_SITES, but Section 4.4 states 'unknown' must be treated as missing. "
        "Awaiting specification owner confirmation before enforcing either behavior."
    )
)
def test_anatomical_localization_unknown_specification_ambiguity():
    """Ambiguity Test:
    - docs/context_rules.md Section 4.4 defines 'unknown' as a missing placeholder (missing_fields).
    - KNOWN_ANATOMICAL_SITES includes 'unknown' as a canonical site tuple member.

    Until the owner resolves this policy, this test is marked as unresolved and skipped.
    """
    res = evaluate_context({"localization": "unknown"})
    assert set(res.keys()) == EXPECTED_OUTPUT_KEYS
    # Intentionally unresolved: pending owner determination whether localization is None or 'unknown'


# ==============================================================================
# PART 3: ROBUSTNESS & IMPLEMENTATION EDGE-CASE TESTS
# ==============================================================================

def test_age_boolean_invalid():
    """Robustness Test: Boolean age is invalid and must not be coerced to numeric (1.0 or 0.0)."""
    res_true = evaluate_context({"age": True})
    assert set(res_true.keys()) == EXPECTED_OUTPUT_KEYS
    assert res_true["age"] is None, (
        f"Specification requires boolean True to be invalid age, but got {res_true['age']}"
    )
    assert "age" in res_true["missing_fields"]

    res_false = evaluate_context({"age": False})
    assert set(res_false.keys()) == EXPECTED_OUTPUT_KEYS
    assert res_false["age"] is None, (
        f"Specification requires boolean False to be invalid age, but got {res_false['age']}"
    )
    assert "age" in res_false["missing_fields"]


def test_alternative_keys_fallback_when_primary_alias_is_missing():
    """Robustness Test: Primary alias missing while another supported alias contains a valid value."""
    meta_age_fallback = {"age": None, "patient_age": 42.0}
    res_age = evaluate_context(meta_age_fallback)
    assert set(res_age.keys()) == EXPECTED_OUTPUT_KEYS
    assert res_age["age"] == 42.0, (
        f"Expected fallback to patient_age=42.0 when age is None, got {res_age['age']}"
    )
    assert "age" not in res_age["missing_fields"]

    meta_sex_fallback = {"sex": None, "gender": "male"}
    res_sex = evaluate_context(meta_sex_fallback)
    assert set(res_sex.keys()) == EXPECTED_OUTPUT_KEYS
    assert res_sex["sex"] == "male", (
        f"Expected fallback to gender='male' when sex is None, got {res_sex['sex']}"
    )

    meta_loc_fallback = {"localization": None, "site": "back"}
    res_loc = evaluate_context(meta_loc_fallback)
    assert set(res_loc.keys()) == EXPECTED_OUTPUT_KEYS
    assert res_loc["localization"] == "back", (
        f"Expected fallback to site='back' when localization is None, got {res_loc['localization']}"
    )


def test_alternative_keys_conflicts_transparency():
    """Robustness Test: Conflicting aliases should be handled transparently; do not silently assume which is correct."""
    meta_conflict = {"age": 20.0, "patient_age": 70.0}
    res = evaluate_context(meta_conflict)
    assert set(res.keys()) == EXPECTED_OUTPUT_KEYS
    notes_text = " ".join(res["notes"]).lower()
    has_conflict_notice = "conflict" in notes_text or "discrepancy" in notes_text or "multiple" in notes_text
    assert has_conflict_notice, (
        "Specification requires conflicting aliases to be documented transparently in notes, "
        f"but got notes: {res['notes']}"
    )


def test_anatomical_localization_unsupported_descriptions():
    """Robustness Test: Verify agreed validation behavior for unsupported anatomical descriptions.
    Contract rules state:
    1. The agent must never invent speculative mappings to unrelated canonical sites (e.g. 'cornea' -> 'face').
    2. Unsupported sites must never trigger specific anatomical clinical flags (HEAD_NECK_SITE, ACRAL_SITE, GENITAL_SITE).
    3. The six-field output contract must be preserved.
    """
    unsupported_sites = ["cornea", "oral mucosa", "elbow", "liver"]
    for unsup in unsupported_sites:
        res = evaluate_context({"localization": unsup})
        assert set(res.keys()) == EXPECTED_OUTPUT_KEYS

        # Verify no invented mappings to canonical sites
        assert res["localization"] != "face", f"Improper mapping of {unsup!r} to 'face'"
        assert res["localization"] != "back", f"Improper mapping of {unsup!r} to 'back'"
        assert res["localization"] != "trunk", f"Improper mapping of {unsup!r} to 'trunk'"

        # Verify no inappropriate clinical site flags triggered
        for flag in ("HEAD_NECK_SITE", "ACRAL_SITE", "GENITAL_SITE"):
            assert flag not in res["context_flags"], f"Unsupported site {unsup!r} falsely triggered {flag}"


# ==============================================================================
# SECTION 4: DIRECT EXECUTION HARNESS
# ==============================================================================

def run_all_tests():
    """Execute test suite directly in environments where pytest CLI is not invoked."""
    test_functions = [
        test_empty_and_none_metadata,
        test_full_valid_context,
        test_partial_context,
        test_demographic_age_ranges,
        test_age_boundary_exact_limits,
        test_age_invalid_ranges_and_non_numerics,
        test_sex_normalization,
        test_sex_normalization_case_and_whitespace,
        test_sex_normalization_extended_categories,
        test_sex_normalization_placeholders_and_unrecognized,
        test_anatomical_site_normalization,
        test_anatomical_localization_all_canonical_sites,
        test_anatomical_localization_missing_and_placeholders,
        test_alternative_metadata_keys_supported,
        test_ground_truth_isolation,
        test_ground_truth_isolation_comparative,
        test_ground_truth_quarantine_all_eight_keys,
        test_ground_truth_quarantine_case_insensitivity,
        test_ground_truth_quarantine_no_influence_on_clinical_evaluation,
        test_completeness_flags_mutually_exclusive,
        test_observational_flags_clinical_definitions,
        test_observational_flags_no_diagnostic_predictions,
        test_output_schema_invariants,
        # Part 3: Robustness tests
        test_age_boolean_invalid,
        test_alternative_keys_fallback_when_primary_alias_is_missing,
        test_alternative_keys_conflicts_transparency,
        test_anatomical_localization_unsupported_descriptions,
    ]

    print("=" * 70)
    print(f"  Running Agent A4 Test Suite ({len(test_functions)} active functions)")
    print("=" * 70)

    passed = 0
    failed = 0
    for test_fn in test_functions:
        try:
            test_fn()
            print(f"  [PASS] {test_fn.__name__}")
            passed += 1
        except AssertionError as exc:
            print(f"  [FAIL] {test_fn.__name__}: {exc}")
            failed += 1
        except Exception as exc:
            print(f"  [ERROR] {test_fn.__name__}: {type(exc).__name__}: {exc}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"  DIRECT RUN RESULTS: {passed} passed, {failed} failed out of {len(test_functions)}")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
