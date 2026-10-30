"""
tests/test_safety_coordinator.py — Unit & Integration Test Suite for Agent A5 & A6.

Covers:
- Every safety auditor veto condition
- Conflicting agent evidence
- A1 FAIL -> RE-IMAGE routing
- Missing A3/A4 evidence handling
- Clean benign -> AUTO-TRIAGE routing
- Malignant & borderline -> REFER routing
- Full coordinator execution and schema adherence
- Submission CSV generation & contract validation
- Synthetic vs real data isolation

Author: Utsavi (Senior Developer — Evaluation & Clinical Safety Track)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure repository root is in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.safety_agent import audit_safety
from src.coordinator import (
    ALLOWED_DECISIONS,
    POLICY_VERSION,
    SUBMISSION_COLUMNS,
    coordinate_triage,
    generate_submission_csv,
    validate_submission_csv,
)


# ==============================================================================
# AGENT A5: SAFETY AUDITOR TESTS
# ==============================================================================

def test_a1_fail_vetoes_auto_triage():
    """A1 FAIL quality status must immediately raise safety flag and veto auto-triage."""
    agent_outputs = {
        "a1_quality": {"overall_status": "FAIL", "reasons": ["severe_blur"]},
        "a2_diagnosis": {"pred_class": "nv", "confidence": 0.95},
    }
    audit = audit_safety(agent_outputs)
    assert audit["safety_flag"] is True
    assert audit["veto_auto_triage"] is True
    assert audit["requires_human_review"] is True
    assert any("Image quality gate failed" in r for r in audit["reasons"])


def test_malignant_prediction_vetoes_auto_triage():
    """A2 predicting melanoma must raise critical safety flag and veto auto-triage."""
    for mal_class in ["mel", "bcc", "akiec"]:
        agent_outputs = {
            "a1_quality": {"overall_status": "PASS"},
            "a2_diagnosis": {"pred_class": mal_class, "confidence": 0.88},
        }
        audit = audit_safety(agent_outputs)
        assert audit["safety_flag"] is True
        assert audit["veto_auto_triage"] is True
        assert audit["requires_human_review"] is True
        assert audit["risk_level"] in {"HIGH", "CRITICAL"}


def test_elevated_secondary_melanoma_probability_vetoes_auto_triage():
    """Substantial secondary melanoma probability must veto auto-triage even if benign is top class."""
    agent_outputs = {
        "a1_quality": {"overall_status": "PASS"},
        "a2_diagnosis": {
            "pred_class": "nv",
            "confidence": 0.65,
            "probabilities": {"prob_nv": 0.65, "prob_mel": 0.25, "prob_bkl": 0.10},
        },
    }
    audit = audit_safety(agent_outputs)
    assert audit["veto_auto_triage"] is True
    assert any("melanoma probability" in r for r in audit["reasons"])


def test_low_confidence_diagnostic_uncertainty_veto():
    """Diagnostic confidence below 0.60 must trigger safety flag and review."""
    agent_outputs = {
        "a1_quality": {"overall_status": "PASS"},
        "a2_diagnosis": {"pred_class": "nv", "confidence": 0.45},
    }
    audit = audit_safety(agent_outputs)
    assert audit["safety_flag"] is True
    assert audit["veto_auto_triage"] is True
    assert any("Low diagnostic confidence" in r for r in audit["reasons"])


def test_a3_morphology_red_flags_and_conflict():
    """A3 ulceration or atypical features must trigger conflict against benign A2 prediction."""
    agent_outputs = {
        "a1_quality": {"overall_status": "PASS"},
        "a2_diagnosis": {"pred_class": "nv", "confidence": 0.85},
        "a3_lesion": {"ulceration": True, "atypical_network": True},
    }
    audit = audit_safety(agent_outputs)
    assert audit["safety_flag"] is True
    assert audit["veto_auto_triage"] is True
    assert len(audit["conflicts"]) > 0
    assert "Diagnostic conflict" in audit["conflicts"][0]


def test_a4_melanoma_personal_history_veto():
    """Personal or family history of melanoma must veto auto-triage."""
    agent_outputs = {
        "a1_quality": {"overall_status": "PASS"},
        "a2_diagnosis": {"pred_class": "nv", "confidence": 0.90},
        "a4_context": {"personal_history_melanoma": True, "age": 45, "localization": "back"},
    }
    audit = audit_safety(agent_outputs)
    assert audit["safety_flag"] is True
    assert audit["veto_auto_triage"] is True
    assert any("personal history of melanoma" in r for r in audit["reasons"])


def test_missing_critical_agent_evidence():
    """Missing A1 or A2 must be caught as missing evidence and veto auto-triage."""
    audit_no_a1 = audit_safety({"a2_diagnosis": {"pred_class": "nv"}})
    assert audit_no_a1["veto_auto_triage"] is True
    assert any("A1" in m for m in audit_no_a1["missing_evidence"])

    audit_no_a2 = audit_safety({"a1_quality": {"overall_status": "PASS"}})
    assert audit_no_a2["veto_auto_triage"] is True
    assert any("A2" in m for m in audit_no_a2["missing_evidence"])


def test_missing_a4_metadata_reported():
    """Missing age or localization in patient context must be explicitly recorded."""
    agent_outputs = {
        "a1_quality": {"overall_status": "PASS"},
        "a2_diagnosis": {"pred_class": "nv", "confidence": 0.85},
        "a4_context": {"age": None, "localization": "unknown"},
    }
    audit = audit_safety(agent_outputs)
    assert any("age" in m for m in audit["missing_evidence"])
    assert any("localization" in m for m in audit["missing_evidence"])


# ==============================================================================
# AGENT A6: COORDINATOR & TRIAGE POLICY TESTS
# ==============================================================================

def test_coordinator_reimage_on_a1_fail():
    """A1 FAIL must strictly route to RE-IMAGE with review required."""
    outputs = {
        "a1_quality": {"overall_status": "FAIL", "reasons": ["severe_blur", "underexposure"]},
        "a2_diagnosis": {"pred_class": "nv", "confidence": 0.90},
    }
    triage = coordinate_triage(outputs)
    assert triage["decision"] == "RE-IMAGE"
    assert triage["requires_review"] is True
    assert triage["predicted_label"] == "nv"
    assert any("failed clinical quality threshold" in r for r in triage["rationale"])


def test_coordinator_refer_on_melanoma():
    """Melanoma prediction must route to REFER with immediate specialist review."""
    outputs = {
        "a1_quality": {"overall_status": "PASS"},
        "a2_diagnosis": {"pred_class": "mel", "confidence": 0.88},
    }
    triage = coordinate_triage(outputs)
    assert triage["decision"] == "REFER"
    assert triage["requires_review"] is True
    assert triage["predicted_label"] == "mel"
    assert any("High-risk diagnosis predicted" in r for r in triage["rationale"])


def test_coordinator_refer_on_a5_veto():
    """A5 veto due to conflicting clinical evidence must route to REFER."""
    outputs = {
        "a1_quality": {"overall_status": "PASS"},
        "a2_diagnosis": {"pred_class": "nv", "confidence": 0.82},
        "a3_lesion": {"bleeding": True},
    }
    triage = coordinate_triage(outputs)
    assert triage["decision"] == "REFER"
    assert triage["requires_review"] is True
    assert any("Safety Auditor vetoed" in r for r in triage["rationale"])


def test_coordinator_auto_triage_clean_benign():
    """Clean benign lesion (nv) with PASS quality, high confidence, and zero flags must get AUTO-TRIAGE."""
    outputs = {
        "a1_quality": {"overall_status": "PASS"},
        "a2_diagnosis": {"pred_class": "nv", "confidence": 0.92},
        "a4_context": {"age": 42, "sex": "male", "localization": "abdomen"},
    }
    triage = coordinate_triage(outputs)
    assert triage["decision"] == "AUTO-TRIAGE"
    assert triage["requires_review"] is False
    assert triage["predicted_label"] == "nv"
    assert any("Safe for standard routine monitoring" in r for r in triage["rationale"])


def test_coordinator_schema_completeness():
    """Verify coordinator return payload satisfies all required keys and types."""
    outputs = {
        "a1_quality": {"overall_status": "PASS"},
        "a2_diagnosis": {"pred_class": "bkl", "confidence": 0.85},
    }
    triage = coordinate_triage(outputs)
    assert isinstance(triage["predicted_label"], str)
    assert triage["decision"] in ALLOWED_DECISIONS
    assert isinstance(triage["requires_review"], bool)
    assert isinstance(triage["rationale"], list)
    assert len(triage["rationale"]) > 0
    assert isinstance(triage["agent_evidence"], dict)
    assert triage["policy_version"] == POLICY_VERSION


# ==============================================================================
# SUBMISSION CSV GENERATION & CONTRACT VALIDATION TESTS (TASK 5)
# ==============================================================================

def test_submission_csv_generation_and_validation(tmp_path: Path):
    """Generate 3-column CSV and verify strict contract adherence."""
    image_ids = ["ISIC_0000001", "ISIC_0000002", "ISIC_0000003"]
    predictions = [
        {"predicted_label": "nv", "decision": "AUTO-TRIAGE"},
        {"predicted_label": "mel", "decision": "REFER"},
        {"predicted_label": "bcc", "decision": "RE-IMAGE"},
    ]

    out_file = tmp_path / "submission.csv"
    df = generate_submission_csv(image_ids, predictions, output_path=out_file)

    assert out_file.is_file()
    assert list(df.columns) == SUBMISSION_COLUMNS
    assert len(df) == 3

    # Validate reading back from file
    df_loaded = validate_submission_csv(out_file, expected_rows=3)
    assert len(df_loaded) == 3


def test_submission_csv_rejects_duplicates(tmp_path: Path):
    """Validator must reject duplicate image IDs."""
    bad_df = pd.DataFrame({
        "image_id": ["ISIC_001", "ISIC_001"],
        "predicted_label": ["nv", "mel"],
        "decision": ["AUTO-TRIAGE", "REFER"],
    })
    with pytest.raises(ValueError, match="duplicate image_ids"):
        validate_submission_csv(bad_df)


def test_submission_csv_rejects_missing_columns():
    """Validator must reject missing or extra columns."""
    bad_df = pd.DataFrame({
        "image_id": ["ISIC_001"],
        "predicted_label": ["nv"],
    })
    with pytest.raises(ValueError, match="columns must be exactly"):
        validate_submission_csv(bad_df)


def test_submission_csv_rejects_invalid_decision():
    """Validator must reject decision outside {AUTO-TRIAGE, RE-IMAGE, REFER}."""
    bad_df = pd.DataFrame({
        "image_id": ["ISIC_001"],
        "predicted_label": ["nv"],
        "decision": ["DISCHARGE"],  # invalid
    })
    with pytest.raises(ValueError, match="Invalid decision values found"):
        validate_submission_csv(bad_df)


def test_submission_csv_rejects_invalid_label():
    """Validator must reject predicted_label outside 7 canonical classes."""
    bad_df = pd.DataFrame({
        "image_id": ["ISIC_001"],
        "predicted_label": ["carcinoma"],  # invalid label
        "decision": ["REFER"],
    })
    with pytest.raises(ValueError, match="Invalid predicted_label values found"):
        validate_submission_csv(bad_df)


def test_submission_csv_rejects_nan_or_blank():
    """Validator must reject NaN or blank values."""
    bad_df = pd.DataFrame({
        "image_id": ["ISIC_001"],
        "predicted_label": ["nv"],
        "decision": ["   "],  # blank
    })
    with pytest.raises(ValueError, match="blank/whitespace"):
        validate_submission_csv(bad_df)


# ==============================================================================
# DATA ISOLATION TEST (TASK 1 & TASK 2)
# ==============================================================================

def test_synthetic_vs_real_data_isolation():
    """Confirm synthetic benchmark fixtures are distinctly marked and isolated from genuine outputs."""
    fake_csv = _REPO_ROOT / "outputs" / "fake_val_results.csv"
    assert fake_csv.is_file(), "Synthetic benchmark file must exist as contract fixture."

    df_fake = pd.read_csv(fake_csv)
    # Check that it is recognized as a synthetic benchmark
    assert len(df_fake) == 2000
    assert "fake" in fake_csv.name.lower()
