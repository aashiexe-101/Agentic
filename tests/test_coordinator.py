"""
tests/test_coordinator.py — Unit Tests for Agent A6 Coordinator and Compatibility Aliases.

Validates:
1. Decision hierarchy rules (RE-IMAGE on optical failure, REFER on malignancy/veto/uncertainty, AUTO-TRIAGE on safe benign).
2. Exact enum adherence: {"AUTO-TRIAGE", "REFER", "RE-IMAGE"}.
3. Policy version string: "1.0".
4. Full agent traceability in agent_evidence and trace dictionary.
5. Zero-hallucination guarantee when diagnosis prediction is missing.
6. Drop-in alias equivalence between src/coordinator.py and src/coordinator_agent.py.

Author: Palak (Integration Owner & Agent A6 Coordinator)
"""

import pytest
from pathlib import Path
import pandas as pd

from src.coordinator import (
    ALLOWED_DECISIONS,
    CLASS_NAMES,
    POLICY_VERSION,
    SUBMISSION_COLUMNS,
    coordinate_triage,
    coordinate,
    triage_case,
    generate_submission_csv,
    validate_submission_csv,
)
import src.coordinator_agent as coordinator_agent


def test_coordinator_policy_version_and_enums():
    """Verify policy version is 1.0 and allowed decisions conform to contract."""
    assert POLICY_VERSION == "1.0"
    assert ALLOWED_DECISIONS == ["AUTO-TRIAGE", "RE-IMAGE", "REFER"]
    assert SUBMISSION_COLUMNS == ["image_id", "predicted_label", "decision"]


def test_coordinator_agent_wrapper_equivalence():
    """Verify src/coordinator_agent.py exports identical members as src/coordinator.py."""
    assert coordinator_agent.POLICY_VERSION == "1.0"
    assert coordinator_agent.ALLOWED_DECISIONS == ALLOWED_DECISIONS
    assert coordinator_agent.SUBMISSION_COLUMNS == SUBMISSION_COLUMNS
    assert coordinator_agent.coordinate_triage is coordinate_triage
    assert coordinator_agent.coordinate is coordinate
    assert coordinator_agent.triage_case is triage_case


def test_coordinator_traceability_payload():
    """Verify agent_evidence and trace include complete 5-agent evidence."""
    outputs = {
        "a1_quality": {"overall_status": "PASS", "score": 0.95},
        "a2_diagnosis": {"predicted_class": "nv", "confidence": 0.94},
        "a3_lesion": {"reliability": "HIGH", "flags": {"asymmetric": False}},
        "a4_context": {"context_flags": [], "missing_fields": []},
        "a5_safety": {"safety_flag": False, "veto_auto_triage": False, "risk_level": "LOW"},
    }

    result = coordinate_triage(outputs)

    assert result["decision"] == "AUTO-TRIAGE"
    assert result["policy_version"] == "1.0"
    assert "agent_evidence" in result
    assert "trace" in result
    trace = result["trace"]
    assert trace["a1_quality_status"] == "PASS"
    assert trace["a1_score"] == 0.95
    assert trace["a2_predicted_class"] == "nv"
    assert trace["a2_confidence"] == 0.94
    assert trace["a3_reliability"] == "HIGH"
    assert trace["a5_safety_flag"] is False
    assert trace["a5_veto"] is False


def test_coordinator_safe_handling_of_missing_diagnosis():
    """Verify that missing A2 diagnosis produces 'unknown' and REFER, never defaulting to nv."""
    outputs = {
        "a1_quality": {"overall_status": "PASS", "score": 0.95},
        "a2_diagnosis": {},  # Missing predicted_class
        "a3_lesion": {"reliability": "HIGH"},
        "a4_context": {},
        "a5_safety": {"safety_flag": False, "veto_auto_triage": False},
    }

    result = coordinate_triage(outputs)

    assert result["predicted_label"] == "unknown"
    assert result["decision"] == "REFER"
    assert result["requires_review"] is True
    assert any("missing" in r.lower() or "unknown" in r.lower() for r in result["rationale"])


def test_coordinator_reimage_on_quality_fail():
    """Verify that quality FAIL triggers RE-IMAGE regardless of other agents."""
    outputs = {
        "a1_quality": {"overall_status": "FAIL", "score": 0.20, "reasons": ["Severe optical blur"]},
        "a2_diagnosis": {"predicted_class": "nv", "confidence": 0.99},
        "a3_lesion": {"reliability": "LOW"},
        "a4_context": {},
        "a5_safety": {"safety_flag": False, "veto_auto_triage": False},
    }

    result = coordinate_triage(outputs)

    assert result["decision"] == "RE-IMAGE"
    assert result["requires_review"] is True


def test_coordinator_refer_on_melanoma():
    """Verify that malignant melanoma diagnosis always forces REFER."""
    outputs = {
        "a1_quality": {"overall_status": "PASS", "score": 0.95},
        "a2_diagnosis": {"predicted_class": "mel", "confidence": 0.92},
        "a3_lesion": {"reliability": "HIGH"},
        "a4_context": {},
        "a5_safety": {"safety_flag": True, "veto_auto_triage": True},
    }

    result = coordinate_triage(outputs)

    assert result["decision"] == "REFER"
    assert result["requires_review"] is True
