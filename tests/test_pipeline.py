"""
tests/test_pipeline.py — Unit and Integration Tests for End-to-End Pipeline & Submission.

Verifies:
1. run_pipeline execution across pristine, degraded, and missing-metadata scenarios.
2. Error isolation and safe fallbacks (A1 FAIL -> RE-IMAGE, A5 Veto -> REFER).
3. 3-column submission CSV generation, formatting, and strict validation.

Author: Palak (Integration Owner & Agent A6 Coordinator)
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.pipeline import run_pipeline, generate_submission_csv
from src.coordinator import ALLOWED_DECISIONS, CLASS_NAMES, POLICY_VERSION


def _create_synthetic_image(is_noisy: bool = False) -> np.ndarray:
    """Generate synthetic 224x224 RGB image for testing."""
    if is_noisy:
        return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    # Smooth benign-like disc
    img = np.full((224, 224, 3), 180, dtype=np.uint8)
    y, x = np.ogrid[:224, :224]
    mask = (x - 112) ** 2 + (y - 112) ** 2 <= 50 ** 2
    img[mask] = [90, 60, 40]
    return img


def test_pipeline_execution_basic():
    """Test standard pipeline execution on a synthetic image."""
    img = _create_synthetic_image()
    metadata = {"age": 45, "sex": "male", "localization": "back"}

    result = run_pipeline(img, metadata=metadata, image_id="ISIC_TEST_001")

    assert result["image_id"] == "ISIC_TEST_001"
    assert result["decision"] in ALLOWED_DECISIONS
    assert isinstance(result["prediction"], str)
    assert isinstance(result["requires_review"], bool)
    assert isinstance(result["rationale"], list)
    assert len(result["rationale"]) > 0
    assert result["policy_version"] == "1.0"
    assert "agent_evidence" in result
    assert "trace" in result
    assert "raw_agent_outputs" in result


def test_pipeline_error_isolation_on_invalid_input():
    """Test that pipeline safely handles invalid inputs without raising unhandled exceptions."""
    result = run_pipeline("non_existent_image_path.jpg", metadata=None, image_id="ISIC_INVALID")

    assert result["image_id"] == "ISIC_INVALID"
    # An invalid image fails quality or diagnosis and must safely REFER or RE-IMAGE, never crash
    assert result["decision"] in ["RE-IMAGE", "REFER"]
    assert result["requires_review"] is True


def test_generate_submission_csv_valid(tmp_path: Path):
    """Test generation of valid 3-column submission CSV."""
    results = [
        {"image_id": "ISIC_0001", "prediction": "nv", "decision": "AUTO-TRIAGE"},
        {"image_id": "ISIC_0002", "prediction": "mel", "decision": "REFER"},
        {"image_id": "ISIC_0003", "prediction": "bcc", "decision": "RE-IMAGE"},
    ]

    out_file = tmp_path / "submission.csv"
    df = generate_submission_csv(results, output_path=out_file)

    assert out_file.exists()
    assert list(df.columns) == ["image_id", "prediction", "decision"]
    assert len(df) == 3
    assert df.loc[0, "prediction"] == "nv"
    assert df.loc[0, "decision"] == "AUTO-TRIAGE"


def test_generate_submission_csv_duplicate_rejection():
    """Test that duplicate image_ids are strictly rejected."""
    results = [
        {"image_id": "ISIC_0001", "prediction": "nv", "decision": "AUTO-TRIAGE"},
        {"image_id": "ISIC_0001", "prediction": "mel", "decision": "REFER"},
    ]
    with pytest.raises(ValueError, match="Duplicate image_id"):
        generate_submission_csv(results)


def test_generate_submission_csv_invalid_decision():
    """Test that invalid decision enums are rejected."""
    results = [
        {"image_id": "ISIC_0001", "prediction": "nv", "decision": "DISCHARGE_HOME"},
    ]
    with pytest.raises(ValueError, match="Invalid decision"):
        generate_submission_csv(results)


def test_coordinator_agent_alias_compatibility():
    """Verify that src.coordinator_agent provides exact identical interface as src.coordinator."""
    import src.coordinator as c1
    import src.coordinator_agent as c2

    assert c1.POLICY_VERSION == c2.POLICY_VERSION == "1.0"
    assert c1.ALLOWED_DECISIONS == c2.ALLOWED_DECISIONS
    assert c1.SUBMISSION_COLUMNS == c2.SUBMISSION_COLUMNS
    assert c1.coordinate_triage is c2.coordinate_triage
