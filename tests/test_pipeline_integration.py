"""
tests/test_pipeline_integration.py — Integration Tests for Multi-Agent Pipeline & Submission.

Validates:
1. End-to-end multi-agent execution via run_pipeline().
2. Safe fallbacks and exception isolation across all 6 agents.
3. 3-column submission CSV output and strict validation rules.

Author: Palak (Integration Owner & Agent A6 Coordinator)
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.pipeline import run_pipeline, generate_submission_csv
from src.coordinator import ALLOWED_DECISIONS, CLASS_NAMES, POLICY_VERSION


def _create_synthetic_dermoscopy_image(color: tuple = (160, 110, 80)) -> np.ndarray:
    """Generate synthetic 224x224 RGB image with central pigmented lesion."""
    img = np.full((224, 224, 3), 190, dtype=np.uint8)
    y, x = np.ogrid[:224, :224]
    mask = (x - 112) ** 2 + (y - 112) ** 2 <= 45 ** 2
    img[mask] = color
    return img


def test_pipeline_integration_full_flow():
    """Verify run_pipeline executes all 6 agents and returns full contract."""
    img = _create_synthetic_dermoscopy_image()
    metadata = {"age": 52.0, "sex": "female", "localization": "lower extremity"}

    output = run_pipeline(img, metadata=metadata, image_id="ISIC_INTEG_001")

    assert output["image_id"] == "ISIC_INTEG_001"
    assert output["decision"] in ALLOWED_DECISIONS
    assert output["policy_version"] == "1.0"
    assert "agent_evidence" in output
    assert "trace" in output
    assert "raw_agent_outputs" in output

    raw = output["raw_agent_outputs"]
    assert "a1_quality" in raw
    assert "a2_diagnosis" in raw
    assert "a3_lesion" in raw
    assert "a4_context" in raw
    assert "a5_safety" in raw
    assert "a6_coordinator" in raw


def test_pipeline_integration_missing_metadata():
    """Verify pipeline functions gracefully with missing metadata (None)."""
    img = _create_synthetic_dermoscopy_image()
    output = run_pipeline(img, metadata=None, image_id="ISIC_INTEG_NO_META")

    assert output["image_id"] == "ISIC_INTEG_NO_META"
    assert output["decision"] in ALLOWED_DECISIONS
    raw = output["raw_agent_outputs"]
    assert "age" in raw["a4_context"]["missing_fields"]


def test_pipeline_submission_csv_contract(tmp_path: Path):
    """Verify generation of compliant 3-column CSV: image_id,prediction,decision."""
    results = [
        {"image_id": "ISIC_001001", "prediction": "nv", "decision": "AUTO-TRIAGE"},
        {"image_id": "ISIC_001002", "prediction": "mel", "decision": "REFER"},
        {"image_id": "ISIC_001003", "prediction": "bkl", "decision": "RE-IMAGE"},
    ]

    csv_path = tmp_path / "final_submission.csv"
    df = generate_submission_csv(results, output_path=csv_path, column_name="prediction")

    assert csv_path.exists()
    assert list(df.columns) == ["image_id", "prediction", "decision"]
    assert len(df) == 3

    # Check non-empty strings
    for col in ["image_id", "prediction", "decision"]:
        assert not df[col].isna().any()
        assert (df[col].astype(str).str.strip() != "").all()


def test_pipeline_submission_csv_invalid_class_rejection():
    """Verify that unsupported class predictions in AUTO-TRIAGE trigger ValueError."""
    results = [
        {"image_id": "ISIC_ERR", "prediction": "invalid_class_xyz", "decision": "AUTO-TRIAGE"}
    ]
    with pytest.raises(ValueError, match="AUTO-TRIAGE"):
        generate_submission_csv(results)
