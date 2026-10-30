"""Integration tests for the Screening Pipeline and Frontend Engine (app.py).

Module: tests/test_frontend.py
Author: Aashi (Frontend & Integration Owner)
"""

import io
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import run_screening_pipeline


def _create_synthetic_test_image() -> bytes:
    """Create a minimal 64x64 synthetic JPEG image in memory."""
    try:
        from PIL import Image
        img = Image.new("RGB", (64, 64), color=(180, 140, 120))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()
    except ImportError:
        # Minimal valid 1x1 JPEG bytes if PIL not available
        return (
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00"
            b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
            b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
            b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342"
            b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00"
            b"\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00"
            b"\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00\x08"
            b"\x01\x01\x00\x00?\x00\xbf\x00\xff\xd9"
        )


def test_pipeline_execution_and_schema():
    """Verify that run_screening_pipeline returns a complete, compliant schema."""
    img_bytes = _create_synthetic_test_image()
    context = {"age": 48.0, "sex": "male", "localization": "back"}

    report = run_screening_pipeline(img_bytes, context_metadata=context)

    # Top level keys
    assert "timestamp" in report
    assert "image_metadata" in report
    assert "agents" in report
    assert "final_decision" in report

    agents = report["agents"]
    assert "A1_quality" in agents
    assert "A2_diagnosis" in agents
    assert "A3_morphology" in agents
    assert "A4_context" in agents
    assert "A5_safety" in agents
    assert "A6_coordinator" in agents

    # Agent A4 output
    ctx = agents["A4_context"]
    assert ctx["age"] == 48.0
    assert ctx["sex"] == "male"
    assert ctx["localization"] == "back"
    assert len(ctx["missing_fields"]) == 0

    # Agent A2 should be transparently NOT AVAILABLE (no fake predictions)
    diag = agents["A2_diagnosis"]
    assert diag["status"] == "NOT AVAILABLE"
    assert diag["predicted_class"] is None

    # Final decision should not crash
    assert report["final_decision"] in ("RE-IMAGE", "REFER", "AUTO-TRIAGE", "PENDING DIAGNOSIS", "INCOMPLETE_PIPELINE")


def test_missing_context_handling():
    """Verify pipeline handles completely missing context gracefully."""
    img_bytes = _create_synthetic_test_image()
    report = run_screening_pipeline(img_bytes, context_metadata={})

    ctx = report["agents"]["A4_context"]
    assert ctx["age"] is None
    assert ctx["sex"] is None
    assert ctx["localization"] is None
    assert set(ctx["missing_fields"]) == {"age", "sex", "localization"}


def test_report_json_serializability():
    """Verify report can be serialized to JSON for clinical export."""
    img_bytes = _create_synthetic_test_image()
    report = run_screening_pipeline(img_bytes, context_metadata={"age": 60})

    json_str = json.dumps(report)
    assert len(json_str) > 100
    parsed = json.loads(json_str)
    assert parsed["final_decision"] == report["final_decision"]


def run_all_tests():
    """Execute all tests sequentially for direct script execution."""
    print("=" * 65)
    print("  Running test_frontend.py Integration Suite")
    print("=" * 65)

    test_pipeline_execution_and_schema()
    print("  [PASS] test_pipeline_execution_and_schema")

    test_missing_context_handling()
    print("  [PASS] test_missing_context_handling")

    test_report_json_serializability()
    print("  [PASS] test_report_json_serializability")

    print("\n" + "=" * 65)
    print("  ALL FRONTEND & INTEGRATION TESTS PASSED!")
    print("=" * 65)


if __name__ == "__main__":
    run_all_tests()
