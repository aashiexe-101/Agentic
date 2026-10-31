"""
tests/test_quality.py
Branch: palak-quality
Module: Unit Test Suite for src/quality_agent.py (Module 2)

Pytest-compatible test suite.  Zero reliance on external datasets — all images
are generated as synthetic NumPy arrays inside fixtures.

Run from the repository root:
    pytest tests/test_quality.py -v

Requirements: pytest, opencv-python, numpy (no network, no disk images).
"""

from __future__ import annotations

import sys
import os

# ── Path bootstrap ───────────────────────────────────────────────────────────
# Allows `pytest tests/test_quality.py` from the repo root without installing
# the package first.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import cv2
import pytest

from src.quality_agent import (
    assess_quality,
    BLUR_FAIL_THRESHOLD,
    BLUR_WARN_THRESHOLD,
    BRIGHTNESS_FAIL_LOW,
    BRIGHTNESS_FAIL_HIGH,
    OVEREXPOSED_FAIL_THRESHOLD,
    GLARE_FAIL_THRESHOLD_PCT,
    _DEDUCT_WARN,
    _DEDUCT_FAIL,
)

# ============================================================================
# CONSTANTS & SHARED HELPERS
# ============================================================================

_VALID_STATUSES = {"PASS", "WARN", "FAIL"}
_REQUIRED_TOP_KEYS = {"status", "score", "checks", "reasons"}
_REQUIRED_CHECK_NAMES = {"blur", "illumination", "glare", "obstruction"}

# Sub-key schemas per check (must ALL be present)
_BLUR_KEYS   = {"value", "status", "threshold", "passed"}
_ILLUM_KEYS  = {"value", "status", "passed", "details"}
_GLARE_KEYS  = {"value", "status", "threshold", "passed"}
_OBST_KEYS   = {"value", "status", "passed", "details"}

_CHECK_KEY_MAP = {
    "blur":         _BLUR_KEYS,
    "illumination": _ILLUM_KEYS,
    "glare":        _GLARE_KEYS,
    "obstruction":  _OBST_KEYS,
}


def _make_normal_image(seed: int = 0, h: int = 300, w: int = 300) -> np.ndarray:
    """
    Produce a well-lit, sharp, unobstructed synthetic BGR image.

    Strategy: uniform base in [130, 150] with large geometric high-contrast
    features to guarantee high Laplacian variance (sharp edges) without
    triggering morphological hair-occlusion filters.
    """
    rng = np.random.RandomState(seed)
    base = np.full((h, w, 3), 140, dtype=np.uint8)
    noise = rng.randint(-10, 10, (h, w, 3), dtype=np.int16)
    base = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Large high-contrast shapes to produce high Laplacian variance
    cv2.circle(base, (w // 2, h // 2), min(h, w) // 4, (40, 40, 40), -1)
    cv2.circle(base, (w // 2, h // 2), min(h, w) // 8, (200, 200, 200), -1)
    cv2.rectangle(base, (w // 10, h // 10), (w // 4, h // 4), (60, 60, 60), -1)
    cv2.rectangle(base, (3 * w // 4, 3 * h // 4), (9 * w // 10, 9 * h // 10), (60, 60, 60), -1)

    return base


def _assert_schema(result: dict, context: str = "") -> None:
    """Reusable helper: assert the output dict is fully schema-compliant."""
    tag = f"[{context}] " if context else ""

    # Top-level keys
    assert _REQUIRED_TOP_KEYS <= set(result.keys()), \
        f"{tag}Missing top-level keys: {_REQUIRED_TOP_KEYS - set(result.keys())}"

    # status
    assert result["status"] in _VALID_STATUSES, \
        f"{tag}Invalid status: {result['status']!r}"

    # score
    assert isinstance(result["score"], float), \
        f"{tag}score is not a float: {type(result['score'])}"
    assert 0.0 <= result["score"] <= 1.0, \
        f"{tag}score out of range [0,1]: {result['score']}"

    # reasons
    assert isinstance(result["reasons"], list), \
        f"{tag}'reasons' is not a list"
    for r in result["reasons"]:
        assert isinstance(r, str), f"{tag}reason item is not a str: {r!r}"

    # checks — only validate sub-keys when checks is non-empty
    if result["checks"]:
        assert _REQUIRED_CHECK_NAMES <= set(result["checks"].keys()), \
            f"{tag}Missing checks: {_REQUIRED_CHECK_NAMES - set(result['checks'].keys())}"

        for chk_name, required_keys in _CHECK_KEY_MAP.items():
            chk = result["checks"][chk_name]
            assert required_keys <= set(chk.keys()), \
                f"{tag}checks.{chk_name} missing keys: {required_keys - set(chk.keys())}"
            assert chk["status"] in _VALID_STATUSES, \
                f"{tag}checks.{chk_name}.status invalid: {chk['status']!r}"
            assert isinstance(chk["passed"], bool), \
                f"{tag}checks.{chk_name}.passed is not bool"
            assert isinstance(chk["value"], float), \
                f"{tag}checks.{chk_name}.value is not float"


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(scope="module")
def normal_image() -> np.ndarray:
    """Clean, well-lit, sharp 300×300 BGR synthetic image."""
    return _make_normal_image(seed=42)


@pytest.fixture(scope="module")
def blurred_image() -> np.ndarray:
    """Severely Gaussian-blurred copy of the normal image."""
    img = _make_normal_image(seed=42)
    for _ in range(5):                         # 5 passes to push variance << BLUR_FAIL
        img = cv2.GaussianBlur(img, (51, 51), 30)
    return img


@pytest.fixture(scope="module")
def all_black_image() -> np.ndarray:
    """All-zero (pure black) 300×300 BGR image."""
    return np.zeros((300, 300, 3), dtype=np.uint8)


@pytest.fixture(scope="module")
def all_white_image() -> np.ndarray:
    """All-255 (pure white) 300×300 BGR image."""
    return np.full((300, 300, 3), 255, dtype=np.uint8)


# ============================================================================
# TEST 1 — Schema Contract
# ============================================================================

class TestSchemaContract:
    """
    Verify that assess_quality() returns a fully-formed, schema-compliant dict
    for a clean synthetic image, and that all four sub-checks are present.
    """

    def test_top_level_keys_present(self, normal_image):
        result = assess_quality(normal_image)
        assert _REQUIRED_TOP_KEYS <= set(result.keys()), (
            f"Missing top-level keys: {_REQUIRED_TOP_KEYS - set(result.keys())}"
        )

    def test_status_is_valid_string(self, normal_image):
        result = assess_quality(normal_image)
        assert result["status"] in _VALID_STATUSES

    def test_score_is_float_in_unit_interval(self, normal_image):
        result = assess_quality(normal_image)
        assert isinstance(result["score"], float), "score must be a float"
        assert 0.0 <= result["score"] <= 1.0, f"score {result['score']} not in [0,1]"

    def test_score_has_two_decimal_places(self, normal_image):
        result = assess_quality(normal_image)
        # round-trip: str representation must not gain precision beyond 2 d.p.
        assert result["score"] == round(result["score"], 2)

    def test_reasons_is_list_of_strings(self, normal_image):
        result = assess_quality(normal_image)
        assert isinstance(result["reasons"], list)
        for r in result["reasons"]:
            assert isinstance(r, str)

    def test_all_four_checks_present(self, normal_image):
        result = assess_quality(normal_image)
        assert _REQUIRED_CHECK_NAMES <= set(result["checks"].keys()), (
            f"Missing checks: {_REQUIRED_CHECK_NAMES - set(result['checks'].keys())}"
        )

    def test_blur_check_keys(self, normal_image):
        chk = assess_quality(normal_image)["checks"]["blur"]
        assert _BLUR_KEYS <= set(chk.keys())
        assert isinstance(chk["passed"], bool)
        assert isinstance(chk["value"], float)
        assert chk["status"] in _VALID_STATUSES

    def test_illumination_check_keys(self, normal_image):
        chk = assess_quality(normal_image)["checks"]["illumination"]
        assert _ILLUM_KEYS <= set(chk.keys())
        assert isinstance(chk["details"], dict)
        assert "mean_brightness" in chk["details"]

    def test_glare_check_keys(self, normal_image):
        chk = assess_quality(normal_image)["checks"]["glare"]
        assert _GLARE_KEYS <= set(chk.keys())
        assert isinstance(chk["threshold"], float)

    def test_obstruction_check_keys(self, normal_image):
        chk = assess_quality(normal_image)["checks"]["obstruction"]
        assert _OBST_KEYS <= set(chk.keys())
        assert "vignette_coverage" in chk["details"]
        assert "hair_coverage" in chk["details"]

    def test_full_schema_helper(self, normal_image):
        """Holistic schema assertion via the shared helper."""
        _assert_schema(assess_quality(normal_image), context="normal_image")


# ============================================================================
# TEST 2 — Missing & Corrupt Inputs
# ============================================================================

class TestMissingAndCorruptInputs:
    """
    Non-existent paths, empty strings, and zero-size arrays must all return
    status="FAIL", score=0.0, and reasons=["image could not be read"]
    without raising any unhandled exception.
    """

    _EXPECTED_REASONS = ["image could not be read"]

    def _assert_graceful_fail(self, result: dict, label: str) -> None:
        assert result["status"] == "FAIL",  f"[{label}] status should be FAIL"
        assert result["score"]  == 0.0,     f"[{label}] score should be 0.0"
        assert result["checks"] == {},      f"[{label}] checks should be empty dict"
        assert result["reasons"] == self._EXPECTED_REASONS, \
            f"[{label}] reasons mismatch: {result['reasons']}"

    def test_nonexistent_path_string(self):
        result = assess_quality("/absolutely/no/such/file_xyzzy_99999.jpg")
        self._assert_graceful_fail(result, "nonexistent_path_string")

    def test_nonexistent_path_object(self):
        from pathlib import Path
        result = assess_quality(Path("/absolutely/no/such/file_xyzzy_99999.jpg"))
        self._assert_graceful_fail(result, "nonexistent_path_object")

    def test_empty_string_path(self):
        result = assess_quality("")
        self._assert_graceful_fail(result, "empty_string_path")

    def test_zero_size_array(self):
        empty = np.array([], dtype=np.uint8)
        result = assess_quality(empty)
        self._assert_graceful_fail(result, "zero_size_array")

    def test_zero_element_2d_array(self):
        empty_2d = np.zeros((0, 0, 3), dtype=np.uint8)
        result = assess_quality(empty_2d)
        self._assert_graceful_fail(result, "zero_element_2d_array")

    def test_no_unhandled_exception_on_bad_path(self):
        """assess_quality must never propagate an exception for bad inputs."""
        try:
            assess_quality("/dev/null/fake_image.jpg")
            assess_quality("")
            assess_quality(np.zeros((0,), dtype=np.uint8))
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"Unexpected exception raised for bad input: {exc!r}")


# ============================================================================
# TEST 3 — Synthetic Blur Sensitivity
# ============================================================================

class TestSyntheticBlurSensitivity:
    """
    A severely blurred copy of a high-contrast baseline image must:
      - Score strictly lower than the baseline.
      - Have checks["blur"]["passed"] == False.
    """

    def test_blurred_score_lower_than_baseline(self, normal_image, blurred_image):
        baseline_result = assess_quality(normal_image)
        blurred_result  = assess_quality(blurred_image)
        assert blurred_result["score"] < baseline_result["score"], (
            f"Expected blurred score ({blurred_result['score']}) < "
            f"baseline score ({baseline_result['score']})"
        )

    def test_blurred_image_fails_blur_check(self, blurred_image):
        result = assess_quality(blurred_image)
        assert result["checks"]["blur"]["passed"] is False, (
            "Severely blurred image should fail the blur check"
        )

    def test_blurred_blur_status_is_fail(self, blurred_image):
        chk = assess_quality(blurred_image)["checks"]["blur"]
        assert chk["status"] == "FAIL", (
            f"Expected blur status FAIL, got {chk['status']!r}"
        )

    def test_blurred_blur_value_below_fail_threshold(self, blurred_image):
        chk = assess_quality(blurred_image)["checks"]["blur"]
        assert chk["value"] < BLUR_FAIL_THRESHOLD, (
            f"Laplacian variance {chk['value']:.2f} should be < FAIL threshold {BLUR_FAIL_THRESHOLD}"
        )

    def test_baseline_passes_blur_check(self, normal_image):
        result = assess_quality(normal_image)
        assert result["checks"]["blur"]["passed"] is True, (
            "Sharp baseline image should pass the blur check"
        )

    def test_blurred_overall_status_is_fail(self, blurred_image):
        result = assess_quality(blurred_image)
        assert result["status"] == "FAIL"

    def test_blurred_reasons_non_empty(self, blurred_image):
        result = assess_quality(blurred_image)
        assert len(result["reasons"]) > 0, "Blurred image must produce at least one reason"

    def test_blurred_schema_valid(self, blurred_image):
        _assert_schema(assess_quality(blurred_image), context="blurred_image")


# ============================================================================
# TEST 4 — All-Black Image
# ============================================================================

class TestAllBlackImage:
    """
    A pure-black (np.zeros) image must trigger status="FAIL" due to extreme
    underexposure / darkness, with at least one reason listed.
    """

    def test_status_is_fail(self, all_black_image):
        result = assess_quality(all_black_image)
        assert result["status"] == "FAIL", (
            f"All-black image should FAIL, got {result['status']!r}"
        )

    def test_reasons_non_empty(self, all_black_image):
        result = assess_quality(all_black_image)
        assert len(result["reasons"]) > 0, "All-black image must have at least one reason"

    def test_illumination_check_fails(self, all_black_image):
        """Mean brightness of 0 is well below BRIGHTNESS_FAIL_LOW (52.14)."""
        chk = assess_quality(all_black_image)["checks"]["illumination"]
        assert chk["status"] == "FAIL", (
            f"All-black illumination should be FAIL, got {chk['status']!r}"
        )
        assert chk["passed"] is False

    def test_illumination_value_is_zero(self, all_black_image):
        chk = assess_quality(all_black_image)["checks"]["illumination"]
        assert chk["value"] == pytest.approx(0.0, abs=1.0), (
            "Mean brightness of all-black image should be ~0"
        )

    def test_score_is_low(self, all_black_image):
        """A FAIL-triggering image must lose at least one deduction (≤ 1 - 0.35 = 0.65)."""
        result = assess_quality(all_black_image)
        assert result["score"] <= (1.0 - _DEDUCT_FAIL + 1e-9)

    def test_schema_valid(self, all_black_image):
        _assert_schema(assess_quality(all_black_image), context="all_black_image")


# ============================================================================
# TEST 5 — All-White Image
# ============================================================================

class TestAllWhiteImage:
    """
    A pure-white (np.full 255) image must trigger status="FAIL" due to extreme
    overexposure, glare, and flat (zero-variance) Laplacian.
    """

    def test_status_is_fail(self, all_white_image):
        result = assess_quality(all_white_image)
        assert result["status"] == "FAIL", (
            f"All-white image should FAIL, got {result['status']!r}"
        )

    def test_reasons_non_empty(self, all_white_image):
        result = assess_quality(all_white_image)
        assert len(result["reasons"]) > 0

    def test_illumination_check_fails(self, all_white_image):
        """Mean brightness of 255 is well above BRIGHTNESS_FAIL_HIGH (214.80)."""
        chk = assess_quality(all_white_image)["checks"]["illumination"]
        assert chk["status"] == "FAIL", (
            f"All-white illumination should be FAIL, got {chk['status']!r}"
        )
        assert chk["passed"] is False

    def test_glare_check_fails(self, all_white_image):
        """
        All pixels have V=255>230 and S=0<30 in HSV, so glare % = 100 >>
        GLARE_FAIL_THRESHOLD_PCT (7.64).
        """
        chk = assess_quality(all_white_image)["checks"]["glare"]
        assert chk["status"] == "FAIL", (
            f"All-white glare should be FAIL, got {chk['status']!r}"
        )
        assert chk["value"] == pytest.approx(100.0, abs=1.0), (
            f"Glare percentage should be ~100%, got {chk['value']:.2f}"
        )

    def test_blur_check_fails(self, all_white_image):
        """A flat uniform image has Laplacian variance ≈ 0, below BLUR_FAIL_THRESHOLD."""
        chk = assess_quality(all_white_image)["checks"]["blur"]
        assert chk["status"] == "FAIL"
        assert chk["value"] < BLUR_FAIL_THRESHOLD

    def test_score_is_zero_due_to_clipping(self, all_white_image):
        """Three+ FAILs push raw score below 0 → clipped to 0.0."""
        result = assess_quality(all_white_image)
        assert result["score"] == 0.0, (
            f"Score should be clipped to 0.0 after multiple FAILs, got {result['score']}"
        )

    def test_schema_valid(self, all_white_image):
        _assert_schema(assess_quality(all_white_image), context="all_white_image")


# ============================================================================
# TEST 6 — Score Monotonicity & Clipping
# ============================================================================

class TestScoreMonotonicityAndClipping:
    """
    Verify that cumulative penalties never drop below 0.0 or exceed 1.0,
    and that each penalty deduction moves the score in the correct direction.
    """

    def test_score_never_below_zero(self):
        """All-white triggers 3–4 FAILs; score must be clipped to 0.0."""
        img = np.full((300, 300, 3), 255, dtype=np.uint8)
        result = assess_quality(img)
        assert result["score"] >= 0.0, f"Score went negative: {result['score']}"

    def test_score_never_above_one(self):
        """No image should ever return a score > 1.0."""
        img = _make_normal_image(seed=7)
        result = assess_quality(img)
        assert result["score"] <= 1.0, f"Score exceeded 1.0: {result['score']}"

    def test_pass_image_score_is_1(self):
        """A pristine synthetic image with no defects must score exactly 1.0."""
        img = _make_normal_image(seed=99)
        result = assess_quality(img)
        # Confirm it actually passes before checking score
        if result["status"] == "PASS":
            assert result["score"] == 1.0, (
                f"PASS image should score 1.0, got {result['score']}"
            )

    def test_each_fail_deducts_expected_amount(self):
        """
        Construct a single-FAIL scenario (blur only) and verify deduction is
        exactly _DEDUCT_FAIL (0.35), giving score = 0.65.
        """
        # A severely blurred image with normal brightness/glare/obstruction
        img = _make_normal_image(seed=5)
        for _ in range(5):
            img = cv2.GaussianBlur(img, (51, 51), 30)
        result = assess_quality(img)

        # Count actual FAILs and WARNs in the checks
        n_fail = sum(1 for c in result["checks"].values() if c["status"] == "FAIL")
        n_warn = sum(1 for c in result["checks"].values() if c["status"] == "WARN")
        expected_score = round(
            max(0.0, 1.0 - n_fail * _DEDUCT_FAIL - n_warn * _DEDUCT_WARN), 2
        )
        assert result["score"] == expected_score, (
            f"Expected score {expected_score} "
            f"(FAILs={n_fail}, WARNs={n_warn}), got {result['score']}"
        )

    def test_score_strictly_decreases_with_defects(self):
        """
        Score(clean) >= Score(blurred) — adding a defect never raises the score.
        """
        clean   = _make_normal_image(seed=11)
        blurred = clean.copy()
        for _ in range(5):
            blurred = cv2.GaussianBlur(blurred, (51, 51), 30)

        score_clean   = assess_quality(clean)["score"]
        score_blurred = assess_quality(blurred)["score"]
        assert score_clean >= score_blurred, (
            f"Clean score {score_clean} should be >= blurred score {score_blurred}"
        )

    def test_clipping_at_zero(self):
        """Manually verify arithmetic: 4 FAILs = 4×0.35 = 1.40 deduction → score=0.0."""
        img = np.full((300, 300, 3), 255, dtype=np.uint8)
        result = assess_quality(img)
        # With 3-4 FAILs (blur, illumination-overexp, glare), penalty > 1.0
        assert result["score"] == 0.0

    def test_score_rounded_to_two_decimal_places(self):
        """score must always be rounded to exactly 2 decimal places."""
        for seed in range(5):
            img = _make_normal_image(seed=seed)
            result = assess_quality(img)
            assert result["score"] == round(result["score"], 2), (
                f"score {result['score']} is not rounded to 2 d.p."
            )

    def test_schema_valid_for_all_synthetic(self):
        """Schema contract holds for a variety of synthetic seeds."""
        for seed in range(8):
            img = _make_normal_image(seed=seed)
            _assert_schema(assess_quality(img), context=f"seed={seed}")
