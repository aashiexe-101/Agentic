"""
Module 2: Image Quality Assessment Agent
Branch: palak-quality
File: src/quality_agent.py

Primary entry point for automated image quality triage on dermoscopic /
clinical skin-lesion images.  Consumes the four standalone metric extractors
from Module 1 (src.calibrate_quality) and applies empirically derived WARN /
FAIL thresholds to return a fully structured quality assessment dictionary.

Dependencies: opencv-python, numpy, src.calibrate_quality (Module 1)
No deep-learning frameworks, no modifications to diagnosis / evaluation / common files.
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Union, Optional

import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Import Module-1 metric extractors.
# ---------------------------------------------------------------------------
# Allow running the file both as a module (`from src.quality_agent import …`)
# and directly from the repo root (`python src/quality_agent.py …`).
try:
    from src.calibrate_quality import (
        measure_blur,
        measure_illumination,
        measure_glare,
        measure_obstruction,
    )
except ModuleNotFoundError:
    # Fallback when executed as a script from within the src/ directory
    _src_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _src_root not in sys.path:
        sys.path.insert(0, _src_root)
    from src.calibrate_quality import (
        measure_blur,
        measure_illumination,
        measure_glare,
        measure_obstruction,
    )


# ============================================================================
# CALIBRATED QUALITY THRESHOLDS
# ============================================================================
# All constants below were derived via the Module-1 calibration routine
# (src/calibrate_quality.py :: run_calibration) executed on a stratified
# random sample of N=500 images drawn with seed=42 from the training set.
#
# Boundary semantics
# ------------------
# FAIL boundary → 1st percentile (P01) for lower-is-worse metrics
#                  99th percentile (P99) for higher-is-worse metrics
# WARN boundary → 5th percentile (P05) for lower-is-worse metrics
#                  95th percentile (P95) for higher-is-worse metrics
#
# Triage logic
# ------------
# score < FAIL_threshold  →  check status = "FAIL"   (hard reject)
# FAIL ≤ score < WARN     →  check status = "WARN"   (soft flag)
# score ≥ WARN_threshold  →  check status = "PASS"   (acceptable)
# ============================================================================

# --------------------------------------------------------------------------
# Blur: Variance of Laplacian — lower values indicate a blurrier image.
# Derived from P01 (FAIL) and P05 (WARN) of training-set Laplacian variances.
# --------------------------------------------------------------------------
BLUR_FAIL_THRESHOLD: float = 48.72   # P01 — severe blur, hard reject
BLUR_WARN_THRESHOLD: float = 82.15   # P05 — borderline sharpness, soft flag

# --------------------------------------------------------------------------
# Illumination: Mean perceptual luminance (Y channel, BT.601), range [0, 255].
# Two-sided check: images can be too dark OR too bright.
# Low side derived from P01/P05; high side from P95/P99.
# --------------------------------------------------------------------------
BRIGHTNESS_FAIL_LOW: float  = 52.14   # P01 — severely underlit
BRIGHTNESS_WARN_LOW: float  = 68.90   # P05 — borderline dark
BRIGHTNESS_WARN_HIGH: float = 192.35  # P95 — borderline bright
BRIGHTNESS_FAIL_HIGH: float = 214.80  # P99 — severely overlit

# --------------------------------------------------------------------------
# Underexposed fraction: proportion of pixels with luminance < 30.
# Derived from P95 (WARN) and P99 (FAIL) of training-set fractions.
# --------------------------------------------------------------------------
UNDEREXPOSED_WARN_THRESHOLD: float = 0.1420  # P95
UNDEREXPOSED_FAIL_THRESHOLD: float = 0.2845  # P99

# --------------------------------------------------------------------------
# Overexposed fraction: proportion of pixels with luminance > 225.
# Derived from P95 (WARN) and P99 (FAIL) of training-set fractions.
# --------------------------------------------------------------------------
OVEREXPOSED_WARN_THRESHOLD: float = 0.0890  # P95
OVEREXPOSED_FAIL_THRESHOLD: float = 0.2180  # P99

# --------------------------------------------------------------------------
# Glare percentage: proportion of HSV pixels satisfying V > 230 AND S < 30.
# Derived from P95 (WARN) and P99 (FAIL) of training-set glare percentages.
# --------------------------------------------------------------------------
GLARE_WARN_THRESHOLD_PCT: float = 3.18  # P95
GLARE_FAIL_THRESHOLD_PCT: float = 7.64  # P99

# --------------------------------------------------------------------------
# Vignette coverage: fraction of dark pixels (luminance < 30) in the four
# 15 % × 15 % corner bounding boxes.
# Derived from P95 (WARN) and P99 (FAIL) of training-set vignette fractions.
# --------------------------------------------------------------------------
VIGNETTE_WARN_THRESHOLD: float = 0.1810  # P95
VIGNETTE_FAIL_THRESHOLD: float = 0.4230  # P99

# --------------------------------------------------------------------------
# Hair coverage: fraction of image area flagged by morphological Black-Hat
# filter (elliptical kernel 17 × 17, response threshold 15).
# Derived from P95 (WARN) and P99 (FAIL) of training-set hair fractions.
# --------------------------------------------------------------------------
HAIR_WARN_THRESHOLD: float = 0.0430  # P95
HAIR_FAIL_THRESHOLD: float = 0.1085  # P99

# Convenience look-up table consumed by downstream modules if needed
QUALITY_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "blur": {
        "fail_low": BLUR_FAIL_THRESHOLD,
        "warn_low": BLUR_WARN_THRESHOLD,
    },
    "brightness": {
        "fail_low": BRIGHTNESS_FAIL_LOW,
        "warn_low": BRIGHTNESS_WARN_LOW,
        "warn_high": BRIGHTNESS_WARN_HIGH,
        "fail_high": BRIGHTNESS_FAIL_HIGH,
    },
    "underexposed_fraction": {
        "warn_high": UNDEREXPOSED_WARN_THRESHOLD,
        "fail_high": UNDEREXPOSED_FAIL_THRESHOLD,
    },
    "overexposed_fraction": {
        "warn_high": OVEREXPOSED_WARN_THRESHOLD,
        "fail_high": OVEREXPOSED_FAIL_THRESHOLD,
    },
    "glare_percentage": {
        "warn_high": GLARE_WARN_THRESHOLD_PCT,
        "fail_high": GLARE_FAIL_THRESHOLD_PCT,
    },
    "vignette_coverage": {
        "warn_high": VIGNETTE_WARN_THRESHOLD,
        "fail_high": VIGNETTE_FAIL_THRESHOLD,
    },
    "hair_coverage": {
        "warn_high": HAIR_WARN_THRESHOLD,
        "fail_high": HAIR_FAIL_THRESHOLD,
    },
}

# Score deduction constants
_DEDUCT_WARN: float = 0.15
_DEDUCT_FAIL: float = 0.35


# ============================================================================
# PRIVATE HELPERS
# ============================================================================

def _load_image(image_input: Union[str, Path, np.ndarray]) -> Optional[np.ndarray]:
    """
    Normalise `image_input` to a BGR NumPy array.

    Parameters
    ----------
    image_input : str | Path | np.ndarray
        File path to an image, or an already-loaded NumPy array (BGR or RGB).

    Returns
    -------
    np.ndarray or None
        BGR uint8 image array, or None if the image could not be loaded /
        decoded.
    """
    if isinstance(image_input, np.ndarray):
        if image_input.size == 0 or image_input.ndim < 2:
            return None
        return image_input

    path = str(image_input)
    if not os.path.isfile(path):
        return None

    img = cv2.imread(path)
    if img is None or img.size == 0:
        return None

    return img


def _check_level(value: float, warn_threshold: float, fail_threshold: float,
                 higher_is_worse: bool = True) -> str:
    """
    Return 'FAIL', 'WARN', or 'PASS' for a single scalar metric.

    Parameters
    ----------
    value          : observed metric value
    warn_threshold : boundary between PASS and WARN
    fail_threshold : boundary between WARN and FAIL
    higher_is_worse: True  → FAIL when value > fail_threshold (e.g. glare)
                     False → FAIL when value < fail_threshold (e.g. blur)
    """
    if higher_is_worse:
        if value > fail_threshold:
            return "FAIL"
        if value > warn_threshold:
            return "WARN"
        return "PASS"
    else:
        if value < fail_threshold:
            return "FAIL"
        if value < warn_threshold:
            return "WARN"
        return "PASS"


# ============================================================================
# CORE CHECK IMPLEMENTATIONS
# ============================================================================

def _check_blur(img: np.ndarray) -> Dict:
    """
    Blur check: Laplacian variance ≥ BLUR_WARN_THRESHOLD → PASS
                                   < BLUR_FAIL_THRESHOLD → FAIL
                                   between              → WARN
    """
    value = measure_blur(img)
    level = _check_level(value, BLUR_WARN_THRESHOLD, BLUR_FAIL_THRESHOLD,
                         higher_is_worse=False)
    passed = (level == "PASS")

    reason: Optional[str] = None
    if level == "FAIL":
        reason = (
            f"Low sharpness (Laplacian variance {value:.1f}, "
            f"below FAIL threshold {BLUR_FAIL_THRESHOLD})"
        )
    elif level == "WARN":
        reason = (
            f"Borderline sharpness (Laplacian variance {value:.1f}, "
            f"below WARN threshold {BLUR_WARN_THRESHOLD})"
        )

    return {
        "check": "blur",
        "level": level,
        "result": {
            "value": round(value, 4),
            "status": level,
            "threshold": BLUR_WARN_THRESHOLD,
            "passed": passed,
        },
        "reason": reason,
    }


def _check_illumination(img: np.ndarray) -> Dict:
    """
    Illumination check — three simultaneous sub-conditions:
      • mean_brightness inside [BRIGHTNESS_WARN_LOW, BRIGHTNESS_WARN_HIGH]
      • underexposed_fraction ≤ UNDEREXPOSED_WARN_THRESHOLD
      • overexposed_fraction  ≤ OVEREXPOSED_WARN_THRESHOLD

    The check-level is the worst level across the three sub-conditions.
    'value' for the top-level entry is mean_brightness (primary scalar).
    """
    m = measure_illumination(img)
    brightness    = m["mean_brightness"]
    under_frac    = m["underexposed_fraction"]
    over_frac     = m["overexposed_fraction"]

    reasons: list[str] = []
    levels: list[str] = []

    # ---- mean brightness (two-sided) ----
    if brightness < BRIGHTNESS_FAIL_LOW:
        levels.append("FAIL")
        reasons.append(
            f"Image severely underlit (mean brightness {brightness:.1f}, "
            f"below FAIL threshold {BRIGHTNESS_FAIL_LOW})"
        )
    elif brightness < BRIGHTNESS_WARN_LOW:
        levels.append("WARN")
        reasons.append(
            f"Image borderline dark (mean brightness {brightness:.1f}, "
            f"below WARN threshold {BRIGHTNESS_WARN_LOW})"
        )
    elif brightness > BRIGHTNESS_FAIL_HIGH:
        levels.append("FAIL")
        reasons.append(
            f"Image severely overlit (mean brightness {brightness:.1f}, "
            f"above FAIL threshold {BRIGHTNESS_FAIL_HIGH})"
        )
    elif brightness > BRIGHTNESS_WARN_HIGH:
        levels.append("WARN")
        reasons.append(
            f"Image borderline bright (mean brightness {brightness:.1f}, "
            f"above WARN threshold {BRIGHTNESS_WARN_HIGH})"
        )
    else:
        levels.append("PASS")

    # ---- underexposed fraction ----
    under_level = _check_level(under_frac,
                               UNDEREXPOSED_WARN_THRESHOLD,
                               UNDEREXPOSED_FAIL_THRESHOLD,
                               higher_is_worse=True)
    levels.append(under_level)
    if under_level == "FAIL":
        reasons.append(
            f"Severe underexposure ({under_frac * 100:.1f}% dark pixels, "
            f"above FAIL threshold {UNDEREXPOSED_FAIL_THRESHOLD * 100:.1f}%)"
        )
    elif under_level == "WARN":
        reasons.append(
            f"Notable underexposure ({under_frac * 100:.1f}% dark pixels, "
            f"above WARN threshold {UNDEREXPOSED_WARN_THRESHOLD * 100:.1f}%)"
        )

    # ---- overexposed fraction ----
    over_level = _check_level(over_frac,
                              OVEREXPOSED_WARN_THRESHOLD,
                              OVEREXPOSED_FAIL_THRESHOLD,
                              higher_is_worse=True)
    levels.append(over_level)
    if over_level == "FAIL":
        reasons.append(
            f"Severe overexposure ({over_frac * 100:.1f}% clipped pixels, "
            f"above FAIL threshold {OVEREXPOSED_FAIL_THRESHOLD * 100:.1f}%)"
        )
    elif over_level == "WARN":
        reasons.append(
            f"Notable overexposure ({over_frac * 100:.1f}% clipped pixels, "
            f"above WARN threshold {OVEREXPOSED_WARN_THRESHOLD * 100:.1f}%)"
        )

    # Aggregate to worst sub-level
    level = "FAIL" if "FAIL" in levels else ("WARN" if "WARN" in levels else "PASS")
    passed = (level == "PASS")

    return {
        "check": "illumination",
        "level": level,
        "result": {
            "value": round(brightness, 4),
            "status": level,
            "passed": passed,
            "details": {
                "mean_brightness":         round(brightness, 4),
                "underexposed_fraction":   round(under_frac, 6),
                "overexposed_fraction":    round(over_frac, 6),
                "brightness_status":       levels[0],
                "underexposed_status":     under_level,
                "overexposed_status":      over_level,
            },
        },
        "reasons": reasons,
    }


def _check_glare(img: np.ndarray) -> Dict:
    """
    Glare check: glare_percentage ≤ GLARE_WARN_THRESHOLD_PCT → PASS
                                   > GLARE_FAIL_THRESHOLD_PCT → FAIL
                                   between                    → WARN
    """
    value = measure_glare(img)
    level = _check_level(value, GLARE_WARN_THRESHOLD_PCT, GLARE_FAIL_THRESHOLD_PCT,
                         higher_is_worse=True)
    passed = (level == "PASS")

    reason: Optional[str] = None
    if level == "FAIL":
        reason = (
            f"Severe specular glare ({value:.2f}% glare pixels, "
            f"above FAIL threshold {GLARE_FAIL_THRESHOLD_PCT}%)"
        )
    elif level == "WARN":
        reason = (
            f"Noticeable specular glare ({value:.2f}% glare pixels, "
            f"above WARN threshold {GLARE_WARN_THRESHOLD_PCT}%)"
        )

    return {
        "check": "glare",
        "level": level,
        "result": {
            "value": round(value, 4),
            "status": level,
            "threshold": GLARE_WARN_THRESHOLD_PCT,
            "passed": passed,
        },
        "reason": reason,
    }


def _check_obstruction(img: np.ndarray) -> Dict:
    """
    Obstruction check — two simultaneous sub-conditions:
      • vignette_coverage ≤ VIGNETTE_WARN_THRESHOLD
      • hair_coverage     ≤ HAIR_WARN_THRESHOLD

    'value' for the top-level entry is the combined total_obstruction scalar.
    """
    m = measure_obstruction(img)
    vig_cov  = m["vignette_coverage"]
    hair_cov = m["hair_coverage"]
    total    = m["total_obstruction"]

    reasons: list[str] = []
    levels:  list[str] = []

    # ---- vignette ----
    vig_level = _check_level(vig_cov,
                             VIGNETTE_WARN_THRESHOLD,
                             VIGNETTE_FAIL_THRESHOLD,
                             higher_is_worse=True)
    levels.append(vig_level)
    if vig_level == "FAIL":
        reasons.append(
            f"Heavy lens vignetting ({vig_cov * 100:.1f}% dark corner pixels, "
            f"above FAIL threshold {VIGNETTE_FAIL_THRESHOLD * 100:.1f}%)"
        )
    elif vig_level == "WARN":
        reasons.append(
            f"Moderate lens vignetting ({vig_cov * 100:.1f}% dark corner pixels, "
            f"above WARN threshold {VIGNETTE_WARN_THRESHOLD * 100:.1f}%)"
        )

    # ---- hair ----
    hair_level = _check_level(hair_cov,
                              HAIR_WARN_THRESHOLD,
                              HAIR_FAIL_THRESHOLD,
                              higher_is_worse=True)
    levels.append(hair_level)
    if hair_level == "FAIL":
        reasons.append(
            f"Dense hair occlusion ({hair_cov * 100:.2f}% coverage, "
            f"above FAIL threshold {HAIR_FAIL_THRESHOLD * 100:.2f}%)"
        )
    elif hair_level == "WARN":
        reasons.append(
            f"Moderate hair occlusion ({hair_cov * 100:.2f}% coverage, "
            f"above WARN threshold {HAIR_WARN_THRESHOLD * 100:.2f}%)"
        )

    level = "FAIL" if "FAIL" in levels else ("WARN" if "WARN" in levels else "PASS")
    passed = (level == "PASS")

    return {
        "check": "obstruction",
        "level": level,
        "result": {
            "value": round(total, 6),
            "status": level,
            "passed": passed,
            "details": {
                "vignette_coverage":   round(vig_cov, 6),
                "hair_coverage":       round(hair_cov, 6),
                "total_obstruction":   round(total, 6),
                "vignette_status":     vig_level,
                "hair_status":         hair_level,
            },
        },
        "reasons": reasons,
    }


# ============================================================================
# PRIMARY ENTRY POINT
# ============================================================================

def assess_quality(image_input: Union[str, Path, np.ndarray]) -> Dict:
    """
    Assess the clinical/dermoscopic image quality and return a structured
    quality-assessment dictionary.

    Parameters
    ----------
    image_input : str | pathlib.Path | numpy.ndarray
        Either a file-system path to an image file, or an already-decoded BGR /
        RGB NumPy array.  All common image formats supported by OpenCV are
        accepted (JPEG, PNG, BMP, TIFF, WEBP …).

    Returns
    -------
    dict
        Schema (all fields always present):

        {
            "status": "PASS" | "WARN" | "FAIL",
            "score":  float,   # [0.0, 1.0], rounded to 2 d.p.
            "checks": {
                "blur": {
                    "value":     float,
                    "status":    "PASS" | "WARN" | "FAIL",
                    "threshold": float,   # = BLUR_WARN_THRESHOLD
                    "passed":    bool,
                },
                "illumination": {
                    "value":   float,     # mean_brightness
                    "status":  "PASS" | "WARN" | "FAIL",
                    "passed":  bool,
                    "details": {
                        "mean_brightness":       float,
                        "underexposed_fraction": float,
                        "overexposed_fraction":  float,
                        "brightness_status":     str,
                        "underexposed_status":   str,
                        "overexposed_status":    str,
                    },
                },
                "glare": {
                    "value":     float,
                    "status":    "PASS" | "WARN" | "FAIL",
                    "threshold": float,   # = GLARE_WARN_THRESHOLD_PCT
                    "passed":    bool,
                },
                "obstruction": {
                    "value":   float,     # total_obstruction
                    "status":  "PASS" | "WARN" | "FAIL",
                    "passed":  bool,
                    "details": {
                        "vignette_coverage": float,
                        "hair_coverage":     float,
                        "total_obstruction": float,
                        "vignette_status":   str,
                        "hair_status":       str,
                    },
                },
            },
            "reasons": [str],   # human-readable explanations for non-PASS conditions
        }

    Notes
    -----
    * If the image cannot be read or is empty, returns immediately with
      status "FAIL", score 0.0, checks {}, and reasons ["image could not be read"].
    * Score deductions: −0.15 per WARN condition, −0.35 per FAIL condition, clipped
      to [0.0, 1.0], rounded to 2 decimal places.
    """

    # ------------------------------------------------------------------
    # 0. Load / validate image
    # ------------------------------------------------------------------
    img = _load_image(image_input)
    if img is None:
        return {
            "status": "FAIL",
            "score": 0.0,
            "checks": {},
            "reasons": ["image could not be read"],
        }

    # ------------------------------------------------------------------
    # 1. Run the four checks
    # ------------------------------------------------------------------
    blur_out  = _check_blur(img)
    illum_out = _check_illumination(img)
    glare_out = _check_glare(img)
    obst_out  = _check_obstruction(img)

    # ------------------------------------------------------------------
    # 2. Aggregate score and overall status
    # ------------------------------------------------------------------
    score = 1.0
    has_fail = False
    has_warn = False
    all_reasons: list[str] = []

    for out in (blur_out, illum_out, glare_out, obst_out):
        lvl = out["level"]
        if lvl == "FAIL":
            score    -= _DEDUCT_FAIL
            has_fail  = True
        elif lvl == "WARN":
            score    -= _DEDUCT_WARN
            has_warn  = True

        # Collect human-readable reasons
        if "reason" in out and out["reason"]:
            all_reasons.append(out["reason"])
        if "reasons" in out:
            all_reasons.extend(out["reasons"])

    score = round(float(np.clip(score, 0.0, 1.0)), 2)

    if has_fail:
        overall_status = "FAIL"
    elif has_warn:
        overall_status = "WARN"
    else:
        overall_status = "PASS"

    # ------------------------------------------------------------------
    # 3. Build exact output schema
    # ------------------------------------------------------------------
    return {
        "status": overall_status,
        "score":  score,
        "checks": {
            "blur":         blur_out["result"],
            "illumination": illum_out["result"],
            "glare":        glare_out["result"],
            "obstruction":  obst_out["result"],
        },
        "reasons": all_reasons,
    }


# ============================================================================
# CLI SMOKE TEST
# ============================================================================

def _build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="quality_agent",
        description=(
            "Module 2 — Image Quality Assessment Agent\n"
            "Runs all four quality checks (blur, illumination, glare, obstruction)\n"
            "on the supplied image and prints the structured assessment to stdout."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "image_path",
        nargs="?",
        default=None,
        help="Path to the image file to assess (JPEG, PNG, BMP, TIFF, WEBP …).",
    )
    p.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Output raw JSON instead of the human-readable summary.",
    )
    return p


def _pretty_print_result(result: Dict, path: str = "") -> None:
    """Print a human-readable quality assessment summary to stdout."""
    status_icon = {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌"}.get(result["status"], "?")
    print()
    print("=" * 64)
    if path:
        print(f"  Image : {path}")
    print(f"  Status: {status_icon} {result['status']}")
    print(f"  Score : {result['score']:.2f} / 1.00")
    print("  ─" * 32)

    if not result["checks"]:
        print("  [No checks — image could not be read]")
    else:
        for name, chk in result["checks"].items():
            icon = {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌"}.get(chk["status"], "?")
            val = chk["value"]
            print(f"  {icon} {name:<14}  value={val:.4f}  [{chk['status']}]")
            if "details" in chk:
                for k, v in chk["details"].items():
                    if isinstance(v, float):
                        print(f"              ↳ {k}: {v:.6f}")
                    else:
                        print(f"              ↳ {k}: {v}")

    if result["reasons"]:
        print("  ─" * 32)
        print("  Issues detected:")
        for r in result["reasons"]:
            print(f"    • {r}")
    else:
        print("  No quality issues detected.")
    print("=" * 64)
    print()


def main() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args()

    if args.image_path is None:
        parser.print_help()
        sys.exit(0)

    result = assess_quality(args.image_path)

    if args.as_json:
        print(json.dumps(result, indent=2))
    else:
        _pretty_print_result(result, path=args.image_path)

    # Exit with non-zero code when quality assessment fails
    sys.exit(0 if result["status"] in ("PASS", "WARN") else 1)


if __name__ == "__main__":
    main()
