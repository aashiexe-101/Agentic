"""
src/lesion_agent.py — Agent A3: Lesion Morphology Analysis
===========================================================
Branch  : feature/palak-lesion-analysis
Author  : Palak (A1 Quality / A3 Lesion Analysis owner)
Project : Agentic DermaGuard — JAI 2026 Healthcare Hackathon

Public function
---------------
    analyze_lesion(image) -> dict

Implements transparent, classical image-analysis steps for ABCD-aligned
lesion morphology evidence extraction.  It does NOT produce clinical risk
scores or clinical diagnoses.  When segmentation is unreliable, the module
returns null values with explicit reliability flags rather than fabricated
morphology scores.

Design philosophy
-----------------
1. Every step is documented and auditable.
2. Reliability is reported alongside every metric.
3. No deep-learning models are used — only classical OpenCV / NumPy / SciPy.
4. Output schema is stable and versioned (SCHEMA_VERSION constant).
5. All limitations are surfaced in the "limitations" list.

Output schema (coordinate with Utsavi before changing)
------------------------------------------------------
{
    "schema_version": "1.0",
    "reliable": bool,               # Top-level boolean: True if reliability != "LOW"
    "error": str | null,            # Primary failure reason when unreliable, else null
    "features": [str],              # Alias pointing to structural_features
    "evidence": [str],              # Alias pointing to limitations
    "flags": {                      # Active boolean findings
        "asymmetric": bool,
        "irregular_border": bool,
        "colour_variegated": bool,
        "unreliable_morphology": bool,
        ...
    },
    "asymmetry": {
        "value": float | null,      # 0.0 = perfect symmetry, 1.0 = maximum asymmetry
        "reliable": bool,
        "method": str
    },
    "border_irregularity": {
        "value": float | null,      # 0.0 = smooth circle, higher = more irregular
        "reliable": bool,
        "method": str,
        "compactness": float | null # 4π·Area / Perimeter²
    },
    "colour_variegation": {
        "value": float | null,      # coefficient of variation of hue within lesion
        "reliable": bool,
        "method": str,
        "details": {
            "mean_hue": float | null,
            "std_hue": float | null,
            "n_colour_clusters": int | null
        }
    },
    "structural_features": [str],   # e.g. ["blue_white_veil_candidate",
                                    #        "atypical_network_candidate"]
    "suspicion_score": float | null, # [0.0, 1.0] — purely heuristic, NOT clinical
    "reliability": "HIGH" | "MEDIUM" | "LOW",
    "segmentation": {
        "method": str,
        "mask_coverage": float | null,  # fraction of image covered by lesion mask
        "contour_count": int | null,
        "success": bool
    },
    "limitations": [str]
}

Dependencies
------------
    opencv-python >= 4.5
    numpy >= 1.21
    scipy (optional — used for clustering; gracefully degraded if absent)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from scipy.ndimage import label as scipy_label
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Schema version — bump when output structure changes (coordinate with Utsavi)
# ---------------------------------------------------------------------------
SCHEMA_VERSION = "1.0"

# ---------------------------------------------------------------------------
# Segmentation parameters
# ---------------------------------------------------------------------------
_MIN_LESION_COVERAGE = 0.03    # mask must cover ≥ 3% of image to be reliable
_MAX_LESION_COVERAGE = 0.92    # mask must cover ≤ 92% (else background leak)
_MIN_CONTOUR_AREA_PX = 500     # ignore tiny noise contours

# ---------------------------------------------------------------------------
# Colour thresholds (HSV, OpenCV ranges: H[0-179] S[0-255] V[0-255])
# ---------------------------------------------------------------------------
_BLUE_WHITE_VEIL_V_MIN = 180
_BLUE_WHITE_VEIL_S_MAX = 80
_BLUE_WHITE_VEIL_H_MIN = 90
_BLUE_WHITE_VEIL_H_MAX = 130
_BLUE_WHITE_VEIL_COVERAGE_THRESH = 0.06  # ≥ 6% of lesion → candidate

_DARK_NETWORK_V_MAX = 80
_DARK_NETWORK_COVERAGE_THRESH = 0.08     # ≥ 8% of lesion → candidate

_REGRESSION_V_MIN = 200
_REGRESSION_S_MAX = 30
_REGRESSION_COVERAGE_THRESH = 0.05      # ≥ 5% → white regression candidate

# Heuristic suspicion weights for score calculation (NOT clinical)
_SUSPICION_WEIGHTS = {
    "asymmetry": 0.30,
    "border": 0.25,
    "colour": 0.25,
    "structures": 0.20,
}


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def _load_image(image: Any) -> tuple[np.ndarray | None, str | None]:
    """
    Load image from a file path (str/Path) or a NumPy array.

    Returns (bgr_array, error_message).  bgr_array is None on failure.
    """
    if isinstance(image, (str, Path)):
        path = Path(image)
        if not path.is_file():
            return None, f"File not found: {path}"
        bgr = cv2.imread(str(path))
        if bgr is None:
            return None, f"cv2.imread returned None for: {path}"
        return bgr, None

    if isinstance(image, np.ndarray):
        if image.size == 0:
            return None, "Empty NumPy array provided."
        if image.ndim == 2:
            # Grayscale → convert to BGR for consistent downstream processing
            bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            return bgr, None
        if image.ndim == 3 and image.shape[2] == 3:
            return image.copy(), None
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR), None
        return None, f"Unsupported NumPy array shape: {image.shape}"

    return None, f"Unsupported image type: {type(image).__name__}"


# ---------------------------------------------------------------------------
# Step 1 — Preprocessing
# ---------------------------------------------------------------------------

def _preprocess(bgr: np.ndarray) -> np.ndarray:
    """
    Light denoising to reduce JPEG artefacts and hair-like noise before
    segmentation.  Uses a small bilateral filter that preserves edges.
    """
    return cv2.bilateralFilter(bgr, d=9, sigmaColor=75, sigmaSpace=75)


# ---------------------------------------------------------------------------
# Step 2 — Segmentation (lesion vs. background skin)
# ---------------------------------------------------------------------------

def _segment_lesion(bgr: np.ndarray) -> tuple[np.ndarray, str, dict]:
    """
    Produce a binary lesion mask using a multi-strategy approach.

    Strategy
    --------
    1. Convert to grayscale, apply CLAHE for local contrast enhancement.
    2. Threshold with Otsu's method.
    3. The lesion is typically darker than surrounding skin; invert if the
       thresholded foreground covers > 50% (background/skin is lighter).
    4. Apply morphological closing to fill small gaps, then opening to
       remove noise.
    5. Select the largest connected component as the lesion candidate.

    Returns
    -------
    (mask, method_description, diagnostics_dict)
    mask : uint8 binary image, 255 = lesion, 0 = background
    """
    h, w = bgr.shape[:2]
    total_px = h * w

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # CLAHE for local contrast normalisation
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Otsu threshold
    _, otsu = cv2.threshold(enhanced, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Dermoscopic images: lesion is usually darker → invert if fg > 50%
    fg_fraction = np.count_nonzero(otsu) / total_px
    if fg_fraction > 0.50:
        otsu = cv2.bitwise_not(otsu)

    # Morphological cleanup
    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (int(min(h, w) * 0.04) | 1,
                            int(min(h, w) * 0.04) | 1))
    kernel_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (int(min(h, w) * 0.02) | 1,
                            int(min(h, w) * 0.02) | 1))
    closed = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel_close)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

    # Keep only the largest connected component
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        opened, connectivity=8)

    mask = np.zeros((h, w), dtype=np.uint8)
    if n_labels > 1:
        # stats columns: LEFT, TOP, WIDTH, HEIGHT, AREA; label 0 is background
        largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask[labels == largest_label] = 255

    diagnostics = {
        "otsu_fg_fraction_before_invert": float(fg_fraction),
        "was_inverted": fg_fraction > 0.50,
        "connected_components": int(n_labels - 1),
    }

    return mask, "CLAHE + Otsu + largest-connected-component", diagnostics


def _assess_mask_reliability(mask: np.ndarray) -> tuple[bool, list[str]]:
    """
    Decide whether the segmentation mask is reliable enough for metric
    extraction.

    Returns (reliable_bool, list_of_limitation_strings).
    """
    h, w = mask.shape
    total_px = h * w
    coverage = float(np.count_nonzero(mask)) / total_px

    limitations: list[str] = []

    if coverage < _MIN_LESION_COVERAGE:
        limitations.append(
            f"Lesion mask too small ({coverage:.1%} of image < "
            f"{_MIN_LESION_COVERAGE:.0%} minimum).")
        return False, limitations

    if coverage > _MAX_LESION_COVERAGE:
        limitations.append(
            f"Lesion mask covers {coverage:.1%} of image — likely a "
            "segmentation leak (background included in mask).")
        return False, limitations

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    large_contours = [c for c in contours
                      if cv2.contourArea(c) >= _MIN_CONTOUR_AREA_PX]
    if not large_contours:
        limitations.append("No contour large enough found after segmentation.")
        return False, limitations

    return True, limitations


# ---------------------------------------------------------------------------
# Step 3 — Asymmetry
# ---------------------------------------------------------------------------

def _compute_asymmetry(mask: np.ndarray) -> tuple[float, str]:
    """
    Measure shape asymmetry by comparing the lesion mask to its 180°-rotated
    version after alignment to the principal axis.

    Method
    ------
    1. Compute the centroid and orientation of the lesion via image moments.
    2. Rotate the mask so its principal axis is horizontal.
    3. Flip horizontally and compare with the original → horizontal asymmetry.
    4. Flip vertically and compare with the original → vertical asymmetry.
    5. Asymmetry value = mean(horizontal, vertical) ∈ [0.0, 1.0].
       0.0 = perfectly symmetric, 1.0 = maximally asymmetric.

    Returns (asymmetry_value, method_description).
    """
    M = cv2.moments(mask)
    if M["m00"] == 0:
        return 0.0, "moments-failed (zero area)"

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # Central moments for orientation angle
    mu20 = M["mu20"] / M["m00"]
    mu02 = M["mu02"] / M["m00"]
    mu11 = M["mu11"] / M["m00"]
    angle_rad = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    angle_deg = float(np.degrees(angle_rad))

    # Rotate to align principal axis to horizontal
    h, w = mask.shape
    rot_mat = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    rotated = cv2.warpAffine(mask, rot_mat, (w, h),
                             flags=cv2.INTER_NEAREST)

    total = float(np.count_nonzero(rotated))
    if total == 0:
        return 0.0, "rotation-empty"

    # Horizontal flip asymmetry
    flipped_h = cv2.flip(rotated, 1)
    diff_h = float(np.count_nonzero(
        cv2.bitwise_xor(rotated, flipped_h))) / (2.0 * total)

    # Vertical flip asymmetry
    flipped_v = cv2.flip(rotated, 0)
    diff_v = float(np.count_nonzero(
        cv2.bitwise_xor(rotated, flipped_v))) / (2.0 * total)

    asymmetry = float(np.clip((diff_h + diff_v) / 2.0, 0.0, 1.0))
    method = ("principal-axis rotation + horizontal/vertical flip XOR "
               "(IoU-complement)")
    return asymmetry, method


# ---------------------------------------------------------------------------
# Step 4 — Border irregularity
# ---------------------------------------------------------------------------

def _compute_border_irregularity(
        mask: np.ndarray) -> tuple[float | None, float | None, str]:
    """
    Measure border irregularity using the compactness (isoperimetric quotient)
    of the lesion contour.

    Compactness = 4π·Area / Perimeter²
      - A perfect circle: compactness = 1.0
      - Irregular shapes: compactness < 1.0

    Border irregularity = 1 − compactness  ∈ [0.0, 1.0]
      - 0.0 = perfectly circular (no irregularity)
      - 1.0 = maximally irregular

    Returns (irregularity, compactness, method_description).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, None, "no-contour"

    # Use the largest contour
    main_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(main_contour)
    perimeter = cv2.arcLength(main_contour, closed=True)

    if perimeter < 1.0 or area < 1.0:
        return None, None, "degenerate-contour"

    compactness = float(4.0 * np.pi * area / (perimeter ** 2))
    compactness = float(np.clip(compactness, 0.0, 1.0))
    irregularity = float(1.0 - compactness)

    return irregularity, compactness, "isoperimetric-quotient (1 − compactness)"


# ---------------------------------------------------------------------------
# Step 5 — Colour variegation
# ---------------------------------------------------------------------------

def _compute_colour_variegation(
        bgr: np.ndarray,
        mask: np.ndarray) -> tuple[float | None, dict, str]:
    """
    Measure colour variation within the lesion region.

    Method
    ------
    1. Extract pixels within the lesion mask.
    2. Convert to HSV. Hue captures dermoscopic colour variety well.
    3. Compute Coefficient of Variation (CoV) of hue: std_hue / 90.0
       (divided by 90 to normalise; max meaningful hue range is ~90° for
        skin lesions spanning red to blue).
    4. Attempt simple k-means colour clustering (k=5) if ≥50 lesion pixels;
       report number of clusters with >1% membership as colour count.

    Returns (variegation_value, details_dict, method_description).
    variegation_value ∈ [0.0, 1.0].
    """
    lesion_pixels_bgr = bgr[mask > 0]
    if len(lesion_pixels_bgr) < 50:
        return None, {}, "insufficient-lesion-pixels"

    # Convert lesion pixels to HSV
    lesion_bgr_img = lesion_pixels_bgr.reshape(-1, 1, 3).astype(np.uint8)
    lesion_hsv_img = cv2.cvtColor(lesion_bgr_img, cv2.COLOR_BGR2HSV)
    hue = lesion_hsv_img[:, 0, 0].astype(np.float32)  # H ∈ [0, 179]

    mean_hue = float(np.mean(hue))
    std_hue = float(np.std(hue))

    # Normalise by 90 (plausible max hue spread for skin lesions)
    variegation = float(np.clip(std_hue / 90.0, 0.0, 1.0))

    # Colour cluster count via simple k-means
    n_colour_clusters: int | None = None
    try:
        if len(lesion_pixels_bgr) >= 50:
            pixels_f32 = lesion_pixels_bgr.astype(np.float32)
            k = 5
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        20, 1.0)
            _, labels_km, _ = cv2.kmeans(
                pixels_f32, k, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
            counts = np.bincount(labels_km.flatten(), minlength=k)
            threshold = 0.01 * len(lesion_pixels_bgr)
            n_colour_clusters = int(np.sum(counts >= threshold))
    except cv2.error:
        n_colour_clusters = None

    details = {
        "mean_hue": round(mean_hue, 2),
        "std_hue": round(std_hue, 2),
        "n_colour_clusters": n_colour_clusters,
    }
    method = ("HSV hue CoV normalised by 90° + "
               "k-means(k=5) cluster count within lesion mask")
    return variegation, details, method


# ---------------------------------------------------------------------------
# Step 6 — Structural features (qualitative candidates only)
# ---------------------------------------------------------------------------

def _extract_structural_features(bgr: np.ndarray,
                                  mask: np.ndarray) -> list[str]:
    """
    Identify dermoscopic structural feature candidates using colour-based
    heuristics.

    IMPORTANT: These are *candidates*, not validated clinical findings.
    Each feature string is labelled with "_candidate" to make this explicit.

    Features tested
    ---------------
    - blue_white_veil_candidate  : High V, low S, hue in blue range within mask
    - dark_network_candidate     : Very dark pixels forming a mesh (low V)
    - regression_white_candidate : Bright, desaturated white areas (regression)
    - peripheral_lighter_rim     : Mean brightness of mask border > centre
      (suggestive of regression or amelanotic halo)
    """
    features: list[str] = []
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    lesion_hsv = hsv[mask > 0]
    if len(lesion_hsv) == 0:
        return features

    total_lesion_px = len(lesion_hsv)

    # Blue-white veil candidate
    bwv_mask_condition = (
        (lesion_hsv[:, 0] >= _BLUE_WHITE_VEIL_H_MIN) &
        (lesion_hsv[:, 0] <= _BLUE_WHITE_VEIL_H_MAX) &
        (lesion_hsv[:, 1] <= _BLUE_WHITE_VEIL_S_MAX) &
        (lesion_hsv[:, 2] >= _BLUE_WHITE_VEIL_V_MIN)
    )
    if np.sum(bwv_mask_condition) / total_lesion_px >= _BLUE_WHITE_VEIL_COVERAGE_THRESH:
        features.append("blue_white_veil_candidate")

    # Dark network candidate (atypical pigment network)
    dark_condition = lesion_hsv[:, 2] <= _DARK_NETWORK_V_MAX
    if np.sum(dark_condition) / total_lesion_px >= _DARK_NETWORK_COVERAGE_THRESH:
        features.append("dark_network_candidate")
        features.append("atypical_network_candidate")

    # White regression candidate
    regression_condition = (
        (lesion_hsv[:, 2] >= _REGRESSION_V_MIN) &
        (lesion_hsv[:, 1] <= _REGRESSION_S_MAX)
    )
    if np.sum(regression_condition) / total_lesion_px >= _REGRESSION_COVERAGE_THRESH:
        features.append("regression_white_candidate")

    # Peripheral brighter rim (border vs. centre brightness)
    h, w = mask.shape
    eroded_mask = cv2.erode(mask, np.ones((max(1, h // 20), max(1, w // 20)),
                                          np.uint8))
    border_mask = cv2.subtract(mask, eroded_mask)
    v_channel = hsv[:, :, 2]
    border_vals = v_channel[border_mask > 0]
    centre_vals = v_channel[eroded_mask > 0]
    if len(border_vals) > 0 and len(centre_vals) > 0:
        if float(np.mean(border_vals)) > float(np.mean(centre_vals)) * 1.20:
            features.append("peripheral_lighter_rim_candidate")

    return features


# ---------------------------------------------------------------------------
# Step 7 — Heuristic suspicion score (NOT clinical)
# ---------------------------------------------------------------------------

def _compute_suspicion_score(
        asymmetry: float | None,
        irregularity: float | None,
        variegation: float | None,
        structural_features: list[str]) -> float | None:
    """
    Combine morphology metrics into a single heuristic suspicion score.

    This is a weighted sum of available metrics, each normalised to [0, 1].
    It is NOT a validated clinical risk score and MUST NOT be used for
    diagnosis.  It is labelled 'suspicion_score' to be transparent about
    its purpose as an evidence aggregation aid for A6.

    If fewer than 2 metrics are available, returns None to avoid giving
    false confidence.
    """
    available = 0
    score = 0.0

    if asymmetry is not None:
        score += _SUSPICION_WEIGHTS["asymmetry"] * asymmetry
        available += 1

    if irregularity is not None:
        score += _SUSPICION_WEIGHTS["border"] * irregularity
        available += 1

    if variegation is not None:
        score += _SUSPICION_WEIGHTS["colour"] * variegation
        available += 1

    # Structural features: each candidate adds a partial structural signal
    struct_signal = min(len(structural_features) / 3.0, 1.0)
    if structural_features:
        score += _SUSPICION_WEIGHTS["structures"] * struct_signal
        available += 1

    if available < 2:
        return None

    # Normalise by the weight of available metrics
    weight_used = (
        (_SUSPICION_WEIGHTS["asymmetry"] if asymmetry is not None else 0.0) +
        (_SUSPICION_WEIGHTS["border"] if irregularity is not None else 0.0) +
        (_SUSPICION_WEIGHTS["colour"] if variegation is not None else 0.0) +
        (_SUSPICION_WEIGHTS["structures"] if structural_features else 0.0)
    )
    if weight_used < 0.01:
        return None

    return float(round(np.clip(score / weight_used, 0.0, 1.0), 3))


# ---------------------------------------------------------------------------
# Step 8 — Overall reliability assessment
# ---------------------------------------------------------------------------

def _overall_reliability(seg_reliable: bool,
                          n_metrics_available: int,
                          limitations: list[str]) -> str:
    """
    Assign overall reliability tier.

    HIGH   : segmentation succeeded, all 3 quantitative metrics available
    MEDIUM : segmentation succeeded, 1–2 quantitative metrics available
    LOW    : segmentation failed or 0 quantitative metrics available
    """
    if not seg_reliable:
        return "LOW"
    if n_metrics_available >= 3:
        return "HIGH"
    if n_metrics_available >= 1:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_lesion(image: Any) -> dict:
    """
    Analyse a dermoscopy image for lesion morphology evidence.

    Parameters
    ----------
    image : str | pathlib.Path | np.ndarray
        File path to a JPEG/PNG image or an already-loaded NumPy array
        (BGR or RGB, 2-D or 3-D).

    Returns
    -------
    dict  — stable schema defined by SCHEMA_VERSION = "1.0"
        asymmetry            : shape asymmetry metric
        border_irregularity  : contour compactness-based irregularity
        colour_variegation   : hue coefficient of variation within lesion
        structural_features  : list of string feature candidates
        suspicion_score      : heuristic aggregated score [0,1] or null
        reliability          : "HIGH" | "MEDIUM" | "LOW"
        segmentation         : segmentation diagnostics
        limitations          : human-readable list of caveats

    This function NEVER raises an unhandled exception. All errors are
    captured and returned in the "limitations" list with reliability="LOW".
    """
    # --- Default failure result ---
    def _failure(reason: str) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "reliable": False,
            "error": reason,
            "features": [],
            "evidence": [reason],
            "flags": {
                "asymmetric": False,
                "irregular_border": False,
                "colour_variegated": False,
                "unreliable_morphology": True,
            },
            "asymmetry": {"value": None, "reliable": False, "method": "n/a"},
            "border_irregularity": {
                "value": None, "reliable": False,
                "method": "n/a", "compactness": None
            },
            "colour_variegation": {
                "value": None, "reliable": False, "method": "n/a",
                "details": {"mean_hue": None, "std_hue": None,
                            "n_colour_clusters": None}
            },
            "structural_features": [],
            "suspicion_score": None,
            "reliability": "LOW",
            "segmentation": {
                "method": "n/a",
                "mask_coverage": None,
                "contour_count": None,
                "success": False,
            },
            "limitations": [reason],
        }

    # --- Load ---
    try:
        bgr, load_error = _load_image(image)
    except Exception as exc:
        return _failure(f"Image loading raised exception: {exc}")

    if bgr is None:
        return _failure(f"Image could not be loaded: {load_error}")

    limitations: list[str] = []

    # --- Preprocess ---
    try:
        preprocessed = _preprocess(bgr)
    except Exception as exc:
        limitations.append(f"Preprocessing failed: {exc}")
        preprocessed = bgr  # fall back to raw image

    # --- Segment ---
    try:
        mask, seg_method, seg_diag = _segment_lesion(preprocessed)
    except Exception as exc:
        return _failure(f"Segmentation raised exception: {exc}")

    h, w = mask.shape
    coverage = float(np.count_nonzero(mask)) / (h * w)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    n_contours = len(contours)

    seg_result = {
        "method": seg_method,
        "mask_coverage": round(coverage, 4),
        "contour_count": n_contours,
        "success": False,  # updated below if reliable
    }

    seg_reliable, seg_limitations = _assess_mask_reliability(mask)
    limitations.extend(seg_limitations)

    if not seg_reliable:
        result = _failure("Segmentation unreliable — see limitations.")
        result["segmentation"] = seg_result
        result["limitations"] = limitations
        result["evidence"] = limitations
        result["error"] = limitations[0] if limitations else "Segmentation unreliable"
        return result

    seg_result["success"] = True

    # --- Asymmetry ---
    asym_value: float | None = None
    asym_reliable = False
    asym_method = "n/a"
    try:
        asym_value, asym_method = _compute_asymmetry(mask)
        asym_reliable = True
    except Exception as exc:
        limitations.append(f"Asymmetry calculation failed: {exc}")

    # --- Border irregularity ---
    irreg_value: float | None = None
    compactness: float | None = None
    irreg_reliable = False
    irreg_method = "n/a"
    try:
        irreg_value, compactness, irreg_method = _compute_border_irregularity(mask)
        irreg_reliable = irreg_value is not None
    except Exception as exc:
        limitations.append(f"Border irregularity calculation failed: {exc}")

    # --- Colour variegation ---
    var_value: float | None = None
    var_details: dict = {"mean_hue": None, "std_hue": None,
                          "n_colour_clusters": None}
    var_reliable = False
    var_method = "n/a"
    try:
        var_value, var_details, var_method = _compute_colour_variegation(
            preprocessed, mask)
        var_reliable = var_value is not None
    except Exception as exc:
        limitations.append(f"Colour variegation calculation failed: {exc}")

    # --- Structural features ---
    struct_features: list[str] = []
    try:
        struct_features = _extract_structural_features(preprocessed, mask)
    except Exception as exc:
        limitations.append(f"Structural feature extraction failed: {exc}")

    # --- Suspicion score ---
    suspicion: float | None = None
    try:
        suspicion = _compute_suspicion_score(
            asym_value, irreg_value, var_value, struct_features)
    except Exception as exc:
        limitations.append(f"Suspicion score calculation failed: {exc}")

    # --- Reliability ---
    n_available = sum([
        asym_reliable,
        irreg_reliable,
        var_reliable,
    ])
    reliability = _overall_reliability(seg_reliable, n_available, limitations)

    # --- Standard limitations note ---
    limitations.append(
        "Segmentation uses classical Otsu thresholding; accuracy varies with "
        "dermoscope background, hair occlusion, and colour cast."
    )
    limitations.append(
        "suspicion_score is a heuristic evidence aggregation aid only — "
        "NOT a validated clinical risk score or diagnosis."
    )

    # --- Contract aliases for A5 Safety & A6 Coordinator ---
    is_reliable = (reliability != "LOW")
    primary_error = limitations[0] if (not is_reliable and limitations) else None

    flags = {
        "asymmetric": bool(asym_value is not None and asym_value > 0.35),
        "irregular_border": bool(irreg_value is not None and irreg_value > 0.35),
        "colour_variegated": bool(var_value is not None and var_value > 0.35),
        "unreliable_morphology": not is_reliable,
    }
    for feat in struct_features:
        feat_clean = feat.replace("_candidate", "")
        flags[feat_clean] = True

    return {
        "schema_version": SCHEMA_VERSION,
        "reliable": is_reliable,
        "error": primary_error,
        "features": struct_features,
        "evidence": limitations,
        "flags": flags,
        "asymmetry": {
            "value": round(asym_value, 4) if asym_value is not None else None,
            "reliable": asym_reliable,
            "method": asym_method,
        },
        "border_irregularity": {
            "value": round(irreg_value, 4) if irreg_value is not None else None,
            "reliable": irreg_reliable,
            "method": irreg_method,
            "compactness": round(compactness, 4) if compactness is not None else None,
        },
        "colour_variegation": {
            "value": round(var_value, 4) if var_value is not None else None,
            "reliable": var_reliable,
            "method": var_method,
            "details": var_details,
        },
        "structural_features": struct_features,
        "suspicion_score": suspicion,
        "reliability": reliability,
        "segmentation": seg_result,
        "limitations": limitations,
    }


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

def _cli() -> None:
    import argparse, json, sys

    parser = argparse.ArgumentParser(
        description="A3 Lesion Morphology Agent — single-image smoke test")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--pretty", action="store_true",
                        help="Pretty-print JSON output")
    args = parser.parse_args()

    result = analyze_lesion(args.image)
    indent = 2 if args.pretty else None
    print(json.dumps(result, indent=indent))


if __name__ == "__main__":
    _cli()
