# Agent A3: Lesion Morphology Analysis — Technical Reference
**File:** `docs/lesion_analysis_notes.md`  
**Author:** Palak (A1 Quality / A3 Lesion owner)  
**Branch:** `feature/palak-lesion-analysis`  
**Last updated:** 2026-10-30

---

## 1. Purpose and Scope

Agent A3 (`src/lesion_agent.py`) extracts classical image-analysis evidence
from dermoscopy images to support Agent A6's coordinator decision.

It does **not** produce clinical diagnoses, clinical risk scores, or validated
dermatological assessments.  All output values are labelled with their
limitations and confidence levels.

---

## 2. Public API

```python
from src.lesion_agent import analyze_lesion

result = analyze_lesion(image)
# image: str | pathlib.Path | np.ndarray (BGR or RGB)
```

### Output Schema (v1.1 — Contract-Hardened & Backward-Compatible)

```python
{
    "schema_version": "1.0",          # stable schema version
    "reliable": bool,                 # Top-level boolean: True if reliability != "LOW", False if "LOW"
    "error": str | None,              # Primary failure reason when unreliable, else None (triggers A5 veto)
    "features": [str],                # Alias pointing to structural_features (Phase 2 contract)
    "evidence": [str],                # Alias pointing to limitations (Phase 2 contract)
    "flags": {                        # Discrete boolean flags for coordinator/auditor
        "asymmetric": bool,
        "irregular_border": bool,
        "colour_variegated": bool,
        "unreliable_morphology": bool,
        "blue_white_veil": bool,
        "dark_network": bool,
        "atypical_network": bool,
        ...
    },
    "asymmetry": {
        "value":    float | None,     # 0=symmetric, 1=maximum asymmetry
        "reliable": bool,
        "method":   str
    },
    "border_irregularity": {
        "value":      float | None,   # 0=circular, 1=maximally irregular
        "reliable":   bool,
        "method":     str,
        "compactness": float | None   # 4π·A/P² ; value + compactness ≈ 1.0
    },
    "colour_variegation": {
        "value":    float | None,     # hue CoV normalised to [0,1]
        "reliable": bool,
        "method":   str,
        "details": {
            "mean_hue":         float | None,
            "std_hue":          float | None,
            "n_colour_clusters": int | None
        }
    },
    "structural_features": [str],     # candidate findings (e.g. "blue_white_veil_candidate")
    "suspicion_score":  float | None, # heuristic [0,1] — NOT clinical
    "reliability": "HIGH"|"MEDIUM"|"LOW",
    "segmentation": {
        "method":         str,
        "mask_coverage":  float | None,
        "contour_count":  int | None,
        "success":        bool
    },
    "limitations": [str]
}
```

---

## 3. Pipeline Steps

### Step 1 — Input Loading
Accepts file path (str/Path) or NumPy array (BGR/RGB/grayscale/BGRA).
Converts all inputs to BGR.  Graceful failure on any load error.

### Step 2 — Preprocessing
Bilateral filter (d=9, σ_colour=75, σ_space=75) to reduce JPEG artefacts
and hair noise while preserving lesion edges.

### Step 3 — Segmentation
1. Grayscale + CLAHE (clipLimit=2, tileGridSize=8×8) for local contrast.
2. Otsu global threshold.
3. Invert if foreground fraction > 50% (lesion typically darker than skin).
4. Morphological closing (ellipse kernel ≈ 4% of image min-dimension).
5. Morphological opening (ellipse kernel ≈ 2%) to remove noise.
6. Keep largest connected component.

**Reliability gates:**
- Mask coverage < 3% → UNRELIABLE (lesion too small / not detected)
- Mask coverage > 92% → UNRELIABLE (background leak)
- No contour ≥ 500 px → UNRELIABLE

### Step 4 — Asymmetry
1. Compute centroid and principal axis angle via image moments.
2. Rotate mask to align principal axis to horizontal.
3. XOR with horizontal flip → horizontal asymmetry fraction.
4. XOR with vertical flip → vertical asymmetry fraction.
5. Final value = mean(horizontal, vertical) ∈ [0, 1].

### Step 5 — Border Irregularity
Isoperimetric quotient:
```
compactness = 4π · Area / Perimeter²
irregularity = 1 − compactness
```
Circle → compactness ≈ 1.0, irregularity ≈ 0.0.
Irregular shape → compactness < 0.5, irregularity > 0.5.

### Step 6 — Colour Variegation
1. Extract HSV pixels within lesion mask.
2. Coefficient of variation of hue = std(H) / 90.0 (clipped to [0,1]).
3. k-means (k=5, 20 iterations) on BGR lesion pixels → count clusters
   with > 1% membership.

### Step 7 — Structural Feature Candidates
All features are *candidates only* (suffixed `_candidate`):

| Feature | Condition |
|---|---|
| `blue_white_veil_candidate` | ≥6% lesion pixels with H∈[90,130], S<80, V≥180 |
| `dark_network_candidate` | ≥8% lesion pixels with V≤80 |
| `regression_white_candidate` | ≥5% lesion pixels with V≥200, S<30 |
| `peripheral_lighter_rim_candidate` | Border mean V > 120% of centre mean V |

### Step 8 — Heuristic Suspicion Score
Weighted sum of available metrics (NOT clinical):
```
score = 0.30·asymmetry + 0.25·irregularity + 0.25·variegation + 0.20·struct
```
Normalised by weight of available metrics.  Returns `null` if < 2 metrics
are available.

---

## 4. A1 Quality Agent Integration

`analyze_lesion` should only be called on images that pass A1's quality check.

Recommended pattern in `src/run_validation.py`:

```python
from src.quality_agent import assess_quality
from src.lesion_agent import analyze_lesion

q = assess_quality(image_path)
if q["status"] == "FAIL":
    lesion_result = None   # skip morphology on unreadable images
else:
    lesion_result = analyze_lesion(image_path)
```

---

## 5. Downstream Agent Integration Contracts (A5 Safety & A6 Coordinator)

Agent A3 output is consumed directly by Agent A5 (`src/safety_agent.py`) and Agent A6 (`src/coordinator.py`).

### 5.1 Agent A5 Safety Auditor Integration
In `src/safety_agent.py`, the clinical safety auditor inspects A3 via:
```python
if a3.get("reliable") is False or a3.get("error"):
    safety_flag = True
    veto_auto_triage = True
    requires_human_review = True
```
- **Top-level `reliable: bool`:** When segmentation fails or an unreadable input is provided, A3 sets `"reliable": False`, ensuring A5's safety veto activates without exception.
- **Top-level `error: str | None`:** Provides the primary segmentation or input error message (e.g. `"Segmentation unreliable — see limitations."`).
- **High-Risk Feature Detection:** A5 inspects `structural_features` scanning for substrings `["blue_white", "veil", "atypical", "streak", "ulcer"]`. A3 outputs `"blue_white_veil_candidate"` and `"atypical_network_candidate"` which reliably trigger A5's red-flag and cross-agent conflict checks against benign A2 predictions.

### 5.2 Agent A6 Coordinator Contract
In `src/coordinator.py`, the coordinator collects evidence for reporting and audit logs:
- **`flags`:** Returns a discrete boolean map:
  - `asymmetric: bool` (asymmetry > 0.35)
  - `irregular_border: bool` (irregularity > 0.35)
  - `colour_variegated: bool` (variegation > 0.35)
  - `unreliable_morphology: bool` (reliability == "LOW")
  - `blue_white_veil`, `atypical_network`, `dark_network`, `regression_white`: `bool`
- **`features` / `evidence`:** Aliases for `structural_features` and `limitations` complying with the Phase 2 Section 6 contract.

**Policy Rule:** If `reliability == "LOW"`, A6 and A5 strictly veto `AUTO-TRIAGE` and route to `REFER` (or `RE-IMAGE` if A1 failed).

### 5.3 Clinical Safety Disclaimer
> ⚠️ **CRITICAL CLINICAL DISCLAIMER:**
> `suspicion_score` and all morphological indicators are **experimental heuristic aids** derived from classical computer vision operators. They are **NOT** a clinically validated risk score, NOT an official dermatological ABCD score, and MUST NEVER be used as a standalone diagnostic determinant. All findings are labeled as candidates.

---

## 6. A1 Quality Agent Audit (Task 2)

### Training Data Status

The HAM10000 training dataset (`data/train/`) has **not been mounted** on
this machine.  A full 7,762-image quality audit cannot be executed in this
environment.

**What is verified:**
- A1 thresholds were derived from a simulated calibration description
  (500-image sample, seed=42) — **not from actual HAM10000 images**.
- The `calibrate_quality.py` module contains the full calibration pipeline
  and will produce real thresholds when run with the actual dataset mounted.

**To execute the real audit (Ilka's environment or CI):**

```bash
# Step 1 — Recalibrate on actual training images
python src/calibrate_quality.py \
    --data-dir data/train \
    --sample-size 500 \
    --seed 42 \
    --output-file thresholds_calibrated.py

# Step 2 — Run quality audit over all 7,762 training images
python src/summarize_quality.py \
    --input data/train/metadata.csv \
    --image-dir data/train/images \
    --output outputs/training_quality_audit.md

# Step 3 — Compare flagged rate against 15% anomaly gate
# If flagged rate > 15%: document evidence, propose threshold revision
# Do NOT silently recalibrate using evaluation images
```

**Threshold revision policy:**
- If real-data calibration produces thresholds materially different from
  the current documented values, open a PR with the evidence table before
  merging any threshold change into `quality_agent.py`.
- Changes require approval from Ilka (diagnosis impact) and Utsavi
  (evaluation contract compatibility).

---

## 7. Known Limitations

1. **Otsu segmentation** assumes the lesion is darker than surrounding skin.
   Amelanotic (light-coloured) lesions or images with heavy glare may produce
   inverted or leaked masks.

2. **Hair occlusion** can break the contour and cause asymmetry over-estimation.
   A1's hair obstruction check (`hair_coverage`) should be consulted before
   trusting A3 border or asymmetry values.

3. **Image size assumption** — designed and tested on 600×450 px HAM10000 images.
   Very small images (< 200×200) may produce degenerate contours.

4. **`suspicion_score` is NOT clinical.** It is a heuristic evidence aggregation
   aid. The Agentic DermaGuard system is for hackathon benchmarking only.

5. **No external model weights** — all steps use classical OpenCV/NumPy only.

---

## 8. Integration Instructions

### For Aashi (frontend)

Recommended display fields:
```python
result = analyze_lesion(image)
display = {
    "Reliability": result["reliability"],
    "Asymmetry":   f"{result['asymmetry']['value']:.2f}" if result['asymmetry']['value'] else "N/A",
    "Border":      f"{result['border_irregularity']['value']:.2f}" if result['border_irregularity']['value'] else "N/A",
    "Colour":      f"{result['colour_variegation']['value']:.2f}" if result['colour_variegation']['value'] else "N/A",
    "Features":    ", ".join(result["structural_features"]) or "None detected",
    "Note":        "Heuristic only — not a clinical assessment"
}
```

Always display `reliability` prominently. If `reliability == "LOW"`, show
a clear warning: *"Lesion analysis unreliable — see image quality check."*

### For Utsavi (A6 coordinator / evaluation)

The A3 output dict should be stored alongside A1 (`assess_quality`) and
A2 (`predict`) outputs in `val_results.csv`.  Suggested additional columns:

```
a3_reliability, a3_asymmetry, a3_border, a3_colour,
a3_suspicion_score, a3_features
```

Schema version (`schema_version`) should be logged to allow future audits
to detect breaking changes.
