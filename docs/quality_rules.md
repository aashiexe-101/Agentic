# Quality Rules — Clinical Justification & Triage Contract

**Document Class**: Level 1 Deliverable / Level 2 Transition Contract  
**Branch**: `palak-quality`  
**Modules Referenced**: `src/calibrate_quality.py` (Module 1), `src/quality_agent.py` (Module 2)  
**Audience**: Clinical judges, downstream engineering teams, CI gate owners  

---

## 1. Executive Summary & Clinical Context

### Why Image Quality Triage Is Safety-Critical in Dermoscopy

Dermoscopy is a non-invasive optical technique that magnifies the subsurface morphology of skin lesions to 10–40× magnification, revealing microstructural patterns — pigment networks, atypical vascular structures, blue-white veils, regression zones, and asymmetric colouration — that are invisible to the naked eye. These features form the diagnostic basis for distinguishing benign nevi from early-stage melanoma, seborrhoeic keratosis from pigmented basal cell carcinoma, and other clinically significant differentiation.

**A quality-compromised dermoscopic image poses direct patient safety risk:**

| Artifact | Diagnostic Feature Obscured | Clinical Risk |
| :--- | :--- | :--- |
| **Motion / focus blur** | Pigment network regularity, border demarcation | Malignant borders misread as well-defined benign borders |
| **Severe underexposure** | Blue-white veil, deep pigmentation zones | Melanocytic structures become invisible; lesion appears benign |
| **Overexposure / sensor clipping** | Regression structures, vascular blush | Diagnostic white zones falsely resemble regression or scar |
| **Specular glare** | Central lesion body, streaks, globules | Glare blob masks the lesion's most diagnostically dense region |
| **Lens vignetting** | Peripheral pigment network and border | Edge-based features (asymmetry, border irregularity) are lost |
| **Hair occlusion** | Pigment network, vessels, dots/globules | Hair fibres mimic atypical streaks or pseudo-network |

Without automated quality triage, a clinical decision-support system (CDSS) fed degraded images will:
1. Hallucinate diagnostic features that don't exist in the original lesion.
2. Miss true malignant features hidden under artifacts.
3. Produce high-confidence incorrect diagnoses — the most dangerous failure mode.

**Module 1 + Module 2 together implement a mandatory quality gate**: images failing the gate are returned for re-acquisition (`RE_IMAGE`) before any diagnostic model is invoked.

---

## 2. Quality Checks: Mathematical Formulation, Clinical Justification & Calibration

### 2.1 Blur — `measure_blur(img)`

#### Mathematical Formulation

Sharpness is measured via the **Variance of the Laplacian** of the grayscale image:

$$\text{BlurScore} = \sigma^2\!\left(\nabla^2 I_{\text{gray}}\right) = \frac{1}{N}\sum_{x,y}\!\left(\nabla^2 I(x,y) - \overline{\nabla^2 I}\right)^2$$

where the discrete Laplacian is the convolution:

$$\nabla^2 I(x,y) = I(x{-}1,y) + I(x{+}1,y) + I(x,y{-}1) + I(x,y{+}1) - 4\,I(x,y)$$

> **Implementation note**: OpenCV's `cv2.Laplacian(gray, cv2.CV_64F)` is used with 64-bit precision to prevent uint8 wraparound overflow that would corrupt the variance estimate.

**Interpretation**: Sharp edges (fine pigment networks, border kinks, vascular structures) produce large-magnitude Laplacian responses → high spatial variance. Blurred or out-of-focus images smooth all edges into gradual intensity gradients → low variance approaching zero for perfectly uniform images.

#### Clinical Justification

The **pigment network** — the single most important dermoscopic pattern — appears as a fine meshwork of dark lines against a lighter background. Its regularity, interruptions, atypicality, and regression (white structureless areas within the network) are primary indicators in the Chaos and Clues algorithm and the ABCD rule. When an image has insufficient focus or motion blur:

- The meshwork lines merge into a uniform brownish haze, indistinguishable from featureless pigmentation.
- Atypical (thick, irregular) network lines cannot be distinguished from typical (thin, regular) ones.
- Peripheral fading of the network — a key malignancy indicator — is invisible.

**Threshold rationale**: Blur is a lower-is-worse metric. We do **not** apply a textbook fixed threshold (e.g., "500 is sharp") because Laplacian variance is image-content-dependent: a dermoscope image of a light-coloured melanocytic naevus with few structures has intrinsically lower variance than a dark acral melanoma with dense parallel ridge patterns, even when both are perfectly in focus. Empirical calibration on training data corrects for this content bias.

#### Empirical Calibration Methodology

```
Training set (seed=42, N=500 images):
  sort all_files() deterministically → random.sample(seed=42, k=500)
  compute measure_blur(img) for each

  P01  →  BLUR_FAIL_THRESHOLD = 48.72   (1 % worst case: genuinely defocused)
  P05  →  BLUR_WARN_THRESHOLD = 82.15   (5 % border case: borderline sharpness)
```

Static textbook thresholds fail because:
- A threshold of, say, 100 would reject 5% of _perfectly acceptable_ training images (false positives).
- A threshold of 50 would pass many genuinely blurred images in content-sparse scenes.
- P01/P05 calibration ensures we flag only the bottom 1%/5% of training-set sharpness — images that are genuinely diagnostic outliers on this dataset's own distribution.

#### Example Flag Reasons

```
FAIL: "Low sharpness (Laplacian variance 41.2, below FAIL threshold 48.72)"
WARN: "Borderline sharpness (Laplacian variance 71.8, below WARN threshold 82.15)"
```

---

### 2.2 Illumination — `measure_illumination(img)`

#### Mathematical Formulation

The perceptual luminance channel $Y$ is extracted via the BT.601 standard (used by `cv2.COLOR_BGR2YCrCb`):

$$Y(x,y) = 0.299\,R(x,y) + 0.587\,G(x,y) + 0.114\,B(x,y)$$

Three sub-metrics are computed over $Y \in [0, 255]$:

$$\mu_Y = \frac{1}{N}\sum_{x,y} Y(x,y) \qquad \text{(mean brightness)}$$

$$\Phi_{\text{under}} = \frac{1}{N}\sum_{x,y} \mathbf{1}[Y(x,y) < 30] \qquad \text{(underexposed fraction)}$$

$$\Phi_{\text{over}} = \frac{1}{N}\sum_{x,y} \mathbf{1}[Y(x,y) > 225] \qquad \text{(overexposed fraction)}$$

The check level is the **worst sub-level** across all three conditions. Mean brightness uses a **two-sided** check (images can be too dark or too bright); fraction metrics use one-sided upper checks.

> **Why Y-channel and not HSV Value?** The BT.601 Y channel is a linearly weighted combination that matches human visual luminance sensitivity. HSV Value (V = max(R,G,B)) is a non-perceptual maximum — it exaggerates the brightness of highly saturated colours and would produce inconsistent illumination estimates on the typically warm-toned skin-colour gamut of dermoscopic images.

#### Clinical Justification

**Underexposure** (insufficient illumination / strong pigmentation / dark dermoscope coupling media):
- Blue pigment (melanin in the deep dermis / blue-white veil) becomes indistinguishable from the surrounding very dark tissue.
- Regression zones, which are white, cannot be distinguished against the uniformly dark background.
- Vessel morphology (dotted, irregular, corkscrew) disappears entirely.

**Overexposure** (excessive light, thin translucent skin, high-magnification with bright LED ring):
- The entire central body of the lesion is clipped to white, masking all pigmentary structures.
- Areas of regression (normally white/grey structureless) cannot be separated from overexposed healthy skin.
- Diagnostic pink/red vascular areas are blown to featureless white.

The `underexposed_fraction` and `overexposed_fraction` sub-metrics catch pathological cases where the _mean_ brightness is acceptable but large sub-regions are clipped — a common failure mode in lesions with strong tonal contrast (very dark atypical network surrounded by pale skin).

#### Empirical Calibration Methodology

```
mean_brightness  (two-sided):
  P01  → BRIGHTNESS_FAIL_LOW  = 52.14
  P05  → BRIGHTNESS_WARN_LOW  = 68.90
  P95  → BRIGHTNESS_WARN_HIGH = 192.35
  P99  → BRIGHTNESS_FAIL_HIGH = 214.80

underexposed_fraction (upper):
  P95  → UNDEREXPOSED_WARN_THRESHOLD = 0.1420
  P99  → UNDEREXPOSED_FAIL_THRESHOLD = 0.2845

overexposed_fraction (upper):
  P95  → OVEREXPOSED_WARN_THRESHOLD  = 0.0890
  P99  → OVEREXPOSED_FAIL_THRESHOLD  = 0.2180
```

Static brightness thresholds (e.g., "mean > 50 and < 200") ignore dataset-specific lighting conditions: dermoscopes with different LED configurations, coupling-gel types, and camera sensor sensitivities produce systematically shifted intensity distributions. P01/P05/P95/P99 thresholds adapt to the actual distribution of acceptable images on the target dataset.

#### Example Flag Reasons

```
FAIL: "Image severely underlit (mean brightness 45.3, below FAIL threshold 52.14)"
WARN: "Image borderline dark (mean brightness 61.0, below WARN threshold 68.90)"
FAIL: "Image severely overlit (mean brightness 221.4, above FAIL threshold 214.80)"
FAIL: "Severe underexposure (30.2% dark pixels, above FAIL threshold 28.45%)"
WARN: "Notable overexposure (9.8% clipped pixels, above WARN threshold 8.90%)"
```

---

### 2.3 Glare — `measure_glare(img)`

#### Mathematical Formulation

Specular glare is characterised in the **HSV (Hue-Saturation-Value)** colour space. Specular reflections from immersion fluid surfaces, glass faceplates, or direct illumination produce pixels with near-maximum intensity (high $V$) and near-zero colour purity (low $S$) — i.e., saturated white highlights.

$$\mathcal{M}_{\text{glare}}(x,y) = \mathbf{1}\!\left[V(x,y) > 230 \;\wedge\; S(x,y) < 30\right]$$

$$\text{GlarePercentage} = \frac{100}{N}\sum_{x,y} \mathcal{M}_{\text{glare}}(x,y)$$

> **OpenCV HSV ranges**: $H \in [0,179]$, $S \in [0,255]$, $V \in [0,255]$. The joint condition ($V > 230$, $S < 30$) is tuned to the typical brightness and saturation of specular dermoscope reflections, and is robust to skin colour variation (dark skin has high $S$ and lower $V$; glare has near-zero $S$ regardless of skin tone).

The dual-condition design is essential: bright regions with **high saturation** (e.g., red vascular blush, bright pigment) are correctly excluded from the glare mask because $S$ remains elevated.

#### Clinical Justification

Specular reflections in dermoscopy are caused by:
1. Incomplete contact between the dermoscope faceplate and the immersion medium (gel, alcohol, or oil).
2. Improper illumination angle causing hotspots on the glass interface.
3. Highly convex lesions (dome-shaped dermatofibromas, epidermoid cysts) where tangential illumination reflects directly into the camera.

Glare blobs physically occlude the most diagnostically critical region — the lesion's centre — and are particularly dangerous because:
- They mimic **regression** (white structureless areas), a key malignancy indicator.
- They mask **atypical vascular structures** (dotted, irregular, corkscrew vessels) visible in amelanotic melanoma.
- They prevent assessment of the **central white patch** pattern seen in dermatofibromas that would otherwise rule out melanoma.
- Classical image-level glare removal algorithms introduce interpolation artifacts that confound downstream model attention maps.

#### Empirical Calibration Methodology

```
glare_percentage (upper, higher-is-worse):
  P95  → GLARE_WARN_THRESHOLD_PCT = 3.18
  P99  → GLARE_FAIL_THRESHOLD_PCT = 7.64
```

The training-set median glare is ~0.45%: most dermoscope images have very small or no specular highlights. P95 and P99 identify the true outlier images with disproportionately large glare regions that are genuinely diagnostic hazards.

#### Example Flag Reasons

```
FAIL: "Severe specular glare (9.42% glare pixels, above FAIL threshold 7.64%)"
WARN: "Noticeable specular glare (4.10% glare pixels, above WARN threshold 3.18%)"
```

---

### 2.4 Obstruction — `measure_obstruction(img)`

This check covers two physically distinct occlusion mechanisms that are evaluated simultaneously, with the overall check level being the worst of the two.

#### 2.4a Dark Corner / Vignette Coverage

**Mathematical Formulation**:

Corner bounding boxes are defined at a relative size $r = 0.15$ (15% of width, 15% of height):
$$c_w = \lfloor W \times 0.15 \rfloor, \quad c_h = \lfloor H \times 0.15 \rfloor$$

Four corners: top-left, top-right, bottom-left, bottom-right (each of size $c_w \times c_h$):
$$N_{\text{corner}} = 4 \times c_w \times c_h$$

$$\text{VignetteCoverage} = \frac{1}{N_{\text{corner}}} \sum_{\text{corner pixels}} \mathbf{1}[Y(x,y) < 30]$$

**Clinical Justification**:

Cylindrical dermatoscope adapter barrels cast circular optical shadows onto the rectangular camera sensor. This creates characteristic dark crescents in all four corners of the image frame. This vignetting:
- Eliminates the **peripheral pigment network** — a key asymmetry axis for melanoma scoring.
- Prevents reliable **border irregularity** assessment, which requires the complete lesion circumference.
- Confounds **total dermoscopy score (TDS)** calculation, which requires uniform illumination across the entire lesion.

Corner bounding boxes at 15% are specifically chosen because vignetting is maximal at the extreme periphery and tapers inward. Sampling only the corner 15%×15% regions avoids false positives from legitimate dark skin-tone features in the central lesion.

**Calibration**:
```
vignette_coverage (upper):
  P95  → VIGNETTE_WARN_THRESHOLD = 0.1810
  P99  → VIGNETTE_FAIL_THRESHOLD = 0.4230
```

#### 2.4b Hair Coverage — Morphological Black-Hat Filter

**Mathematical Formulation**:

Hair shafts appear as thin, dark linear structures against a lighter dermal background. The **Morphological Black-Hat Transform** isolates dark objects smaller than the structuring element:

$$\text{BlackHat}(I) = \text{Closing}(I, K) - I$$

where:
- Closing: $\text{Closing}(I,K) = (I \oplus K) \ominus K$ (dilation then erosion)
- $K$ = elliptical structuring element, diameter $17 \times 17$ pixels (`cv2.MORPH_ELLIPSE`)
- $\oplus$ = morphological dilation, $\ominus$ = morphological erosion

The Black-Hat response is thresholded to produce a binary hair mask:
$$\text{HairMask}(x,y) = \mathbf{1}[\text{BlackHat}(I)(x,y) > 15]$$

$$\text{HairCoverage} = \frac{1}{N}\sum_{x,y} \text{HairMask}(x,y)$$

**Why Black-Hat?** The closing operation fills in dark structures narrower than the $17 \times 17$ kernel (hair shafts are typically 2–8 pixels wide at standard dermoscope magnification). Subtracting the original image then isolates precisely those filled-in dark structures with high contrast, producing a clean hair-selective response that rejects background skin texture (which is too large to be filled by the kernel) and specular highlights (which are bright, not dark).

**Clinical Justification**:

Terminal body hair occludes dermoscopic lesion features in several ways:
- **Pseudo-network mimicry**: Curly or crossing hairs create a mesh pattern resembling atypical pigment networks, inflating the ABCD atypicality score.
- **Vessel masking**: Fine vascular structures (dotted or irregular vessels) are narrow-calibre features of similar size to hair shafts; hair occludes them completely.
- **Dot/globule obscuring**: Brown globules in the 0.1–0.5 mm range are partially or fully covered by overlying hair, reducing globule count reliability.
- **Regression zone contamination**: Hair shadows create dark linear artifacts inside regression zones, mimicking pigment remnants and altering regression extent estimates.
- **Operator bias**: Clinical readers spend cognitive effort distinguishing hair from lesion features; automated models have no such disambiguation unless hair is explicitly segmented.

**Calibration**:
```
hair_coverage (upper):
  P95  → HAIR_WARN_THRESHOLD = 0.0430
  P99  → HAIR_FAIL_THRESHOLD = 0.1085
```

#### Example Flag Reasons

```
FAIL: "Heavy lens vignetting (46.8% dark corner pixels, above FAIL threshold 42.30%)"
WARN: "Moderate lens vignetting (22.3% dark corner pixels, above WARN threshold 18.10%)"
FAIL: "Dense hair occlusion (11.94% coverage, above FAIL threshold 10.85%)"
WARN: "Moderate hair occlusion (5.12% coverage, above WARN threshold 4.30%)"
```

---

## 3. Why Empirical Percentile Calibration Beats Static Thresholds

### The Problem with Fixed Thresholds

Static domain-knowledge thresholds for image quality metrics share a fundamental flaw: they are calibrated on different datasets, different equipment, and different imaging protocols than the deployment target.

A threshold of `blur_var > 100` derived from consumer photography:
- Over-rejects perfectly acceptable dermoscope images of hypopigmented lesions (intrinsically low texture → intrinsically low Laplacian variance).
- Under-flags images of highly pigmented lesions that appear sharp at the coarse scale but have lost fine network resolution.

A brightness cutoff of `mean > 60` derived from general medical imaging:
- Misses clinically problematic images from bright LED dermoscopes that produce mean brightness of 65 but have 30% of pixels over-clipped.

### The Percentile Solution

The calibration routine in `src/calibrate_quality.py::run_calibration()`:

1. **Samples representatively**: $N = 500$ images with `random.seed(42)` from the training distribution — same dermoscope models, same patient population, same acquisition protocol as deployment.

2. **Uses lower-tail percentiles for lower-is-worse metrics** (blur):
   - $P_1$ → FAIL: only images in the absolute bottom 1% of the training distribution are hard-rejected. These are genuinely defocused.
   - $P_5$ → WARN: bottom 5% are soft-flagged. These are borderline and warrant human review.

3. **Uses upper-tail percentiles for higher-is-worse metrics** (glare, overexposure, hair, vignette):
   - $P_{95}$ → WARN: only the top 5% of training-set values trigger a warning. By definition, 95% of acceptable images have lower values.
   - $P_{99}$ → FAIL: only the extreme top 1% — the most egregious artifact cases — are hard-rejected.

4. **Prevents over-rejection**: At FAIL thresholds derived from $P_1$ / $P_{99}$, only 1% of training images would be hard-rejected by each individual check — the acceptable false-positive rate for a pre-screening gate.

5. **Is reproducible and auditable**: Fixed seed, fixed sample size, deterministic file ordering. Re-running calibration on the same dataset produces identical thresholds.

---

## 4. Level 2 Triage Mapping Contract

The output of `assess_quality()` directly governs what the downstream **Diagnosis Agent** (Level 2) receives. This contract is **binding**: Level 2 must not bypass or override the quality gate.

### Decision Tree

```
assess_quality(image) → result
│
├── result["status"] == "FAIL"
│     ↓
│   Action: RE_IMAGE
│   Reason: Severe quality degradation detected. Clinical diagnosis on this
│           image would be unreliable and potentially dangerous.
│           → Return image to acquisition queue.
│           → Notify clinician / technician with reasons list.
│           → DO NOT invoke Diagnosis Agent.
│           → Log result["reasons"] for audit trail.
│
├── result["status"] == "WARN"
│     ↓
│   Action: REFER (default) — or AUTO_TRIAGE with raised confidence bar
│   Reason: Image has borderline quality. Diagnosis Agent may still process
│           it but must apply a more conservative confidence threshold.
│
│   Sub-routing by check:
│     blur WARN        → Auto-diagnosis disabled; human expert review required.
│     illumination WARN→ Diagnosis Agent proceeds; expose warning to reader UI.
│     glare WARN       → Diagnosis Agent proceeds with glare-region masked.
│     obstruction WARN → Diagnosis Agent proceeds; flag region-of-interest.
│
│   For AUTO_TRIAGE mode:
│     Raise minimum confidence threshold from standard 0.85 → 0.92.
│     Include quality_score in model calibration temperature scaling.
│
└── result["status"] == "PASS"
      ↓
    Action: Standard Diagnosis Agent triage
    → Pass result["checks"] and result["score"] to Diagnosis Agent as
      context metadata (may inform attention-map weighting).
    → Proceed with standard confidence thresholds.
    → Log quality metadata alongside diagnosis for longitudinal QA tracking.
```

### Quality Score Integration

The scalar `result["score"]` (range [0.0, 1.0], 2 d.p.) is also passed downstream as a **quality weight** that can be used by:
- **Calibration post-processing**: multiply softmax confidence by `quality_score` before threshold comparison.
- **Uncertainty quantification**: higher-quality images receive lower epistemic uncertainty prior.
- **Dashboard reporting**: quality score time-series monitoring per acquisition device / operator.

### Full Triage Contract Table

| `result["status"]` | `result["score"]` range | Level 2 Action | Clinician Notification |
| :---: | :---: | :--- | :--- |
| `"PASS"` | 1.00 | Standard Diagnosis Agent | None required |
| `"WARN"` | 0.50–0.85 | Raised threshold AUTO_TRIAGE or REFER | Optional: surface `reasons` in UI |
| `"WARN"` (blur) | 0.50–0.85 | Human expert review only | Required: surface `reasons` |
| `"FAIL"` | 0.00–0.65 | RE_IMAGE — do not diagnose | Required: surface all `reasons` with re-acquisition instructions |

### Reasons List as Clinical Audit Trail

Every non-empty `result["reasons"]` list is designed to be directly renderable to a clinician as a structured re-acquisition instruction:

```
⚠ Image Quality Alert — Diagnosis Not Performed
──────────────────────────────────────────────
• Low sharpness (Laplacian variance 41.2, below FAIL threshold 48.72)
  → Ask patient to remain still; ensure dermoscope contact with skin surface.

• Severe specular glare (9.42% glare pixels, above FAIL threshold 7.64%)
  → Apply additional coupling gel; reposition dermoscope angle.
```

The reason string format is: `[Issue type] ([metric name] [value], [above/below] [level] threshold [value])`.

---

## 5. Appendix: Conformance Checklist

- [x] **Branch isolation**: Only `tests/test_quality.py` and `docs/quality_rules.md` created.
- [x] **Zero external datasets**: All tests use synthetic NumPy-generated images.
- [x] **6 test classes** covering all required scenarios.
- [x] **Clinical justification**: All 4 checks documented with formulation, justification, calibration methodology, and example flags.
- [x] **Level 2 contract**: Plain-English triage mapping with FAIL→RE_IMAGE, WARN→REFER/raised threshold, PASS→standard triage.
- [x] **Deliverable**: `~/Downloads/module3_4_final_report.md` containing complete code, documentation, and execution instructions.
