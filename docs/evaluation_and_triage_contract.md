# Agentic DermaGuard — Evaluation & Clinical Triage Contract

**Document Class:** System Contract & Clinical Architecture  
**Author:** Utsavi Sinha (Senior Developer — Evaluation & Clinical Safety Track)  
**Policy Version:** `v2.1.0-clinical-safety`  
**Applicability:** Level 1 Evaluation, Level 2 Multi-Agent Triage, Final Portal Submission  

---

## 1. Executive Summary

This document defines the canonical data contracts, clinical triage policies, and evaluation standards for **Agentic DermaGuard**. It establishes strict separation between:
1. **Internal Validation 18-Column CSV Contract:** Used for offline model validation, error analysis, quality stratification, and clinical benchmark scoring.
2. **Official Final Submission 3-Column CSV Contract:** Required for final blind test evaluation on the hackathon portal.
3. **Synthetic Test Fixtures vs Genuine Performance:** Guarantees that synthetic fixtures (`outputs/fake_val_results.csv`) are never represented as real model metrics.

---

## 2. Canonical Internal Validation Contract (18 Columns)

### Exact Column Order & Schema Specification

| Index | Column Name | Type | Permitted Values / Range | Description |
|:---:|:---|:---:|:---:|:---|
| 1 | `image_id` | `str` | Non-empty string (e.g. `ISIC_0024306`) | Unique image identifier matching dataset |
| 2 | `true_label` | `str` | `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc` | Ground-truth lesion category (HAM10000) |
| 3 | `pred_class` | `str` | `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc` | Diagnostic model prediction ($\operatorname{argmax}$) |
| 4 | `confidence` | `float` | $[0.0, 1.0]$ | Model confidence, strictly equals $\max(\text{prob}_*)$ |
| 5 | `prob_akiec` | `float` | $[0.0, 1.0]$ | Softmax probability for Actinic Keratosis |
| 6 | `prob_bcc` | `float` | $[0.0, 1.0]$ | Softmax probability for Basal Cell Carcinoma |
| 7 | `prob_bkl` | `float` | $[0.0, 1.0]$ | Softmax probability for Benign Keratosis-like |
| 8 | `prob_df` | `float` | $[0.0, 1.0]$ | Softmax probability for Dermatofibroma |
| 9 | `prob_mel` | `float` | $[0.0, 1.0]$ | Softmax probability for Melanoma |
| 10 | `prob_nv` | `float` | $[0.0, 1.0]$ | Softmax probability for Melanocytic Nevus |
| 11 | `prob_vasc` | `float` | $[0.0, 1.0]$ | Softmax probability for Vascular Lesion |
| 12 | `q_status` | `str` | `PASS`, `WARN`, `FAIL` | Agent A1 image quality triage status |
| 13 | `q_score` | `float` | $[0.0, 1.0]$ | Aggregated image quality score |
| 14 | `q_reasons` | `str` | String or empty / NaN | Human-readable quality defect flags |
| 15 | `q_blur` | `float` | $[0.0, 1.0]$ | Normalized sharpness metric |
| 16 | `q_illumination`| `float`| $[0.0, 1.0]$ | Normalized illumination metric |
| 17 | `q_glare` | `float` | $[0.0, 1.0]$ | Normalized specular reflection metric |
| 18 | `q_obstruction` | `float`| $[0.0, 1.0]$ | Normalized hair/vignette obstruction metric |

### Validation & Integrity Rules (`validate_results_dataframe`)
1. **Column Order & Count:** The CSV must contain exactly these 18 columns in this precise sequence. Any missing or unexpected extra column raises a `ValueError`.
2. **Probability Sum Normalization:** $\sum_{c=1}^{7} \text{prob}_c = 1.0 \pm 10^{-3}$. Any violation triggers a hard rejection.
3. **Argmax Consistency:** $\text{pred\_class}$ must strictly match $\operatorname{argmax}_{c} (\text{prob}_c)$.
4. **Missing Values:** No nulls, NaNs, or empty strings are permitted in columns 1–13 or 15–18. Column 14 (`q_reasons`) is permitted to be empty or NaN for images with `q_status == 'PASS'`.
5. **Class Taxonomy:** The 7 canonical classes are immutable:
   - `akiec`: Actinic keratoses / Bowen's disease (malignant/pre-malignant)
   - `bcc`: Basal cell carcinoma (malignant)
   - `bkl`: Benign keratosis-like lesions (solar lentigines, seborrheic keratoses)
   - `df`: Dermatofibroma (benign)
   - `mel`: Melanoma (malignant — primary safety concern)
   - `nv`: Melanocytic nevi (benign — majority class)
   - `vasc`: Vascular lesions (angiomas, pyogenic granulomas)

---

## 3. Official Hidden-Test Submission Contract (3 Columns)

For the final submission portal, the internal 18-column fields must **never** be included. The submission CSV must strictly follow:

```csv
image_id,predicted_label,decision
ISIC_0000001,nv,AUTO-TRIAGE
ISIC_0000002,mel,REFER
ISIC_0000003,bcc,RE-IMAGE
```

### Strict Requirements (`validate_submission_csv`)
- **Columns:** Exactly `["image_id", "predicted_label", "decision"]`.
- **Allowed Decisions:** `AUTO-TRIAGE`, `RE-IMAGE`, `REFER`.
- **Allowed Labels:** Exactly one of the 7 canonical classes.
- **Uniqueness:** Zero duplicate `image_id` entries. Exactly one row per image.
- **Completeness:** Zero missing rows (e.g. exactly 200 rows when evaluated against 200 hidden images).

---

## 4. Multi-Agent Triage Policy Architecture

```
                    ┌──────────────────────────┐
                    │ Raw Dermoscopic Image    │
                    └────────────┬─────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
       ┌───────────────────┐           ┌───────────────────┐
       │ Agent A1: Quality │           │ Agent A2: Model   │
       │ PASS / WARN / FAIL│           │ 7 Probabilities   │
       └─────────┬─────────┘           └─────────┬─────────┘
                 │                               │
                 │   ┌───────────────────────┐   │
                 │   │ Agent A3: Morphology  │   │
                 │   │ Ulceration/Veil/Asym  │   │
                 │   └───────────┬───────────┘   │
                 │               │               │
                 │   ┌───────────┴───────────┐   │
                 │   │ Agent A4: Context     │   │
                 │   │ History / Age / Local │   │
                 │   └───────────┬───────────┘   │
                 │               │               │
                 └───────► ┌─────┴─────┐ ◄───────┘
                           │ Agent A5  │
                           │ Safety    │
                           │ Auditor   │
                           └─────┬─────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │ Agent A6: Coordinator │
                     │ Clinical Triage Rule  │
                     └───────────┬───────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
    [ RE-IMAGE ]            [   REFER   ]         [ AUTO-TRIAGE ]
 (A1 Quality FAIL)    (Malignancy, A5 Veto,    (High Confidence Benign,
                        Borderline, Conflicts)   A1 PASS, Zero Flags)
```

### Hierarchy of Decision Rules

1. **Rule 1 — Optical Degradation (`RE-IMAGE`):**
   - If Agent A1 reports `overall_status == "FAIL"`:
   - Output: `RE-IMAGE`, `requires_review = True`.
   - Justification: Severe blur, specular glare, or underexposure obscures microstructural pigment networks. Clinical interpretation without re-imaging creates unacceptable diagnostic risk.

2. **Rule 2 — Malignant / Pre-malignant Suspicion (`REFER`):**
   - If Agent A2 predicts `mel`, `bcc`, or `akiec`:
   - Output: `REFER`, `requires_review = True`.
   - Justification: Any suspected malignancy requires specialist dermatological clinical examination and biopsy consideration.

3. **Rule 3 — Safety Auditor Veto (`REFER`):**
   - If Agent A5 issues `veto_auto_triage == True`:
   - Output: `REFER`, `requires_review = True`.
   - Triggers:
     - Elevated secondary melanoma probability ($\ge 0.20$).
     - Diagnostic uncertainty (confidence $< 0.60$).
     - High-risk morphology detected by A3 (ulceration, bleeding, blue-white veil, atypical pigment network).
     - Cross-agent conflict (model predicts benign `nv`, but A3 observes ulceration).
     - Patient risk factors (personal or family history of melanoma, immunocompromised).

4. **Rule 4 — Borderline Image Quality (`REFER`):**
   - If Agent A1 reports `overall_status == "WARN"`:
   - Output: `REFER`, `requires_review = True`.
   - Justification: Borderline focus or lighting requires human validation before clinical clearance.

5. **Rule 5 — Safe Low-Risk Auto-Triage (`AUTO-TRIAGE`):**
   - Condition: Can **ONLY** be assigned if ALL of the following are true:
     1. Agent A1 status is `PASS`.
     2. Agent A2 predicts benign (`nv` or `bkl`) with high confidence.
     3. Agent A5 reports `veto_auto_triage == False` and `safety_flag == False`.
     4. Zero high-risk morphology flags (no ulceration, no blue-white veil).
     5. No personal/family history of melanoma.
   - Output: `AUTO-TRIAGE`, `requires_review = False`.

---

## 5. Metric Calculations & Clinical Prioritization

- **Overall Accuracy:** $\frac{\sum_{i=1}^N \mathbb{I}(\hat{y}_i = y_i)}{N}$
- **Balanced Multiclass Accuracy:** $\frac{1}{K} \sum_{c=1}^K \frac{\text{TP}_c}{\text{TP}_c + \text{FN}_c}$ (critical because `nv` represents 67% of cases).
- **Macro F1:** $\frac{1}{K} \sum_{c=1}^K \text{F1}_c$
- **Melanoma Sensitivity (Recall):** $\frac{\text{TP}_{\text{mel}}}{\text{TP}_{\text{mel}} + \text{FN}_{\text{mel}}}$ — The single most important patient-safety metric. A false-negative melanoma represents delayed cancer treatment.
