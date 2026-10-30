# Agentic DermaGuard — Architectural Design Notes

**Project:** Agentic DermaGuard | Autonomous Multi-Agent Skin Lesion Screening & Triage  
**Track:** JAI 2026 International Agentic AI Hackathon — Healthcare Track  
**Technical Lead:** Ilka Mansoori  
**Author:** Aashi (Data Pipeline, Context Agent, Frontend & Submission Integration)  
**Date:** 2026-10-30  
**Version:** 1.0 (Day 2 Milestone)  

---

## 1. Executive Summary & Clinical Rationale

Skin cancer screening in primary care and outpatient clinics is subject to two major failure modes:
1. **Low Sensitivity / Missed Malignancies:** Early malignant lesions (e.g. melanoma, basal cell carcinoma) misclassified as benign nevi or keratoses.
2. **Operational Overload from Degraded Acquisitions:** Blurry, specular-glared, or shadowed images fed to automated classifiers produce uncalibrated, erratic predictions, leading to diagnostic error.

**Agentic DermaGuard** addresses these vulnerabilities by implementing an agentic, multi-stage clinical decision-support pipeline. Instead of routing raw images directly to a classification model, the architecture employs specialized agents that evaluate acquisition quality, extract clinical demographic context, assess diagnostic probabilities, and enforce clinical triage boundaries.

---

## 2. Multi-Agent Pipeline & Information Flow

The decision-support flow operates sequentially through six clinical agents (A1–A6):

```
[ Dermatoscopic Image + Demographic Context ]
                      │
                      ▼
        ┌───────────────────────────┐
        │  Agent A1: Image Quality  │
        │  (Blur, light, glare)     │
        └─────────────┬─────────────┘
                      │
         ┌────────────┴────────────┐
         │                         │
     [FAIL]                   [PASS / WARN]
         ▼                         │
 ┌───────────────┐                 ▼
 │   RE-IMAGE    │   ┌───────────────────────────┐     ┌───────────────────────────┐
 │ (Hard Reject) │   │  Agent A2: Diagnosis      │     │  Agent A4: Patient Context│
 └───────────────┘   │  (7-class probabilities)  │     │  (Age, sex, site audit)   │
                     └─────────────┬─────────────┘     └─────────────┬─────────────┘
                                   │                                 │
                                   └────────────────┬────────────────┘
                                                    │
                                                    ▼
                                    ┌───────────────────────────────┐
                                    │ Agent A6: Clinical Coordinator│
                                    │ Triage policy mapping         │
                                    └───────────────┬───────────────┘
                                                    │
                                    ┌───────────────┴───────────────┐
                                    ▼                               ▼
                            ┌───────────────┐               ┌───────────────┐
                            │  AUTO-TRIAGE  │               │     REFER     │
                            │ Benign / safe │               │ High-risk/Warn│
                            └───────────────┘               └───────────────┘
```

---

## 3. Agent Specifications & Data Contracts

### 3.1 Agent A1: Image Quality Agent (`src/quality_agent.py`)
- **Owner:** Palak
- **Public Interface:** `assess_quality(image_input: Union[str, Path, np.ndarray]) -> dict`
- **Output Schema:**
  ```python
  {
      "status": "PASS" | "WARN" | "FAIL",
      "score": float,  # [0.0, 1.0]
      "reasons": list[str],
      "checks": {
          "blur": {"status": str, "value": float, "threshold": float, "passed": bool},
          "illumination": {"status": str, "mean_luminance": float, "passed": bool},
          "glare": {"status": str, "glare_percentage": float, "threshold": float, "passed": bool},
          "obstruction": {"status": str, "passed": bool}
      }
  }
  ```
- **Triage Action:**
  - `FAIL` $\implies$ Hard Stop (`RE-IMAGE`). No diagnosis performed.
  - `WARN` $\implies$ Proceed with soft flag / human referral recommendation.
  - `PASS` $\implies$ Proceed to standard diagnosis.

### 3.2 Agent A2: Lesion Diagnosis Agent (`src/diagnosis_agent.py`)
- **Owner:** Ilka (Technical Lead)
- **Architecture Contract:** EfficientNet-B0 backbone (1,280 features) + Logistic Regression classifier.
- **Preprocessing:** Native $600 \times 450$ RGB images resized to $224 \times 224$ with ImageNet normalization ($mean=[0.485, 0.456, 0.406]$, $std=[0.229, 0.224, 0.225]$).
- **Public Interface:** `diagnose(image_path: str | Path) -> dict`
- **Output Schema:**
  ```python
  {
      "class": "akiec" | "bcc" | "bkl" | "df" | "mel" | "nv" | "vasc",
      "confidence": float,  # max probability in [0.0, 1.0]
      "probabilities": {
          "akiec": float, "bcc": float, "bkl": float, "df": float,
          "mel": float, "nv": float, "vasc": float
      }
  }
  ```
- **Invariants:** Probabilities sum to $1.0 \pm 10^{-3}$; `class` matches `argmax(probabilities)`.
- **Current Status:** *Pending final model training.* In the frontend and submission modules, unverified predictions are never invented; the status is reported as `NOT AVAILABLE`.

### 3.3 Agent A3: Lesion Morphology Agent
- **Role:** Extracts ABCD dermoscopic morphological features (Asymmetry index, Border irregularity, Color heterogeneity, Differential dermoscopic structures).
- **Public Interface:** `analyze_lesion(image: PIL.Image.Image) -> dict`
- **Current Status:** *Planned for Day 2 integration.*

### 3.4 Agent A4: Demographic & Context Evaluation Agent (`src/context_agent.py`)
- **Owner:** Aashi
- **Public Interface:** `evaluate_context(metadata: Optional[dict]) -> dict`
- **Design Principles:**
  - Preserves missing values as explicit `None` without fabrication.
  - Strict decoupling from ground-truth labels (`dx`, `diagnosis`, `true_label` are quarantined and ignored).
  - Normalizes demographic age, biological sex, and anatomical sites.
  - Generates factual clinical observation flags (`PEDIATRIC_PATIENT`, `GERIATRIC_PATIENT`, `HEAD_NECK_SITE`, `ACRAL_SITE`).
- **Output Schema:**
  ```python
  {
      "age": float | None,
      "sex": "male" | "female" | "other" | None,
      "localization": str | None,
      "missing_fields": list[str],
      "context_flags": list[str],
      "notes": list[str]
  }
  ```

### 3.5 Agent A5: Safety & Regulatory Audit Agent
- **Role:** Evaluates epistemic uncertainty, out-of-distribution detection, and clinical risk thresholds.
- **Public Interface:** `audit_safety(evidence: dict) -> dict`
- **Current Status:** *Planned for Day 2 integration.*

### 3.6 Agent A6: Clinical Triage Coordinator
- **Role:** Synthesizes outputs from Agents A1–A5 into an actionable triage recommendation.
- **Triage Categories:**
  1. `AUTO-TRIAGE`: High-confidence benign prediction on a high-quality acquisition (`PASS`).
  2. `REFER`: Any suspected malignancy (`mel`, `bcc`, `akiec`), borderline confidence ($< 0.85$), or borderline image quality (`WARN`).
  3. `RE-IMAGE`: Severe image quality degradation (`FAIL`).

---

## 4. Data Pipeline & Partition Engineering (`src/data.py`)

### 4.1 Path Security & Directory Containment
All file path operations enforce strict path traversal defenses:
- `validate_image_id()` rejects directory separators (`/`, `\`), path traversal tokens (`..`), drive colons, and non-alphanumeric characters.
- Base filenames are checked to ensure resolved paths remain strictly within `data/train/images` or `data/val/images`.

### 4.2 Lesion-Grouped Splitting (Zero Leakage Guarantee)
Dermatological datasets often contain multiple images of the same physical skin lesion acquired at different angles or magnification levels. A naive random split would place images of the same lesion into both training and validation sets, causing catastrophic data leakage and optimistic bias.

`src/data.py` implements:
- `split_metadata_by_lesion(metadata, val_fraction, seed)`: Partitions unique `lesion_id`s deterministically, ensuring that all images for a given lesion belong exclusively to one partition.
- `verify_split_leakage(train_df, val_df)`: Programmatically confirms that:
  $$\text{Train}_{\text{images}} \cap \text{Val}_{\text{images}} = \emptyset \quad \text{and} \quad \text{Train}_{\text{lesions}} \cap \text{Val}_{\text{lesions}} = \emptyset$$
- Audit verified that the official HAM10000 training (7,762 images, 5,815 lesions) and validation (1,942 images, 1,456 lesions) subsets have **exactly zero image and lesion overlap**.

---

## 5. Submission Client Integration (`src/submit.py`)

For the official hidden-test evaluation, the submission module:
- Ingests hidden test images.
- Enforces the strict 3-column CSV schema:
  ```csv
  image_id,predicted_label,decision
  ```
- Rejects duplicate `image_id` entries.
- Validates that `predicted_label` is one of the 7 canonical classes.
- Validates that `decision` is one of `AUTO-TRIAGE`, `REFER`, `RE-IMAGE`.
- Includes an authorization safety gate (`submit_to_portal`) requiring `confirm=True` and environment credentials (`PORTAL_API_URL`, `PORTAL_API_KEY`) to prevent unauthorized external network transmissions.

---

## 6. Frontend Clinical Screening Dashboard (`app.py`)

The Streamlit dashboard provides an intuitive, transparent interface for primary care clinicians:
1. **Acquisition Ingestion:** File upload (`.jpg`, `.jpeg`, `.png`) or local validation sample selection.
2. **Patient Demographics Form:** Optional age, sex, and anatomical localization fields.
3. **Multi-Agent Evidence Tabs:**
   - Quality Tab: Real-time status badge, overall score, and metric breakdowns (blur, illumination, glare, obstruction).
   - Context Tab: Demographic summary and clinical observation flags.
   - Diagnosis Tab: Class probability bar chart (or clear `NOT AVAILABLE` notice when model training is pending).
   - Audit Tab: Comprehensive JSON inspection and one-click download button for the clinical screening report.

---

## 7. Known Limitations & Clinical Disclaimer

1. **Investigational Research Scope:** Agentic DermaGuard is developed solely for the JAI 2026 Hackathon. It has not undergone prospective clinical trials, CE mark, or FDA 510(k) clearance.
2. **Model Training Dependencies:** The diagnosis agent (A2) relies on final model training and artifact persistence from Technical Lead Ilka. No diagnostic accuracy claims (e.g. 90% accuracy) are made prior to running official evaluation over the full validation set.
3. **Class Imbalance:** In HAM10000, 67% of cases are benign nevi (`nv`). Downstream models must incorporate class-balanced weighting to avoid under-detecting rare, life-threatening categories such as melanoma (`mel`).
