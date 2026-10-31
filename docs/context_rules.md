# Context Rules — Clinical Demographics & Anatomical Evidence Contract

**Document Class**: Level 1 Deliverable / Agent A4 Specification  
**Branch**: `feature/aashi-a4-context`  
**Module Referenced**: `src/context_agent.py` (Implementation), `tests/test_context_agent.py` (Validation)  
**Author**: Aashi (Agent A4 Owner — Context & Demographic Evaluation)  
**Audience**: Clinical judges, downstream orchestration agents, diagnostic model developers, CI gatekeepers  

---

## 1. Executive Summary & Clinical Rationale

### The Role of Context in Dermatology Clinical Decision Support

In clinical dermatology, a cutaneous lesion cannot be evaluated in isolation from the patient to whom it belongs. Experienced clinicians integrate visual dermoscopic morphology with non-image clinical context — primarily **patient age**, **biological sex**, and **anatomical lesion location** — to formulate differential diagnoses, assess prior risk probabilities, and determine urgency.

Agent A4 (**Context & Demographic Evaluation Agent**) is responsible for ingesting, validating, and structuring non-image patient context into an objective, standardized evidence contract for downstream clinical coordination.

| Clinical Variable | Diagnostic Significance | Safety & Bias Consideration |
| :--- | :--- | :--- |
| **Patient Age** | Melanoma is rare in pediatric populations (<18 years), where Spitz nevi and congenital melanocytic nevi are common. Conversely, the incidence of melanoma, basal cell carcinoma (BCC), squamous cell carcinoma (SCC), and actinic keratosis increases sharply in geriatric patients (>=65 years) due to cumulative UV exposure. | Age must never be used to rule out malignancy entirely; it serves as a risk modifier, not a definitive rule. Out-of-bounds ages (<0, >125) indicate record corruption and must be quarantined. |
| **Biological Sex** | Epidemiological distributions exhibit distinct anatomic predilections by sex: superficial spreading melanoma is more common on the trunk in males and on the lower extremities in females. | Normalization must handle diverse data conventions while preserving unconfirmed records as missing. |
| **Anatomical Site** | Lesion biology varies radically by location. **Acral sites** (palms, soles, nail units) harbor Acral Lentiginous Melanoma (ALM), which is non-UV-related and has high mortality due to diagnostic delay. **Head and neck** lesions have heavy cumulative solar elastosis (e.g., Lentigo Maligna). | Non-canonical or unknown anatomical descriptions must be mapped conservatively without speculative guessing. |

---

## 2. Agent A4 Core Design Principles

1. **Strict Non-Invention of Data (Zero Imputation)**:
   Agent A4 never assumes or imputes missing demographic data. Missing fields remain explicitly `None` in values and are systematically enumerated in `missing_fields`.
2. **Ground-Truth Quarantine (Strict Label Decoupling)**:
   Diagnostic ground-truth keys (`dx`, `dx_type`, `diagnosis`, `true_label`, etc.) must **never** influence or be propagated by Agent A4. If present in input dictionaries, they are detected, quarantined, logged in `notes`, and excluded from output.
3. **Objective Observational Evidence**:
   Agent A4 generates descriptive, factual epidemiological flags (e.g., `ACRAL_SITE`, `PEDIATRIC_PATIENT`) rather than speculative diagnostic predictions. Diagnostic classification is reserved for Agent A2.
4. **Deterministic and Standalone Execution**:
   Agent A4 executes without mandatory third-party heavy deep learning dependencies, allowing fast, reliable unit testing in any runtime environment.

---

## 3. Output Schema for `evaluate_context()`

The primary interface for Agent A4 is `evaluate_context(metadata: Optional[dict[str, Any]] = None) -> dict[str, Any]`.

### Type Specification

```python
{
    "age": float | None,               # Normalized age in years (rounded to 1 decimal place), or None if missing/invalid
    "sex": str | None,                 # Normalized biological sex ("male", "female", "other"), or None if missing
    "localization": str | None,        # Normalized anatomical localization site (lowercase string), or None if missing
    "missing_fields": list[str],       # Explicit list of unprovided or invalid demographic keys (subset of ["age", "sex", "localization"])
    "context_flags": list[str],        # Clinical observational and completeness flags
    "notes": list[str]                 # Transparent audit trail explaining parsing, normalization, and warnings
}
```

### JSON Schema Contract Example

```json
{
  "age": 72.0,
  "sex": "male",
  "localization": "scalp",
  "missing_fields": [],
  "context_flags": [
    "GERIATRIC_PATIENT",
    "HEAD_NECK_SITE",
    "CONTEXT_COMPLETE"
  ],
  "notes": [
    "Patient age recorded: 72.0 years.",
    "Observation: Geriatric patient (>=65 yr).",
    "Recorded biological/clinical sex: male.",
    "Anatomical site recorded: scalp.",
    "Full demographic and anatomical localization context provided."
  ]
}
```

---

## 4. Handling Missing, Null, and Invalid Metadata

When handling real-world clinical and EHR feeds, demographic data is frequently incomplete, blank, or improperly formatted. Agent A4 enforces a rigorous parsing rule set:

### 4.1 Missing Value Detection (`_is_missing`)

A field value is treated as missing if:
- The value is Python `None`.
- The numeric value is `math.isnan(val)`.
- The stripped lowercase string equals: `""`, `"nan"`, `"none"`, `"null"`, `"unknown"`, `"?"`, `"n/a"`, or `"not reported"`.

### 4.2 Age Validation Rules
- **Alternative Keys Ingested**: `'age'`, `'patient_age'`, `'patientAge'`.
- **Parsing**: Coerced to `float` and rounded to 1 decimal point.
- **Valid Range**: $0.0 \le \text{age} \le 125.0$.
- **Out-of-Bounds / Malformed**: If `age < 0.0`, `age > 125.0`, or cannot be parsed as a float, it is discarded, appended to `missing_fields`, and an audit note is emitted explaining the exclusion.

### 4.3 Biological Sex Normalization Rules
- **Alternative Keys Ingested**: `'sex'`, `'gender'`, `'patient_sex'`, `'patientGender'`.
- **Canonical Mappings**:
  - `{"m", "male"}` $\rightarrow$ `"male"`
  - `{"f", "female"}` $\rightarrow$ `"female"`
  - `{"other", "non-binary", "intersex"}` $\rightarrow$ Preserved as lowercase string.
  - Any unconfirmed or unrecognized strings (e.g. `"unknown"`, `"?"`) $\rightarrow$ Discarded to `None`, registered in `missing_fields`.

### 4.4 Anatomical Localization Rules
- **Alternative Keys Ingested**: `'localization'`, `'site'`, `'anatomical_site'`, `'anatomical_localization'`.
- **Canonical Sites Recognized**:
  `abdomen`, `acral`, `back`, `chest`, `ear`, `face`, `foot`, `genital`, `hand`, `lower extremity`, `neck`, `scalp`, `trunk`, `upper extremity`, `unknown`.
- Missing/Placeholder strings (`"unknown"`, `"not reported"`, `"?"`) $\rightarrow$ Treated as missing, registered in `missing_fields`.

### 4.5 Completeness Flags
Depending on the number of valid fields, exactly one completeness flag is assigned:
- `ALL_CONTEXT_MISSING`: All 3 demographic variables (`age`, `sex`, `localization`) are absent or invalid.
- `INCOMPLETE_CONTEXT`: 1 or 2 demographic variables are absent.
- `CONTEXT_COMPLETE`: All 3 demographic variables are present and validated.

---

## 5. Ground-Truth Quarantine Policy

### The Danger of Label Contamination

In multi-agent clinical decision support architectures, diagnostic ground truth must remain completely isolated from early perception and context modules. If an agent inspects diagnostic columns from training metadata (such as HAM10000 `dx`, `dx_type`, or evaluation labels `true_label`), the following failures occur:
1. **Data Leakage**: The agent bases prior assessments on ground truth rather than authentic patient demographics.
2. **Evaluation Invalidation**: Performance metrics on test splits become artificial and untrustworthy.
3. **Safety Breach**: In production deployment, ground truth is never present at inference time; models with leakage fail catastrophically.

### Quarantine Implementation

Agent A4 enforces an explicit quarantine blacklist:

```python
FORBIDDEN_GROUND_TRUTH_KEYS = (
    "dx",
    "dx_type",
    "diagnosis",
    "true_label",
    "ground_truth",
    "label",
    "label_name",
    "label_idx",
)
```

**Quarantine Actions**:
1. Scan input dictionary keys against `FORBIDDEN_GROUND_TRUTH_KEYS` (case-insensitive).
2. Quarantine all identified ground-truth keys: they are strictly ignored and never copied to the output dictionary.
3. Record an audit quarantine log in `notes`:
   `"Audit note: Ground-truth key(s) ['dx', ...] detected in input and strictly quarantined; context agent evaluates demographics only."`

---

## 6. Clinical Observational Flags Reference

Agent A4 generates specific clinical observational flags that inform the downstream coordination graph (e.g. LangGraph / Orchestrator):

| Flag | Trigger Condition | Clinical Meaning & Downstream Guidance |
| :--- | :--- | :--- |
| `PEDIATRIC_PATIENT` | $0.0 \le \text{age} < 18.0$ | Pediatric lesion. High prior for benign congenital nevi, Spitz nevi, or viral lesions. Alerts downstream triage to avoid unnecessary invasive biopsies unless unequivocal malignant dermoscopy is observed. |
| `GERIATRIC_PATIENT` | $\text{age} \ge 65.0$ | Geriatric patient. Elevated prior risk for cumulative sun damage, keratinocyte carcinomas (BCC/SCC), and Lentigo Maligna Melanoma. |
| `HEAD_NECK_SITE` | `localization` in `{"face", "scalp", "ear", "neck"}` | Chronically sun-exposed skin. High susceptibility to solar elastosis, actinic keratoses, and lentigo maligna. Cosmetic and surgical margin sensitivity. |
| `ACRAL_SITE` | `localization` in `{"acral", "hand", "foot"}` | Palmar, plantar, or subungual site. High risk zone for Acral Lentiginous Melanoma (ALM). ALM occurs across all skin types; atypical parallel ridge patterns demand immediate specialist evaluation. |
| `GENITAL_SITE` | `localization` == `"genital"` | Non-sun-exposed specialized anatomy. Differential includes extramammary Paget disease, genital lentigines, and mucosal melanoma. |
| `ALL_CONTEXT_MISSING` | $\text{len}(\text{missing\_fields}) == 3$ | No demographic context available. Downstream agents must rely exclusively on image quality and visual dermoscopy without demographic priors. |
| `INCOMPLETE_CONTEXT` | $1 \le \text{len}(\text{missing\_fields}) < 3$ | Partial context available. Downstream agents should utilize available fields while noting uncertainty in missing parameters. |
| `CONTEXT_COMPLETE` | $\text{len}(\text{missing\_fields}) == 0$ | Full non-image patient context validated. Enables full multi-modal evidence synthesis. |

---

## 7. Integration & Architecture Summary

Agent A4 participates in the overall Agentic DermaGuard pipeline:

```
                  Raw Input (Image + Metadata)
                               |
            +------------------+------------------+
            |                                     |
            v                                     v
     [ Agent A1: Quality ]               [ Agent A4: Context ]
     - Motion/focus blur                 - Age validation
     - Illumination & Glare              - Sex normalization
     - Vignetting & Occlusion            - Site categorization
            |                            - Ground-truth quarantine
            |                            - Missing-data flags
            |                                     |
            +------------------+------------------+
                               |
                               v
                 [ Agent A2: Diagnosis & ML ]
                 - Feature extraction & CNN
                 - Calibrated probabilities
                               |
                               v
                 [ Downstream Orchestration ]
                 - Synthesis of Quality + Context + Diagnosis
                 - Final clinical report & triage decision
```

---

## 8. Verification & Test Suite Traceability

All specifications detailed in this contract are verified continuously via `tests/test_context_agent.py`:
- `test_empty_and_none_metadata`: Validates null/missing data preservation and `ALL_CONTEXT_MISSING`.
- `test_full_valid_context`: Validates complete extraction and `CONTEXT_COMPLETE`.
- `test_partial_context`: Validates incomplete metadata reporting.
- `test_ground_truth_isolation`: Confirms quarantine of all forbidden keys.
- `test_demographic_age_ranges`: Checks `<18`, `>=65`, `<0`, `>125` thresholds.
- `test_sex_normalization`: Tests biological sex normalization.
- `test_anatomical_site_normalization`: Tests site categorization and clinical flags.

---

## 9. HAM10000 Metadata Audit

A second-stage read-only audit of the local HAM10000 dataset evaluated through `evaluate_context()` confirmed the demographic context profile across partitions:

- **Dataset Size**: **7,762** training records and **1,942** validation records (**9,704** total records across `data/train/metadata.csv` and `data/val/metadata.csv`).
- **Context Completeness**:
  - **Complete Context (`CONTEXT_COMPLETE`)**: **9,457** records (**97.45%**) have full demographic context (valid age, sex, and anatomical localization).
  - **Incomplete Context (`INCOMPLETE_CONTEXT`)**: **202** records (**2.08%**) have partial context with exactly one field missing.
  - **All Context Missing (`ALL_CONTEXT_MISSING`)**: **45** records (**0.46%**) have all three demographic fields missing or unknown.
- **Localization-Only Missingness**:
  - Localization is the predominant contributor to incomplete context, representing 182 training records and 33 validation records (182 records with localization-only missingness across the combined dataset; 149 training and 33 validation).
- **Missingness Pattern Invariant**:
  - **No records have exactly two missing fields**: Across the entire 9,704 instances, pairwise missingness is zero (age+sex, age+localization, and sex+localization are all 0). Records strictly possess either 0, 1, or all 3 missing demographic fields.
- **Ground-Truth Isolation**:
  - `dx` and any other ground-truth diagnostic labels were strictly excluded from Agent A4 inputs, upholding complete decoupling between patient demographic context and diagnostic classification.
