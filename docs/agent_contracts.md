# Agentic DermaGuard: Multi-Agent Contract Specifications

**Project**: JAI 2026 International Agentic AI Hackathon — Healthcare Track  
**Architecture**: 6-Agent Collaborative Diagnostic Hierarchy  
**Policy Version**: `1.0`  
**Integration Base**: `integration/phase2-integration`  

---

## 1. Agent A1: Image Quality Assurance Agent

- **Module**: `src/quality_agent.py`
- **Owner**: Palak
- **Primary Function**: `assess_quality(image_input) -> dict`
- **Supported Inputs**: File path (str / Path), PIL Image, or NumPy array (`uint8`).

### Output Schema

| Field Name | Type | Allowed Values / Range | Description |
|---|---|---|---|
| `overall_status` | `str` | `"PASS"`, `"WARN"`, `"FAIL"` | Aggregate optical quality verdict. |
| `score` | `float` | `0.0` to `1.0` | Weighted continuous quality score. |
| `blur_check` | `dict` | `{"passed": bool, "metric": float, "threshold": float}` | Laplacian variance sharpness evaluation. |
| `illumination_check` | `dict` | `{"passed": bool, "mean_intensity": float, ...}` | Brightness uniformity evaluation. |
| `glare_check` | `dict` | `{"passed": bool, "saturated_ratio": float, ...}` | Specular reflection / glare saturation check. |
| `obstruction_check` | `dict` | `{"passed": bool, "hair_density": float, ...}` | Dark filament and hair occlusion detector. |
| `reasons` | `list[str]` | Free-text clinical rationales | Explicit explanations for any failed checks. |

---

## 2. Agent A2: Diagnosis Classifier Agent

- **Module**: `src/diagnosis_agent.py`
- **Owner**: Ilka
- **Primary Function**: `predict(image_input) -> dict` / `diagnose(image_input) -> dict`
- **Supported Inputs**: Dermoscopy image path.

### Output Schema

| Field Name | Type | Allowed Values / Range | Description |
|---|---|---|---|
| `predicted_class` | `str` | `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc` | Primary model classification. |
| `confidence` | `float` | `0.0` to `1.0` | Top-1 probability score. |
| `probabilities` | `dict[str, float]` | 7 canonical keys summing to `1.0` | Full softmax probability distribution across classes. |

---

## 3. Agent A3: Lesion Morphology & Evidence Agent

- **Module**: `src/lesion_agent.py`
- **Owner**: Palak
- **Primary Function**: `analyze_lesion(image_input) -> dict`
- **Supported Inputs**: Dermoscopy image path or NumPy array.

### Output Schema (Schema v1.1 Hardened)

| Field Name | Type | Allowed Values / Range | Description |
|---|---|---|---|
| `suspicion_score` | `float` | `0.0` to `1.0` | Heuristic morphological ABCD risk index. |
| `reliability` | `str` | `"HIGH"`, `"MEDIUM"`, `"LOW"` | Segmentation and measurement confidence. |
| `reliable` | `bool` | `True` if reliability != LOW, else `False` | Top-level boolean matching A5 veto guard. |
| `error` | `str | None` | `None` on success, error message on failure | Explicit error indicator matching A5 check. |
| `features` | `list[str]` | List of detected morphology tokens | e.g., `["asymmetry", "atypical_network_candidate"]`. |
| `flags` | `dict[str, bool]` | Boolean findings map | `{"asymmetric": bool, "border_irregular": bool, ...}`. |
| `evidence` | `list[str]` | Explanatory rationale strings | Diagnostic evidence and limitations. |
| `measurements` | `dict` | Area, perimeter, compactness, color stats | Raw quantitative lesion measurements. |
| `mask` | `np.ndarray | None` | Binary 2D array or None | Lesion segmentation mask. |

---

## 4. Agent A4: Clinical Context Evaluator Agent

- **Module**: `src/context_agent.py`
- **Owner**: Aashi
- **Primary Function**: `evaluate_context(metadata) -> dict`
- **Supported Inputs**: Dictionary with `age`, `sex`, `localization` (or None).

### Output Schema

| Field Name | Type | Allowed Values / Range | Description |
|---|---|---|---|
| `age` | `float | None` | `0.0` to `125.0` (or `None`) | Validated patient age in years. |
| `sex` | `str | None` | `"male"`, `"female"`, or `None` | Normalized biological sex. |
| `localization` | `str | None` | Standard anatomical site or `None` | Lesion anatomical location. |
| `missing_fields` | `list[str]` | e.g. `["age", "sex"]` | List of missing demographic parameters. |
| `context_flags` | `list[str]` | e.g. `["PEDIATRIC_PATIENT", "ELDERLY_PATIENT"]` | Clinical demographic risk flags. |
| `notes` | `list[str]` | Free-text clinical notes | Summary observations and audit log. |

---

## 5. Agent A5: Clinical Safety Auditor Agent

- **Module**: `src/safety_agent.py`
- **Owner**: Utsavi
- **Primary Function**: `audit_safety(agent_outputs) -> dict`
- **Supported Inputs**: Dictionary containing `{ "a1_quality": ..., "a2_diagnosis": ..., "a3_lesion": ..., "a4_context": ... }`.

### Output Schema

| Field Name | Type | Allowed Values / Range | Description |
|---|---|---|---|
| `safety_flag` | `bool` | `True` or `False` | Indicates whether any safety condition was flagged. |
| `veto_auto_triage` | `bool` | `True` or `False` | Overrides and strictly forbids `AUTO-TRIAGE`. |
| `requires_human_review`| `bool` | `True` or `False` | Mandates human specialist review. |
| `risk_level` | `str` | `"LOW"`, `"MODERATE"`, `"HIGH"`, `"CRITICAL"` | Stratified clinical risk level. |
| `reasons` | `list[str]` | Auditable list of safety findings | Clinical triggers behind veto or flags. |
| `conflicts` | `list[str]` | Cross-agent discrepancy notes | e.g., benign model vs atypical dermoscopy. |
| `missing_evidence` | `list[str]` | Missing fields | Context or morphology fields absent. |

---

## 6. Agent A6: Triage Coordinator Agent

- **Module**: `src/coordinator.py` (and alias `src/coordinator_agent.py`)
- **Owner**: Palak (Integration Owner) & Utsavi
- **Primary Function**: `coordinate_triage(agent_outputs) -> dict`
- **Convenience Aliases**: `coordinate()`, `triage_case()`

### Output Schema

| Field Name | Type | Allowed Values / Range | Description |
|---|---|---|---|
| `predicted_label` | `str` | 7 canonical classes or `"unknown"` | Primary diagnostic classification. |
| `decision` | `str` | `"AUTO-TRIAGE"`, `"REFER"`, `"RE-IMAGE"` | Authoritative triage outcome. |
| `requires_review` | `bool` | `True` or `False` | Mandatory clinician review flag. |
| `rationale` | `list[str]` | Structured decision sentences | Explains exact deterministic rule applied. |
| `agent_evidence` | `dict` | Multi-agent synthesis payload | Traceability map of all 5 upstream agents. |
| `trace` | `dict` | Alias to `agent_evidence` | Direct access for audit loggers. |
| `policy_version` | `str` | `"1.0"` | Policy version string. |

---

## 7. Official 3-Column Submission Format

Generated via `generate_submission_csv()` in `src/pipeline.py` and `src/coordinator.py`:

```csv
image_id,prediction,decision
ISIC_0024306,nv,AUTO-TRIAGE
ISIC_0024307,mel,REFER
ISIC_0024308,bkl,RE-IMAGE
```

- **Validation Constraints**:
  - No duplicate `image_id`s.
  - Zero missing/NaN values.
  - `prediction` must be one of `['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']`.
  - `decision` must be one of `['AUTO-TRIAGE', 'REFER', 'RE-IMAGE']`.
