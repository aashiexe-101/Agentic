# Agentic DermaGuard 🔬
### Autonomous Multi-Agent Skin Lesion Screening & Clinical Triage System
**Event:** JAI 2026 International Agentic AI Hackathon — Healthcare Track  
**Technical Lead:** Ilka Mansoori  
**Engineering Team:** Ilka Mansoori, Utsavi Sinha, Aashi, Palak Singhal  
**Repository:** [https://github.com/aashiexe-101/Agentic](https://github.com/aashiexe-101/Agentic)

---

## 1. System Overview

**Agentic DermaGuard** is an agentic clinical decision-support system designed for dermatoscopic skin lesion screening and risk-stratified triage. Rather than relying on an opaque, uncalibrated deep-learning black box, the system orchestrates six specialized clinical agents to evaluate image quality, lesion morphology, patient demographic context, and diagnostic probabilities before reaching an auditable clinical decision:

```
                  ┌──────────────────────────────────────────────┐
                  │    Clinical Dermatoscopic Photograph Input   │
                  │         + Patient Demographic Context        │
                  └──────────────────────┬───────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
     ┌────────────────────────┐                     ┌────────────────────────┐
     │ Agent A1: Image Quality│                     │ Agent A4: Context      │
     │ Sharpness, glare, light│                     │ Age, sex, site audit   │
     └───────────┬────────────┘                     └───────────┬────────────┘
                 │                                               │
      [FAIL]     │ [PASS / WARN]                                 │
     ┌───────────┴───────────────────────┐                       │
     ▼                                   ▼                       │
┌──────────────┐             ┌───────────────────────┐           │
│   RE-IMAGE   │             │ Agent A2: Diagnosis   │           │
│ (Hard Reject)│             │ 7-class probabilities │           │
└──────────────┘             └───────────┬───────────┘           │
                                         │                       │
                                         ▼                       ▼
                             ┌───────────────────────────────────────┐
                             │ Agent A6: Clinical Triage Coordinator │
                             │ Multi-agent synthesis & safety checks │
                             └───────────────────┬───────────────────┘
                                                 │
                               ┌─────────────────┴─────────────────┐
                               ▼                                   ▼
                   ┌───────────────────────┐           ┌───────────────────────┐
                   │      AUTO-TRIAGE      │           │         REFER         │
                   │ High-confidence benign│           │ High-risk / borderline│
                   └───────────────────────┘           └───────────────────────┘
```

---

## 2. Multi-Agent Architecture & Team Ownership

| Agent / Module | Owner | Source File | Responsibilities | Current Status |
|:---|:---:|:---|:---|:---:|
| **Data Pipeline** | **Aashi** | `src/data.py` | Path security, HAM10000 ingestion, lesion-grouped splits, zero leakage. | **COMPLETE (VERIFIED)** |
| **Agent A1: Image Quality** | **Palak** | `src/quality_agent.py` | Laplacian blur, illumination, glare %, obstruction checks (PASS/WARN/FAIL). | **COMPLETE (VERIFIED)** |
| **Agent A2: Diagnosis** | **Ilka** | `src/diagnosis_agent.py` | EfficientNet-B0 feature extraction + 7-class Logistic Regression. | *Awaiting Model Training* |
| **Agent A3: Morphology** | **Team** | `nodes/research.py` | ABCD rule extraction (Asymmetry, Border, Color, Dermoscopic structures). | *Design Specification* |
| **Agent A4: Patient Context** | **Aashi** | `src/context_agent.py` | Demographic ingestion, missing field auditing, anatomical site isolation. | **COMPLETE (VERIFIED)** |
| **Agent A5: Safety Audit** | **Team** | `nodes/verify.py` | Epistemic uncertainty scoring and clinical boundary verification. | *Design Specification* |
| **Agent A6: Coordinator** | **Team** | `app.py` / `graph.py` | Evidence synthesis, clinical triage mapping (`AUTO-TRIAGE`, `REFER`, `RE-IMAGE`). | **COMPLETE (VERIFIED)** |
| **Evaluation Engine** | **Utsavi** | `src/evaluate.py` | 18-column CSV evaluation, per-class metrics, confusion matrix export. | **COMPLETE (VERIFIED)** |
| **Submission Client** | **Aashi** | `src/submit.py` | Hidden-test 3-column CSV serialization (`image_id,predicted_label,decision`). | **COMPLETE (VERIFIED)** |
| **Clinical UI Dashboard** | **Aashi** | `app.py` | Streamlit interactive screening interface & JSON audit report export. | **COMPLETE (VERIFIED)** |

---

## 3. Dataset Specifications & Directory Layout

The system utilizes a verified, curated subset of the benchmark **HAM10000** ("Human Against Machine with 10000 training images") dataset:
- **Total Dataset Size:** 9,704 dermatoscopic images ($600 \times 450$ px, 24-bit RGB JPEGs).
- **Training Partition:** 7,762 images (5,815 unique lesions).
- **Validation Partition:** 1,942 images (1,456 unique lesions).
- **Partition Ratio:** Exact $80.0\% / 20.0\%$ split with **0 image overlap and 0 lesion overlap** ($Train \cap Val = \emptyset$).

```
data/
├── train/
│   ├── metadata.csv       # 7,762 clinical records
│   └── images/            # 7,762 JPEG image files
└── val/
    ├── metadata.csv       # 1,942 clinical records
    └── images/            # 1,942 JPEG image files
```

### Seven Canonical Diagnostic Classes
1. `akiec` — Actinic keratoses and intraepithelial carcinoma
2. `bcc` — Basal cell carcinoma
3. `bkl` — Benign keratosis-like lesions (solar lentigines / seborrheic keratoses)
4. `df` — Dermatofibroma
5. `mel` — Melanoma
6. `nv` — Melanocytic nevi
7. `vasc` — Vascular lesions (angiomas, pyogenic granulomas, hemorrhage)

---

## 4. Quickstart & Installation

### Environment Setup
Clone the repository and install dependencies declared in `requirements.txt`:
```bash
git clone https://github.com/aashiexe-101/Agentic.git
cd Agentic
pip install -r requirements.txt
```

### 1. Launch Interactive Clinical Screening Dashboard (Streamlit)
```bash
streamlit run app.py
```
- Upload dermatoscopic images (`.jpg`, `.jpeg`, `.png`) or test on sample validation images.
- Input patient demographic context (age, sex, anatomical lesion site).
- Inspect real-time multi-agent assessments (A1 Quality, A4 Context, A2 Diagnosis probabilities).
- Download the complete structured JSON screening report.

### 2. Run Data Pipeline Sanity Check
```bash
python src/data.py
```

### 3. Evaluate Patient Context (Agent A4 CLI)
```bash
python src/context_agent.py --age 54 --sex female --site back
```

### 4. Validate Quality of a Single Image (Agent A1 CLI)
```bash
python src/quality_agent.py --image data/val/images/ISIC_0024306.jpg
```

### 5. Validate Hidden-Test Submission CSV (Agent Submission Client)
```bash
python src/submit.py --validate-only outputs/submission.csv
```

### 6. Run Evaluation Metrics Engine
```bash
python src/evaluate.py --input outputs/fake_val_results.csv --output-metrics outputs/metrics.json
```

### 7. Run Complete Automated Test Suite
```bash
pytest tests/
```
Or execute standalone test modules without pytest:
```bash
python tests/test_data.py
python tests/test_context_agent.py
python tests/test_submit.py
python tests/test_frontend.py
python tests/test_quality.py
python tests/test_contracts.py
```

---

## 5. Data Contracts & Output Formats

### A. Internal Validation CSV Contract (`outputs/val_results.csv` — 18 columns)
Codified in `src/evaluate.py` and `tests/test_contracts.py`:
```csv
image_id,true_label,pred_class,confidence,prob_akiec,prob_bcc,prob_bkl,prob_df,prob_mel,prob_nv,prob_vasc,q_status,q_score,q_reasons,q_blur,q_illumination,q_glare,q_obstruction
```

### B. Official Hidden-Test Submission Contract (`outputs/submission.csv` — 3 columns)
Codified in `src/submit.py`:
```csv
image_id,predicted_label,decision
ISIC_0024306,nv,AUTO-TRIAGE
ISIC_0024307,mel,REFER
ISIC_0024308,bcc,RE-IMAGE
```
- **Constraints:** Zero duplicate image IDs; exact 3 columns; `predicted_label` $\in$ 7 canonical classes; `decision` $\in$ `{"AUTO-TRIAGE", "REFER", "RE-IMAGE"}`.

---

## 6. Known Limitations & Clinical Disclaimer

- **Decision-Support Scope:** Agentic DermaGuard is an investigational decision-support research tool developed for the JAI 2026 Hackathon. It is **not** an approved diagnostic medical device and must not replace evaluation by a licensed board-certified dermatologist.
- **Model Training Notice:** The baseline diagnosis model (`src/diagnosis_agent.py`) is awaiting final weights training from Technical Lead Ilka. In compliance with strict hackathon integrity standards, the frontend displays `NOT AVAILABLE` rather than displaying unverified or synthetic diagnostic claims.
- **External Portal Transmission:** The submission client (`src/submit.py`) runs in safe offline mode by default and requires explicit technical lead authorization (`confirm=True`) before transmitting to any external endpoint.
