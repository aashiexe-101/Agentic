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

| Agent / Module | Owner | Source File | Responsibilities | Implementation Status |
|:---|:---:|:---|:---|:---:|
| **Data Pipeline** | **Aashi** | `src/data.py` | Path security, HAM10000 ingestion, lesion-grouped splits, zero leakage. | **COMPLETE (VERIFIED)** |
| **Agent A1: Image Quality** | **Palak** | `src/quality_agent.py` | Laplacian blur, illumination, glare %, obstruction checks (PASS/WARN/FAIL). | **COMPLETE (VERIFIED)** |
| **Agent A2: Diagnosis** | **Ilka** | `src/diagnosis_agent.py` | EfficientNet-B0 feature extraction + 7-class Logistic Regression (`models/classifier_v2.joblib`). | **IMPLEMENTED (Inference blocked on Python 3.14)** |
| **Agent A3: Morphology** | **Palak** | `src/lesion_agent.py` | ABCD rule extraction (Asymmetry, Border, Color, Dermoscopic structures). | **COMPLETE (VERIFIED)** |
| **Agent A4: Patient Context** | **Aashi** | `src/context_agent.py` | Demographic ingestion, missing field auditing, anatomical site isolation. | **COMPLETE (VERIFIED)** |
| **Agent A5: Safety Audit** | **Utsavi** | `src/safety_agent.py` | Epistemic uncertainty scoring, malignancy detection, and clinical veto authority. | **COMPLETE (VERIFIED)** |
| **Agent A6: Coordinator** | **Palak & Utsavi** | `src/coordinator.py` | Multi-agent synthesis, clinical triage mapping (`AUTO-TRIAGE`, `REFER`, `RE-IMAGE`). | **COMPLETE (VERIFIED)** |
| **Pipeline Orchestrator** | **Palak** | `src/pipeline.py` | End-to-end multi-agent execution pipeline and submission generator. | **COMPLETE (VERIFIED)** |
| **Evaluation Engine** | **Utsavi** | `src/evaluate.py` | 18-column CSV evaluation, per-class metrics, confusion matrix export. | **COMPLETE (VERIFIED)** |
| **Submission Client** | **Aashi** | `src/submit.py` | Hidden-test 3-column CSV serialization (`image_id,predicted_label,decision`). | **COMPLETE (VERIFIED)** |
| **Clinical UI Dashboard** | **Aashi** | `app.py` | Streamlit interactive screening workstation & clinical report export. | **COMPLETE (VERIFIED)** |

---

## 3. Dataset Specifications & Proper Attribution (Rule T9 & T10)

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

### Academic Citation & Source
> Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Sci Data* 5, 180161 (2018).  
> **DOI:** [10.1038/sdata.2018.161](https://doi.org/10.1038/sdata.2018.161)  
> **Source Repository:** International Skin Imaging Collaboration (ISIC) Archive / Harvard Dataverse.  
> **Licensing:** Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).

### De-Identification & Privacy Statement (Rule T10)
In strict compliance with patient confidentiality and privacy guidelines, all images and metadata in the HAM10000 dataset are completely de-identified. No personal identifiers, patient names, direct clinical contact information, or protected health information (PHI) are present or stored within the repository.

### Seven Canonical Diagnostic Classes
1. `akiec` — Actinic keratoses and intraepithelial carcinoma
2. `bcc` — Basal cell carcinoma
3. `bkl` — Benign keratosis-like lesions (solar lentigines / seborrheic keratoses)
4. `df` — Dermatofibroma
5. `mel` — Melanoma
6. `nv` — Melanocytic nevi
7. `vasc` — Vascular lesions (angiomas, pyogenic granulomas, hemorrhage)

---

## 4. Third-Party Components & Library Disclosures (Rule T6)

The project leverages the following verified third-party components, pre-trained backbones, and libraries:

| Component / Library | Purpose | License |
|:---|:---|:---|
| **EfficientNet-B0 (torchvision)** | Pretrained convolutional backbone used for feature extraction (1,280-dim representation). | BSD-3-Clause |
| **scikit-learn** | Logistic regression multi-class head, classification metrics, confusion matrices. | BSD-3-Clause |
| **OpenCV (`opencv-python-headless`)** | Laplacian sharpness variance, color segmentation, and morphological filtering. | Apache-2.0 |
| **Pillow (`PIL`)** | Dermatoscopic image loading, conversion, and validation. | HPND |
| **Streamlit** | Clinician screening workstation and interactive triage interface. | Apache-2.0 |
| **Pandas & NumPy** | Tabular metadata ingestion, array math, and contract validation. | BSD-3-Clause |
| **Matplotlib** | Confusion matrix plotting and visual evaluation reports. | PSF |

---

## 5. AI Assistance Disclosure (Rule T6)

In compliance with the official JAI 2026 Hackathon Rule Book:
- **AI Systems Used:** Anthropic Claude 3.5/3.7 / Google Gemini (via Antigravity IDE).
- **Scope of AI Assistance:**
  1. Automated test suite generation for edge-case coverage (`tests/test_context_agent.py`, `tests/test_quality.py`, `tests/test_submit.py`).
  2. Synthesizing inter-agent contract documentation (`docs/agent_contracts.md`, `docs/decision_policy.md`).
  3. Designing UI component layout in Streamlit (`app.py`).
- **Human Verification:** All core clinical rules, safety veto conditions, demographic boundary policies, and architectural contracts were designed, reviewed, and finalized by the project engineering team.

---

## 6. Quickstart & Installation

### Environment Setup
Clone the repository and install dependencies declared in `requirements.txt`:
```bash
git clone https://github.com/aashiexe-101/Agentic.git
cd Agentic
pip install -r requirements.txt
```

> **Note on Python 3.14 Environment:** PyTorch and scikit-learn are required for Agent A2 model inference. If running in a Python 3.14 environment on Windows where pre-built PyTorch binaries are not yet published, the core rule-based agents (A1, A3, A4, A5, A6) and the Streamlit dashboard will execute gracefully with diagnostic model outputs transparently marked as `AWAITING WEIGHTS`.

### 1. Launch Interactive Clinical Screening Dashboard (Streamlit)
```bash
streamlit run app.py
```
- Upload dermatoscopic images (`.jpg`, `.jpeg`, `.png`) or test on sample validation images.
- Input patient demographic context (age, sex, anatomical lesion site).
- Inspect real-time multi-agent assessments (A1 Quality, A4 Context, A3 Morphology, A5 Safety, A6 Coordinator).
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

### 6. Run Complete Automated Test Suite
```bash
pytest tests/
```
Or execute standalone test modules:
```bash
python -m pytest tests/test_context_agent.py tests/test_data.py tests/test_frontend.py tests/test_submit.py -v
```

---

## 7. Data Contracts & Output Formats

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

## 8. Known Limitations, Disclosure of Synthetic Data, & Clinical Disclaimer

- **Decision-Support Scope:** Agentic DermaGuard is an investigational decision-support research tool developed for the JAI 2026 Hackathon. It is **not** an approved diagnostic medical device and must not replace evaluation by a licensed board-certified dermatologist.
- **Disclosure of Synthetic Validation Metrics (Rule F7 & F8):** The performance file `outputs/fake_val_results.csv` and associated `outputs/metrics.json` are synthetic offline fixtures used to validate the 18-column evaluation engine and visualization dashboard. Genuine end-to-end model accuracy on the real 1,942-image validation cohort is unestablished in the current Python 3.14 Windows test environment due to PyTorch dependency constraints.
- **External Portal Transmission:** The submission client (`src/submit.py`) runs in safe offline mode by default and requires explicit technical lead authorization (`confirm=True`) and valid API credentials before transmitting to any external endpoint.
