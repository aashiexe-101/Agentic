# Training and Evaluation Dataset Audit Report

**Project:** Agentic DermaGuard — Track 2 (Healthcare), Jaypee Agentic AI Summit 2026  
**Technical Lead:** Ilka Mansoori  
**Audit Date:** 2026-10-30  
**Verification Method:** Programmatic audit via Python, PIL, and Pandas over all physical disk files  

---

## 1. Executive Summary

This audit establishes the baseline truth of the datasets available for **Agent A2 (Diagnosis)** training and evaluation. All claims in this document have been verified programmatically against the physical files on disk in `/Users/ilkamansoori/Desktop/Agentic`.

| Metric | Training Set (`data/train`) | Evaluation Set (`data/eval`) | Overlap / Anomaly |
| :--- | :---: | :---: | :---: |
| **Total Images in Metadata** | 7,762 | 1,942 | 0 missing |
| **Total Physical JPEG Files** | 7,762 | 1,942 | 0 missing |
| **Corrupted / Unreadable Files** | 0 | 0 | 0 (All verified via PIL) |
| **Unique `image_id` Count** | 7,762 | 1,942 | 0 duplicates |
| **Unique `lesion_id` Count** | 5,815 | 1,456 | **0 overlap** (strictly disjoint) |
| **Shared Image IDs** | — | — | **0 shared images** |

---

## 2. Dataset Reconciliation (7,815 / 2,000 vs 7,762 / 1,942)

The hackathon documentation estimated roughly ~7,815 training images and ~2,000 validation images.  
The actual physical files in the participant release contain exactly **7,762 training images** and **1,942 evaluation images**.

### Proof of Dataset Integrity
1. **Manifest Verification:**  
   `data/train/manifest.json` specifies exactly 7,762 entries.  
   `data/eval/manifest.json` specifies exactly 1,942 entries.
2. **SHA256 Checksums:**  
   - `data/train/metadata.csv`: `5d5bdff86628882203b8e250d9c5e4aa13839456a9da731aaf52724b1cf04355`
   - `data/eval/metadata.csv`: `1a7ab2121e3a36d6fa1ee1298e9a313607dde9509f680958a83a59a34f9cc12b`
3. **Partition Boundary Integrity:**  
   As explained in `data/train/README.md`, the organizers withheld images sharing a `lesion_id` with the 200 hidden Day 2 evaluation images to prevent cross-partition leakage. The resulting counts (7,762 + 1,942 = 9,704 non-hidden images) match the physical files exactly.

---

## 3. Metadata Structure and Schema

Both `data/train/metadata.csv` and `data/eval/metadata.csv` share an identical 6-column schema:

| Column Name | Data Type | Null Count (Train) | Null Count (Eval) | Description |
| :--- | :--- | :---: | :---: | :--- |
| `image_id` | String | 0 | 0 | Unique ISIC identifier (e.g., `ISIC_0024306`) |
| `lesion_id` | String | 0 | 0 | Identifier for the biological lesion (multiple photos per lesion) |
| `dx` | String | 0 | 0 | Diagnostic label in `CLASS_NAMES` (Ground truth) |
| `age` | Float64 | 45 | 12 | Patient age at diagnosis |
| `sex` | String | 45 | 12 | Patient biological sex (`male`, `female`, `unknown`) |
| `localization`| String | 0 | 0 | Anatomical site (e.g., `back`, `lower extremity`, `face`) |

> [!WARNING]
> Ground-truth columns `dx` (and any related demographic priors if unobserved at runtime) must never be accessed by Agent A2 during inference.

---

## 4. Class Distribution Comparison

The 7 diagnostic classes represent standard HAM10000 / ISIC categories:
- `nv`: Melanocytic nevi (benign)
- `mel`: Melanoma (malignant)
- `bkl`: Benign keratosis-like lesions
- `bcc`: Basal cell carcinoma (malignant)
- `akiec`: Actinic keratoses / intraepithelial carcinoma (pre-malignant)
- `vasc`: Vascular lesions (benign)
- `df`: Dermatofibroma (benign)

### Class Distribution Table
| Class | Train Count | Train % | Eval Count | Eval % | Imbalance Ratio (Train) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `nv` | 5,212 | 67.15% | 1,302 | 67.04% | 1.00x (Majority) |
| `mel` | 856 | 11.03% | 214 | 11.02% | 6.09x minority |
| `bkl` | 848 | 10.92% | 212 | 10.92% | 6.15x minority |
| `bcc` | 400 | 5.15% | 101 | 5.20% | 13.03x minority |
| `akiec`| 249 | 3.21% | 63 | 3.24% | 20.93x minority |
| `vasc` | 109 | 1.40% | 27 | 1.39% | 47.82x minority |
| `df` | 88 | 1.13% | 23 | 1.18% | 59.23x minority |
| **Total** | **7,762** | **100.0%** | **1,942** | **100.0%** | — |

The class distributions of `data/train` and `data/eval` are virtually identical in proportion, confirming that the evaluation set is an unbiased stratified sample of the population.

---

## 5. Software Environment and Compute Acceleration

- **Python:** 3.14.2
- **PyTorch:** 2.15.0
- **Torchvision:** 0.29.2
- **Scikit-Learn:** 1.9.1
- **Hardware Acceleration:** Apple Silicon MPS (`torch.backends.mps.is_available() == True`), CUDA: False
- **PIL / Pillow:** 13.0.0
