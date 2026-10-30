# HAM10000 Dataset Documentation — Agentic DermaGuard

- **Project:** Agentic DermaGuard Healthcare Hackathon
- **Module:** Module 1D — Dataset Documentation
- **Author:** Aashi (Data Pipeline Owner)
- **Last Verified:** 2026-10-30
  **Status:** Verified & Documented

---

## 1. Dataset Overview

The dataset used in the **Agentic DermaGuard** project is **HAM10000** ("Human Against Machine with 10000 training images"), as identified in the supplied dataset documentation. It is a benchmark collection of multi-source dermatoscopic images of common pigmented skin lesions.

The dataset is partitioned into two distinct subsets for training and development:

- **Training partition (`train`):** Exactly **7,762** images and corresponding metadata records.
- **Validation partition (`val`):** Exactly **1,942** images and corresponding metadata records.
- **Total images in repository:** **9,704** images.

All records belong to one of seven dermatological diagnostic classes.

---

## 2. Directory Structure

The extracted dataset adheres to the following root-level directory layout inside the repository:

```
data/
├── train/
│   ├── metadata.csv       # Tabular clinical metadata and labels (7,762 rows)
│   └── images/            # Directory containing 7,762 JPEG image files
└── val/
    ├── metadata.csv       # Tabular clinical metadata and labels (1,942 rows)
    └── images/            # Directory containing 1,942 JPEG image files
```

- Each image file is located directly inside its respective `images/` folder without subfolder nesting.
- No root-level extraneous files (such as archive README or manifest files) exist inside the `data/` directory.
- `data/` is strictly excluded from Git version control via `.gitignore`.

---

## 3. Image Characteristics

All images across both training and validation sets share uniform visual and technical characteristics:

- **File Format:** JPEG (`.jpg`)
- **Native Dimensions:** **600 × 450 pixels** (Width × Height)
- **Aspect Ratio:** 4:3 landscape orientation
- **Color Space:** 3-channel **RGB**
- **File Naming:** Image filenames match metadata `image_id` values with a `.jpg` extension (e.g., `ISIC_0027419.jpg`).

---

## 4. Metadata Schema

Both `data/train/metadata.csv` and `data/val/metadata.csv` share an identical 6-column schema:

| Column Name    | Data Type       | Nullable? | Description                                                  | Example        |
| :------------- | :-------------- | :-------- | :----------------------------------------------------------- | :------------- |
| `image_id`     | String          | No        | Unique ISIC image identifier                                 | `ISIC_0027419` |
| `lesion_id`    | String          | No        | Lesion group identifier (multiple images can share a lesion) | `HAM_0000118`  |
| `dx`           | String          | No        | Diagnostic ground truth label (lowercase)                    | `bkl`          |
| `age`          | Float / Numeric | **Yes**   | Patient age in years                                         | `80.0`         |
| `sex`          | String          | No        | Recorded sex value (`male`, `female`, `unknown`)             | `male`         |
| `localization` | String          | No        | Anatomical site of the lesion on the body                    | `scalp`        |

### Missing Values in Metadata

- **`age` column missingness:**
  - **Training set:** **43** missing values (empty string / unrecorded).
  - **Validation set:** **12** missing values (empty string / unrecorded).
- **Other columns:** All other fields (`image_id`, `lesion_id`, `dx`, `sex`, `localization`) have **0 missing values** across both partitions.

---

## 5. Class Distribution

The dataset covers seven diagnostic categories. In the raw dataset and throughout the codebase, all labels are represented in **strictly lowercase** characters.

Canonical class order:

```python
CLASS_LABELS = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
```

### Diagnostic Definitions

1. **`akiec`**: Actinic keratoses and intraepithelial carcinoma (pre-cancerous / early-stage squamous lesion)
2. **`bcc`**: Basal cell carcinoma (common non-melanoma skin cancer)
3. **`bkl`**: Benign keratosis-like lesions (solar lentigines, seborrheic keratoses, lichen-planus like keratoses)
4. **`df`**: Dermatofibroma (benign cutaneous fibrous lesion)
5. **`mel`**: Melanoma (malignant melanocytic neoplasm)
6. **`nv`**: Melanocytic nevi (common benign moles)
7. **`vasc`**: Vascular lesions (angiomas, angiokeratomas, pyogenic granulomas, hemorrhage)

### Training Partition Distribution (`data/train/metadata.csv`)

| Class Label | Index | Sample Count | Percentage  | Diagnosis Full Name           |
| :---------- | :---: | :----------: | :---------: | :---------------------------- |
| `nv`        |   5   |    5,212     |   67.15%    | Melanocytic nevi              |
| `mel`       |   4   |     856      |   11.03%    | Melanoma                      |
| `bkl`       |   2   |     848      |   10.93%    | Benign keratosis-like lesions |
| `bcc`       |   1   |     400      |    5.15%    | Basal cell carcinoma          |
| `akiec`     |   0   |     249      |    3.21%    | Actinic keratoses             |
| `vasc`      |   6   |     109      |    1.40%    | Vascular lesions              |
| `df`        |   3   |      88      |    1.13%    | Dermatofibroma                |
| **Total**   |   —   |  **7,762**   | **100.00%** |                               |

### Validation Partition Distribution (`data/val/metadata.csv`)

| Class Label | Index | Sample Count | Percentage  | Diagnosis Full Name           |
| :---------- | :---: | :----------: | :---------: | :---------------------------- |
| `nv`        |   5   |    1,302     |   67.04%    | Melanocytic nevi              |
| `mel`       |   4   |     214      |   11.02%    | Melanoma                      |
| `bkl`       |   2   |     212      |   10.92%    | Benign keratosis-like lesions |
| `bcc`       |   1   |     101      |    5.20%    | Basal cell carcinoma          |
| `akiec`     |   0   |      63      |    3.24%    | Actinic keratoses             |
| `vasc`      |   6   |      27      |    1.39%    | Vascular lesions              |
| `df`        |   3   |      23      |    1.18%    | Dermatofibroma                |
| **Total**   |   —   |  **1,942**   | **100.00%** |                               |

### Combined Class Comparison

| Class Label | Training Count | Validation Count | Combined Total | Combined Percentage |
| :---------- | :------------: | :--------------: | :------------: | :-----------------: |
| `nv`        |     5,212      |      1,302       |     6,514      |       67.13%        |
| `mel`       |      856       |       214        |     1,070      |       11.03%        |
| `bkl`       |      848       |       212        |     1,060      |       10.92%        |
| `bcc`       |      400       |       101        |      501       |        5.16%        |
| `akiec`     |      249       |        63        |      312       |        3.22%        |
| `vasc`      |      109       |        27        |      136       |        1.40%        |
| `df`        |       88       |        23        |      111       |        1.14%        |
| **Total**   |   **7,762**    |    **1,942**     |   **9,704**    |     **100.00%**     |

---

## 6. Data Validation Checks

Prior to development, a comprehensive read-only dataset verification audit was performed, confirming:

1. **Exact Row and File Counts:**
   - Training metadata contains exactly 7,762 rows matching 7,762 JPEG files on disk.
   - Validation metadata contains exactly 1,942 rows matching 1,942 JPEG files on disk.
2. **Zero Missing or Orphaned Files:**
   - No image files referenced in `metadata.csv` are missing from disk (0 missing).
   - No unindexed or stray images exist in either `images/` directory (0 unindexed).
3. **Class Presence:**
   - All seven canonical classes are present in both the training and validation partitions.
4. **Label Casing Uniformity:**
   - All ground truth `dx` entries in both CSV files are strictly lowercase string values.
5. **Archive Integrity:**
   - Original archive files on the Desktop remain intact and unaltered in size:
     - `health-training.zip`: 2,153,898,562 bytes
     - `health-eval (1).zip`: 533,958,185 bytes

---

## 7. How `src/data.py` Uses the Dataset

The dataset module (`src/data.py`) provides the official, framework-independent loading interface:

### 1. Path Safety & Verification

- `validate_image_id(image_id)` enforces strict validation:
  - Rejects non-string inputs (integers, floats, `None`, `NaN`) with `ValueError`.
  - Rejects empty strings, whitespace, path traversal tokens (`..`), path separators (`/`, `\`), drive colons, and null bytes.
  - Safely handles legitimate IDs with or without `.jpg` / `.jpeg` extensions.
- All resolved paths are checked with `Path.relative_to` to guarantee they do not escape the designated partition directory.

### 2. Metadata Ingestion (`load_metadata`)

- `load_metadata(split="train" | "val", data_dir="data")` loads `metadata.csv` using pandas.
- Validates the required 6-column schema.
- Verifies that all `dx` values match `CLASS_LABELS`.
- Adds a verified `image_path` column pointing directly to disk files.

### 3. Dataset Abstraction (`DermaGuardDataset`)

- Framework-independent dataset class supporting indexing (`dataset[i]`) and iteration (`for item in dataset`).
- **Raw Image Loading:** Loads images via Pillow (`PIL.Image.open`), converting them to 3-channel RGB.
- **Preserves Native Dimensions:** Retains original **600 × 450** resolution.
- **Separation of Concerns:** Does **NOT** resize, normalize, or extract features. Model-specific transforms (e.g. 224 × 224 resolution and ImageNet mean/std normalization) are the responsibility of the downstream diagnosis model.
- **Missing Age Normalization:** Cleanly normalizes missing `age` entries to `None` while converting valid values to `float`.
- **Injected Metadata Validation:** Strictly verifies externally passed DataFrames (checking columns, labels, and matching image filenames).

---

## 8. Known Limitations

When working with this dataset, developers and researchers should be aware of several fundamental constraints:

1. **Severe Class Imbalance:**
   - Melanocytic nevi (`nv`) make up over **67%** of the entire dataset.
   - In contrast, Dermatofibroma (`df`) represents only **1.13%** (88 training images) and Vascular lesions (`vasc`) only **1.40%** (109 training images).
   - Unweighted models risk achieving high accuracy by simply predicting the majority class (`nv`) while failing on critical malignant categories like Melanoma (`mel`).
2. **Missing Clinical Metadata:**
   - Patient `age` is unrecorded for 43 training cases and 12 validation cases. Any multimodal model incorporating demographic variables must implement imputation or handle `None` values.
3. **Lesion-Level Separation Unverified Independently:**
   - The supplied dataset pack documentation states that splits are lesion-disjoint (no `lesion_id` is shared between training and validation). However, this property was not independently verified by comparing lesion ID sets during this audit.
4. **Clinical Efficacy Not Established:**
   - The dataset and any models trained on it are intended for hackathon benchmarking and research development only. Clinical accuracy, diagnostic safety, and generalization to diverse real-world patient demographics have **not** been established.
5. **No Provenance or Licensing Claims:**
   - No independent claims are made regarding original patient consent, clinical collection provenance, or third-party intellectual property licenses beyond what was provided in the official participant pack.

---

## 9. Important Notes for Model Development

1. **Class Weighting & Loss Functions:**
   - Address severe class imbalance by applying class weights, focal loss, or balanced mini-batch sampling. Recommended inverse-frequency weights should account for the ~59:1 ratio between `nv` and `df`.
2. **Preprocessing Pipeline (Model-Specific):**
   - Because `src/data.py` outputs raw 600 × 450 RGB images, vision backbones (e.g., EfficientNet-B0) must include an initial preprocessing stage:
     - Resize / crop to **224 × 224 pixels**.
     - Normalize using ImageNet standard constants: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`.
3. **Canonical Label Indexing:**
   - Always reference `CLASS_LABELS` and `LABEL_TO_IDX` from `src/data.py` to maintain consistent output probability ordering across all modules:
     ```python
     from src.data import CLASS_LABELS, LABEL_TO_IDX, IDX_TO_LABEL
     # Index mapping:
     # 0: akiec, 1: bcc, 2: bkl, 3: df, 4: mel, 5: nv, 6: vasc
     ```
4. **Patient Demographic Features:**
   - If utilizing `sex`, `age`, or `localization` in multi-input architectures, ensure categorical encoding accounts for `'unknown'` values and numerical imputers handle `age = None`.

---

_Documentation compiled and verified by Aashi (Agentic DermaGuard team)._
