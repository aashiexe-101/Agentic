# Agent A2 Model Experiment and Transfer Learning Plan

**Project:** Agentic DermaGuard — Track 2 (Healthcare), Jaypee Agentic AI Summit 2026  
**Technical Lead:** Ilka Mansoori  
**Benchmark Target:** 7-Class Skin Lesion Classification (HAM10000 / ISIC Standard)  

---

## 1. Scientific Background and Literature Foundation

### 1.1 The HAM10000 Benchmark
The HAM10000 dataset (Tschandl et al., 2018, *Scientific Data*, DOI: [10.1038/sdata.2018.161](https://doi.org/10.1038/sdata.2018.161)) is the standard benchmark for dermatoscopic lesion analysis. Key characteristics documented in literature:
1. **Severe Imbalance:** Melanocytic nevi (`nv`) account for ~67% of cases, while critical malignancies like melanoma (`mel`, 11%) and rare benign entities like dermatofibroma (`df`, 1.1%) are heavily under-represented.
2. **Biological Lesion Clustering:** A single patient lesion can have multiple dermoscopic acquisitions. Standard random splitting causes massive data leakage; **lesion-grouped splitting** is scientifically mandatory.
3. **Clinical Priority (Melanoma Sensitivity):** From a triage perspective, missing a malignant melanoma (`mel`) has severe clinical consequences compared to a false alarm on a benign nevus (`nv`).

### 1.2 Deep Transfer Learning with EfficientNet
EfficientNet (Tan & Le, 2019) utilizes compound scaling across depth, width, and resolution. Pretrained ImageNet representations capture fundamental low-level dermatologic features (pigment networks, globules, vascular patterns, blue-white veils). In dermatologic AI research (e.g., Srinivasu et al., 2021; Gessert et al., 2020), EfficientNet-B0 achieves competitive diagnostic accuracy while requiring only 5.3 million parameters and ~0.39 GFLOPs, making it ideally suited for real-time edge screening and Streamlit deployment.

---

## 2. Experimental Hypotheses and Controlled Architectures

We design a controlled multi-phase study across 5 core dimensions:

### Approach A: Frozen EfficientNet-B0 Backbone + Linear/Logistic Head
- **Architecture:** Pretrained EfficientNet-B0 acts as an immutable feature extractor ( \in \mathbb{R}^{1280}$).
- **Classifier:** Multinomial Logistic Regression optimized via L-BFGS with L2 regularization.
- **Hypothesis:** High convex stability, zero risk of catastrophic forgetting in the backbone, fast convergence (< 2s), and highly calibrated class probabilities.

### Approach B: Frozen Backbone + Neural MLP Classification Head
- **Architecture:** Pretrained EfficientNet-B0 ($) $ightarrow$ Linear(, 256$) $ightarrow$ BatchNorm1d $ightarrow$ ReLU $ightarrow$ Dropout(/bin/zsh.3$) $ightarrow$ Linear(, 7$).
- **Hypothesis:** Allows non-linear decision boundaries between difficult visual overlaps (such as `bkl` vs `mel` vs `nv`) while retaining frozen feature stability.

### Approach C: Partial Fine-Tuning of Later EfficientNet Stages
- **Architecture:** Unfreeze stages 6, 7, and the head of EfficientNet-B0; freeze earlier stages (1–5) capturing low-level edges.
- **Differential Learning Rates:** Backbone $\eta = 10^{-5}$, Head $\eta = 10^{-3}$, AdamW optimizer, cosine annealing schedule.
- **Hypothesis:** May adapt high-level feature filters to dermoscopy-specific color gradients, but risks overfitting on majority class `nv` given 7,762 images.

### Approach D: Loss Formulations for Extreme Class Imbalance
1. **Unweighted Cross-Entropy:** Standard loss, prone to predicting majority class `nv`.
2. **Inverse-Frequency Balanced Loss:** Class weight  = rac{N}{K \cdot N_c}$, penalizing errors on rare classes.
3. **Melanoma-Boosted Loss:** Class weights with explicit recall boost for `mel` (.5	imes, 2.0	imes, 3.0	imes$).
4. **Focal Loss (Lin et al., 2017):** $	ext{FL}(p_t) = -lpha_t (1 - p_t)^\gamma \log(p_t)$ with $\gamma = 2.0$ to down-weight easy nevus examples.

### Approach E: Training-Only Augmentation
- Apply stochastic transforms during training:
  - Random Horizontal Flip (=0.5$)
  - Random Vertical Flip (=0.5$)
  - Random Rotation ($\pm 20^\circ$)
  - Color Jitter (Brightness $\pm 0.1$, Contrast $\pm 0.1$, Saturation $\pm 0.1$)
- Deterministic transforms during validation and inference:
  - Resize(, 224$), ToTensor(), Normalize(ImageNet mean & std).

---

## 3. Evaluation Protocol and Selection Criteria

### Lesion-Grouped Split Protocol
- The 7,762 training set is partitioned using `GroupShuffleSplit` on `lesion_id` (\% / 20\%$, `random_state=42`).
- Internal Train: 6,209 images. Internal Val: 1,553 images. Exactly 0 overlapping lesions.
- The official 1,942-image evaluation set is strictly held out.

### Multi-Metric Decision Matrix
A candidate model will not be selected on raw Accuracy alone. Model selection follows:
67828	ext{Score} = 0.35 	imes 	ext{Balanced Accuracy} + 0.35 	imes 	ext{Macro F1} + 0.20 	imes 	ext{Melanoma Recall} + 0.10 	imes 	ext{Overall Accuracy}67828
Requirements:
1. Balanced Accuracy $> 55\%$ (surpassing naive majority baseline of .28\%$ balanced).
2. Melanoma Recall $\ge 50\%$.
3. Sub-second inference latency on standard hardware.
