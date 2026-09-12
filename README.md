# 🩺 Automated Skin Lesion Detection & Classification

**A hybrid deep learning + classical ML pipeline for multi-class skin lesion classification on the ISIC 2019 dataset, with a focus on reducing skin-tone bias and improving minority-class recall.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-DL%20Backbone-ee4c2c)](https://pytorch.org/)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-SVM%20%2B%20PCA-f7931e)](https://scikit-learn.org/)
[![Dataset](https://img.shields.io/badge/Dataset-ISIC%202019-lightgrey)](https://challenge.isic-archive.com/data/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> IVᵗʰ Semester Minor Project — Department of Data Science & Artificial Intelligence, IIIT Naya Raipur
> **Authors:** Sreeja Pal, Patnana Jagan Mohan · **Supervisor:** Dr. Avantika Singh

---

## 📌 Overview

Manual dermoscopic examination is slow and subjective. This project builds an automated pipeline that classifies skin lesions from dermoscopic images into **8 diagnostic categories** (melanoma, melanocytic nevus, basal cell carcinoma, benign keratosis, actinic keratosis, squamous cell carcinoma, vascular lesion, dermatofibroma) using the **ISIC 2019** benchmark dataset.

Rather than training a single end-to-end CNN, the project takes a **hybrid feature-based approach**: deep pretrained backbones are used purely as feature extractors, and the resulting embeddings are refined and classified with lighter, more interpretable classical-ML and metric-learning components. This keeps the pipeline computationally cheap to iterate on while still leveraging transfer learning.

### Key challenges tackled

| Challenge | Approach |
|---|---|
| **Skin-tone bias** — models latching onto surrounding skin color instead of the lesion | HSV / CIELAB color-space lesion masking to isolate the lesion region before feature extraction |
| **Severe class imbalance** — majority classes (NV, UNK) dwarf rare ones (DF, VASC, SCC) | Weighted sampling, class-balanced losses, and a custom low-rank "θ(t) decomposition" head |
| **High inter-class visual similarity** | Quadruplet-loss metric learning to pull same-class embeddings together and push dissimilar/similar-looking classes apart |
| **High-dimensional deep features** | Incremental PCA for dimensionality reduction before classification |

---

## 🧠 Pipeline

```
Dermoscopic Image (ISIC 2019)
        │
        ▼
┌───────────────────────┐
│  1. Preprocessing      │  resize, normalize, HSV/LAB lesion masking
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│  2. Feature Extraction │  DenseNet121 + ResNet50 (pretrained, transfer learning)
│                        │  → concatenated deep feature vector per image
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│  3. Dimensionality      │  Incremental PCA
│     Reduction           │
└───────────┬───────────┘
            ▼
      ┌─────┴─────┐
      ▼           ▼
┌───────────┐ ┌─────────────────────────┐
│  SVM       │ │ Metric-learned embedding │
│  baseline  │ │ (Triplet / Quadruplet    │
│  (RBF)     │ │  loss / low-rank class   │
│            │ │  decomposition) + SVM    │
└───────────┘ └─────────────────────────┘
      │           │
      └─────┬─────┘
            ▼
┌───────────────────────┐
│  4. Evaluation          │  Accuracy, Precision/Recall/F1, Confusion Matrix,
│                        │  t-SNE / UMAP embedding visualization
└───────────────────────┘
```

---

## 📊 Results

Three classification heads were benchmarked on the held-out test split of extracted ISIC 2019 features (8 classes, 5,067 test samples). Baseline and triplet numbers are from the final project report / presentation (`docs/project_report.pdf`, `docs/project_presentation.pptx`); quadruplet numbers are from a supplementary run.

| Model | Accuracy | Macro F1 | Weighted F1 | Notes |
|---|---:|---:|---:|---|
| **SVM baseline** (PCA-reduced deep features) | **75.6%** | 0.62 | 0.75 | Strong majority-class recall, weaker on rare classes |
| **SVM + Triplet-loss embeddings** | 69.2% | 0.52 | 0.70 | Better minority-class recall and tighter UMAP clustering, but margin too small to beat raw-feature SVM outright |
| **SVM + Quadruplet-loss embeddings** | 56.7% | 0.44 | 0.60 | Lowest overall accuracy of the three — the extra negative constraint over-regularized the embedding space rather than helping (see below) |

**Baseline SVM — per-class report**

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 0 | 0.68 | 0.66 | 0.67 | 904 |
| 1 | 0.83 | 0.90 | 0.86 | 2575 |
| 2 | 0.66 | 0.74 | 0.70 | 665 |
| 3 | 0.48 | 0.25 | 0.33 | 173 |
| 4 | 0.67 | 0.53 | 0.59 | 525 |
| 5 | 0.94 | 0.35 | 0.52 | 48 |
| 6 | 0.95 | 0.82 | 0.88 | 51 |
| 7 | 0.63 | 0.33 | 0.43 | 126 |

**SVM + Triplet-loss embeddings — per-class report**

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 0 | 0.60 | 0.62 | 0.61 | 904 |
| 1 | 0.86 | 0.81 | 0.84 | 2575 |
| 2 | 0.66 | 0.65 | 0.66 | 665 |
| 3 | 0.23 | 0.38 | 0.29 | 173 |
| 4 | 0.55 | 0.52 | 0.53 | 525 |
| 5 | 0.16 | 0.40 | 0.23 | 48 |
| 6 | 0.72 | 0.84 | 0.77 | 51 |
| 7 | 0.27 | 0.20 | 0.23 | 126 |

**SVM + Quadruplet-loss embeddings — per-class report**

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 0 | 0.50 | 0.48 | 0.49 | 904 |
| 1 | 0.87 | 0.66 | 0.75 | 2575 |
| 2 | 0.60 | 0.49 | 0.54 | 665 |
| 3 | 0.16 | 0.51 | 0.25 | 173 |
| 4 | 0.33 | 0.41 | 0.36 | 525 |
| 5 | 0.19 | 0.40 | 0.25 | 48 |
| 6 | 0.51 | 0.76 | 0.61 | 51 |
| 7 | 0.17 | 0.43 | 0.24 | 126 |

*(Class indices correspond to the 8 ISIC 2019 diagnostic categories used in this study — see [`docs/project_report.pdf`](docs/project_report.pdf) for the full mapping and class-distribution chart.)*

### Takeaways

- Deep CNN embeddings (DenseNet121 + ResNet50) capture lesion texture/color/structure well enough for a plain RBF-SVM to reach **75.6%** accuracy.
- Incremental PCA keeps the pipeline computationally light without a meaningful accuracy hit.
- Triplet-based metric learning **does** improve recall on a couple of minority classes (e.g. class 5: recall 0.35 → 0.40) and produces visibly tighter clusters in UMAP space — but the learned embedding margins were too small relative to raw feature separability, so it trades majority-class accuracy for partial minority-class gains rather than winning outright (69.2% vs. 75.6% overall).
- Quadruplet-loss embeddings pushed minority-class **recall** up further still (e.g. class 3: 0.25 → 0.51, class 7: 0.33 → 0.43) — the extra negative constraint does what it's meant to do for rare classes — but at a steep cost to majority-class precision and to overall accuracy (56.7%, the lowest of the three). Net effect: recall gains on rare classes don't offset the accuracy lost everywhere else, at least with this margin/mining setup.
- **Class imbalance remains the dominant bottleneck** across all three models — rare classes (support < 200, e.g. classes 3, 5, 7) trail the majority classes by a wide F1 margin regardless of which classifier head is used, and no single approach here fully solves it.
- Ranking by overall accuracy: **baseline SVM (75.6%) > triplet (69.2%) > quadruplet (56.7%)**. Ranking by minority-class recall tends to favor quadruplet on several classes — which head is "best" depends on whether overall accuracy or rare-class recall matters more for the use case.

📄 Full write-up, literature review, and figures: [`docs/project_report.pdf`](docs/project_report.pdf) · 🖼️ Slide deck: [`docs/project_presentation.pptx`](docs/project_presentation.pptx)

---

## 📁 Repository Structure

```
.
├── src/
│   ├── preprocessing/
│   │   └── lesion_masking.py       # HSV / CIELAB lesion segmentation & masking
│   ├── features/
│   │   └── feature_extraction.py   # DenseNet121 + ResNet50 feature extraction
│   ├── models/
│   │   ├── low_rank_decomposition.py  # Low-rank "θ(t)" imbalance-aware classifier
│   │   ├── quadruplet_net.py          # Quadruplet-loss metric-learning embedding net
│   │   └── triplet_net.py             # Triplet-loss metric-learning embedding net (best metric-learning result)
│   └── training/
│       ├── train_svm_baseline.py   # PCA + SVM baseline, with 2D decision-region plot
│       └── evaluate.py             # Shared evaluation / reporting utilities
├── notebooks/
│   ├── 01_quadruplet_metric_learning.ipynb
│   ├── 02_low_rank_class_decomposition.ipynb
│   └── 03_triplet_metric_learning.ipynb
├── results/
│   ├── results.json                # Raw classification report (baseline SVM run)
│   └── metrics_summary.md
├── docs/
│   ├── project_report.pdf
│   └── project_presentation.pptx
├── data/
│   └── README.md                   # How to obtain / regenerate features.npy, labels.npy
├── requirements.txt
└── LICENSE
```

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
git clone https://github.com/<your-username>/skin-lesion-classification.git
cd skin-lesion-classification
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Get the data

This repo does **not** ship the raw ISIC 2019 images or the extracted `.npy` feature files (too large for Git). See [`data/README.md`](data/README.md) for download links and instructions to regenerate `features.npy` / `labels.npy`.

### 3. Run the pipeline

```bash
# 1. Mask lesions (reduces skin-tone bias)
python src/preprocessing/lesion_masking.py --input_dir data/raw --output_dir data/masked

# 2. Extract deep features (DenseNet121 + ResNet50)
python src/features/feature_extraction.py --image_dir data/masked --out_dir data

# 3. Train the PCA + SVM baseline
python src/training/train_svm_baseline.py --features data/features.npy --labels data/labels.npy

# 4. (Optional) Explore the metric-learning approaches
jupyter notebook notebooks/01_quadruplet_metric_learning.ipynb
jupyter notebook notebooks/02_low_rank_class_decomposition.ipynb
```

---

## 🔬 Methodology Details

### Preprocessing — lesion masking
Two color-space masking strategies were evaluated to strip out surrounding skin and reduce skin-tone bias: **HSV thresholding** and **CIELAB channel thresholding**, followed by morphological cleanup to produce a binary lesion mask that is applied to the original image before feature extraction.

### Feature extraction
**DenseNet121** and **ResNet50**, both pretrained on ImageNet, are used as frozen feature extractors via transfer learning. Their penultimate-layer activations are concatenated into a single high-dimensional feature vector per image.

### Dimensionality reduction
**Incremental PCA** is applied so the feature vectors can be reduced in mini-batches without holding the full feature matrix in memory — useful given the dataset scale.

### Classification heads explored
1. **RBF-kernel SVM** directly on PCA-reduced features — the strongest, most stable baseline.
2. **Triplet-loss embedding network** (`src/models/triplet_net.py`) — an MLP trained with a hard/semi-hard-mining triplet loss (`d(a,p) + margin < d(a,n)`) to pull same-class embeddings together and push apart the nearest wrong-class sample, followed by an SVM on the learned embedding. **Result: 69.2% accuracy** — see [`notebooks/03_triplet_metric_learning.ipynb`](notebooks/03_triplet_metric_learning.ipynb).
3. **Quadruplet-loss embedding network** (`src/models/quadruplet_net.py`) — extends the triplet setup with a second, unrelated negative (`d(a,p) < d(a,n1)` and `d(a,p) < d(p,n2)` margins) for a stronger separation constraint. **Result: 56.7% accuracy** — lower overall accuracy than both the baseline and triplet, though it lifts recall on several minority classes. See [`notebooks/01_quadruplet_metric_learning.ipynb`](notebooks/01_quadruplet_metric_learning.ipynb).
4. **Low-rank class-imbalance decomposition** (`src/models/low_rank_decomposition.py`) — an experimental linear layer that decomposes each weight matrix into a shared "general" component (`θg`) and a low-rank "task-specific" correction (`θt = B·A`), trained with a sinusoidally-annealed discrepancy loss (`LMORE`) plus a supervised-contrastive term, to study how much of the network's capacity is being spent adapting to majority vs. minority classes.

---

## 🛣️ Future Work

- [ ] Re-tune the quadruplet margin/mining strategy — current setup trades too much overall accuracy for its minority-class recall gains
- [ ] Data augmentation / synthetic minority oversampling for rare lesion classes
- [ ] Meta-learning (e.g. few-shot adaptation) for classes with very few samples
- [ ] Evaluate with balanced metrics (macro-F1, balanced accuracy) as the primary criterion instead of raw accuracy
- [ ] End-to-end fine-tuning of the CNN backbones instead of frozen feature extraction
- [ ] Fairness evaluation across Fitzpatrick skin-tone groups

---

## 📚 Dataset & References

- **Dataset:** ISIC 2019 — Codella et al., *"Skin Lesion Analysis Toward Melanoma Detection 2019: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC)"*, 2019.
- Sultana, Lu, Fan, Yap — *"Selective Alignment Transfer for Domain Adaptation in Skin Lesion Analysis"*, 2025.
- Xu, Duan, Liu, Li, Jiang, Lemmon, Jin, Shi — *"Incorporating Rather Than Eliminating: Achieving Fairness for Skin Disease Diagnosis Through Group-Specific Experts"*, 2025.
- Wei, Chakraborti — *"Achieving Fair Skin Lesion Detection through Skin Tone Normalization and Channel Pruning"*, 2025.

## 📄 License

This project is licensed under the [MIT License](LICENSE).
