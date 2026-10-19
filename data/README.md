# Data

This project uses the **ISIC 2019** dermoscopic image dataset from the
International Skin Imaging Collaboration.

- Challenge page / download: https://challenge.isic-archive.com/data/
- Citation: Codella et al., *"Skin Lesion Analysis Toward Melanoma Detection 2019"*, ISIC, 2019.

Raw images and the extracted feature arrays are **not** committed to this
repository (the feature arrays alone are >150 MB). To reproduce the pipeline:

1. Download the ISIC 2019 training images and ground-truth CSV from the link
   above and place them under `data/raw/`.
2. Run the masking step to isolate lesion regions:
   ```bash
   python src/preprocessing/lesion_masking.py --input_dir data/raw --output_dir data/masked
   ```
3. Run feature extraction to produce `features.npy` / `labels.npy`
   (and, if you hold out a test split, `test_features.npy` / `test_labels.npy`):
   ```bash
   python src/features/feature_extraction.py --image_dir data/masked --out_dir data
   ```

Expected layout after step 3:

```
data/
├── features.npy        # (N, D) deep feature vectors (DenseNet121 + ResNet50, concatenated)
├── labels.npy           # (N,) integer class labels
├── test_features.npy     # held-out test features
└── test_labels.npy       # held-out test labels
```

These four arrays are what every script under `src/models/` and
`src/training/` expects to load.
