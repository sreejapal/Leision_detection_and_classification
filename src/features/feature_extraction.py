"""
Deep feature extraction with DenseNet121 + ResNet50 (transfer learning).

Both networks are used as frozen, ImageNet-pretrained feature extractors.
Their penultimate-layer (pre-classification) activations are concatenated
into a single feature vector per image, matching the methodology described
in the project report (Section VI.4 / Algorithm 1).

Output: features.npy (N, D) and labels.npy (N,) saved to --out_dir, where
D = 1024 (DenseNet121) + 2048 (ResNet50) = 3072.

Usage
-----
    python feature_extraction.py --image_dir data/masked --out_dir data \
        --labels_csv data/raw/labels.csv
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class LesionImageDataset(Dataset):
    """Loads images from a directory and looks up their integer label
    from a (filename -> label) mapping."""

    def __init__(self, image_dir: str, filename_to_label: dict, image_size: int = 224):
        self.image_dir = Path(image_dir)
        self.items = sorted(filename_to_label.items())
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        filename, label = self.items[idx]
        image = Image.open(self.image_dir / filename).convert("RGB")
        return self.transform(image), label


def build_feature_extractors(device: torch.device):
    """Returns (densenet_backbone, resnet_backbone), both frozen and in eval mode."""
    densenet = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    densenet.classifier = nn.Identity()  # -> 1024-d pooled feature

    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    resnet.fc = nn.Identity()  # -> 2048-d pooled feature

    for model in (densenet, resnet):
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        model.to(device)

    return densenet, resnet


@torch.no_grad()
def extract_features(loader: DataLoader, densenet, resnet, device) -> tuple[np.ndarray, np.ndarray]:
    all_features, all_labels = [], []

    for images, labels in loader:
        images = images.to(device)

        dense_feat = densenet(images)          # (B, 1024)
        resnet_feat = resnet(images)            # (B, 2048)
        combined = torch.cat([dense_feat, resnet_feat], dim=1)  # (B, 3072)

        all_features.append(combined.cpu().numpy())
        all_labels.append(np.asarray(labels))

    return np.concatenate(all_features), np.concatenate(all_labels)


def load_label_map(labels_csv: str) -> dict:
    """CSV with columns: filename,label (label = integer class index)."""
    import csv

    mapping = {}
    with open(labels_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["filename"]] = int(row["label"])
    return mapping


def main():
    parser = argparse.ArgumentParser(description="DenseNet121 + ResNet50 feature extraction")
    parser.add_argument("--image_dir", required=True, help="Directory of (masked) lesion images")
    parser.add_argument("--labels_csv", required=True, help="CSV with filename,label columns")
    parser.add_argument("--out_dir", required=True, help="Where to save features.npy / labels.npy")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefix", default="", help="Optional output filename prefix, e.g. 'test_'")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    label_map = load_label_map(args.labels_csv)
    dataset = LesionImageDataset(args.image_dir, label_map)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    densenet, resnet = build_feature_extractors(device)
    features, labels = extract_features(loader, densenet, resnet, device)

    print("Feature matrix shape:", features.shape)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{args.prefix}features.npy", features)
    np.save(out_dir / f"{args.prefix}labels.npy", labels)

    print(f"✅ Saved {args.prefix}features.npy and {args.prefix}labels.npy to {out_dir}")


if __name__ == "__main__":
    main()
