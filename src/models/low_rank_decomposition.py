"""
Low-rank "theta(t)" class-imbalance-aware classifier.

Experimental classification head that decomposes each linear layer's weight
matrix into:
  - a shared "general" component  Wg           (theta_g)
  - a low-rank, per-batch-adapted correction  B @ A   (theta_t)

The idea being explored: measure how much of the network's output is coming
from the low-rank "task adaptation" term vs. the general term, per class,
to see whether minority classes rely more heavily on theta_t (i.e. need more
task-specific correction) than majority classes.

Trained with a combination of:
  - a margin cross-entropy loss on the full (theta_g + theta_t) output
  - `LMORE`: a class-prior-weighted discrepancy loss between the full output
    and the theta_g-only output, annealed by a sinusoidal schedule alpha(t)
  - a supervised-contrastive loss on the first layer's features

This is a cleaned-up, importable version of
`notebooks/02_low_rank_class_decomposition.ipynb`. It is an exploratory
technique, not the primary reported result -- see README for the headline
SVM / quadruplet-embedding numbers.

Usage
-----
    python low_rank_decomposition.py --features data/features.npy --labels data/labels.npy --epochs 10
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.append(str(Path(__file__).resolve().parent.parent / "training"))
from evaluate import evaluate_predictions  # noqa: E402


class FeatureDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LowRankLinear(nn.Module):
    """Linear layer y = Wg(x) + x @ (B @ A)^T, where B @ A is a rank-`rank`
    correction to the general weight matrix Wg."""

    def __init__(self, in_features: int, out_features: int, rank: int = 32):
        super().__init__()
        self.Wg = nn.Linear(in_features, out_features)
        self.A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.B = nn.Parameter(torch.randn(out_features, rank) * 0.01)

    def forward(self, x):
        low_rank_delta = self.B @ self.A
        return self.Wg(x) + x @ low_rank_delta.T

    def forward_general(self, x):
        """theta_g-only forward pass (skips the low-rank correction)."""
        return self.Wg(x)


class LowRankClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128, rank: int = 32):
        super().__init__()
        self.layer1 = LowRankLinear(input_dim, hidden_dim, rank=rank)
        self.relu = nn.ReLU()
        self.layer2 = LowRankLinear(hidden_dim, num_classes, rank=rank)

    def forward(self, x):
        return self.layer2(self.relu(self.layer1(x)))

    def forward_general(self, x):
        return self.layer2.forward_general(self.relu(self.layer1.forward_general(x)))


def supcon_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07):
    """Supervised contrastive loss (Khosla et al., 2020), simplified single-view form."""
    features = F.normalize(features, dim=1)
    similarity = torch.matmul(features, features.T) / temperature

    labels = labels.unsqueeze(1)
    mask = labels == labels.T

    exp_sim = torch.exp(similarity)
    log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))

    loss = -(mask * log_prob).sum(dim=1) / mask.sum(dim=1)
    return loss.mean()


def margin_loss(logits: torch.Tensor, labels: torch.Tensor, margin: float = 0.3):
    one_hot = F.one_hot(labels, num_classes=logits.size(1)).float()
    adjusted_logits = logits - margin * one_hot
    return F.cross_entropy(adjusted_logits, labels)


def alpha_schedule(t: int, T: int, amplitude: float = 1.5) -> float:
    """Sinusoidal annealing weight for the LMORE discrepancy term."""
    return amplitude * math.sin(math.pi * t / T)


def make_weighted_loader(X: np.ndarray, y: np.ndarray, batch_size: int = 64) -> DataLoader:
    dataset = FeatureDataset(X, y)
    class_counts = np.bincount(y)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = torch.tensor(class_weights[y], dtype=torch.float32)
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def per_class_theta_effect(model, loader, class_counts, device):
    """Reports, per class, the average squared difference between the full
    (theta_g + theta_t) output and the theta_g-only output -- i.e. how much
    the low-rank correction is changing that class's predictions."""
    model.eval()
    class_diff = {}

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            diff = ((model(xb) - model.forward_general(xb)) ** 2).mean(dim=1)
            for i in range(len(yb)):
                cls = yb[i].item()
                class_diff.setdefault(cls, []).append(diff[i].item())

    print("\nPer-class theta(t) effect:")
    for cls in sorted(class_diff):
        print(f"  Class {cls} (count={class_counts[cls]}): {np.mean(class_diff[cls]):.4f}")


def train_low_rank_classifier(
    model, loader, class_priors, device,
    epochs: int = 10, lr: float = 1e-3,
    contrastive_weight: float = 0.3, log_every: int = 100,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    T = len(loader) * epochs
    step = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            out_full = model(xb)
            out_general = model.forward_general(xb)

            base_loss = margin_loss(out_full, yb, margin=0.3)

            diff = (out_full - out_general) ** 2
            M = diff.mean(dim=1)
            weights = class_priors[yb]
            lmore = (weights * M).mean()

            alpha_t = alpha_schedule(step, T, amplitude=1.0)

            feat = model.layer1(xb)
            c_loss = supcon_loss(feat, yb)

            loss = base_loss + alpha_t * lmore + contrastive_weight * c_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            step += 1

            if step % log_every == 0:
                print(f"step={step} base_loss={base_loss.item():.4f} lmore={lmore.item():.4f} alpha={alpha_t:.4f}")

        print(f"Epoch {epoch}: total loss = {total_loss:.4f}")

    return model


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        preds = torch.argmax(model(xb), dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(yb.numpy())
    return np.array(all_labels), np.array(all_preds)


def main():
    parser = argparse.ArgumentParser(description="Low-rank class-imbalance decomposition classifier")
    parser.add_argument("--features", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--test_features", default=None)
    parser.add_argument("--test_labels", default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--results_out", default="results/results_lowrank.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    X = np.load(args.features)
    y = np.load(args.labels)

    loader = make_weighted_loader(X, y, batch_size=args.batch_size)
    class_counts = np.bincount(y)
    class_priors = torch.tensor(class_counts / len(y), dtype=torch.float32).to(device)

    model = LowRankClassifier(X.shape[1], len(np.unique(y)), rank=args.rank).to(device)
    model = train_low_rank_classifier(model, loader, class_priors, device, epochs=args.epochs, lr=args.lr)

    y_true, y_pred = predict(model, loader, device)
    evaluate_predictions(y_true, y_pred, save_path=args.results_out)
    per_class_theta_effect(model, loader, class_counts, device)

    if args.test_features and args.test_labels:
        X_test = np.load(args.test_features)
        y_test = np.load(args.test_labels)
        test_loader = DataLoader(FeatureDataset(X_test, y_test), batch_size=64, shuffle=False)
        y_true_t, y_pred_t = predict(model, test_loader, device)
        print("\n--- Held-out test set ---")
        evaluate_predictions(y_true_t, y_pred_t)


if __name__ == "__main__":
    main()
