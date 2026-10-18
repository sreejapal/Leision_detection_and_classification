"""
Quadruplet-loss metric-learning embedding network.

Trains a small MLP to project deep CNN features into a lower-dimensional
embedding space where same-class samples are pulled together and both a
"hard" and a "semi-hard" negative are pushed away, using hard-mining over
each batch. The resulting embeddings are then classified with an SVM.

This is a cleaned-up, importable version of `notebooks/01_quadruplet_metric_learning.ipynb`.

Usage
-----
    python quadruplet_net.py --features data/features.npy --labels data/labels.npy --epochs 100
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

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


class EmbeddingNet(nn.Module):
    """Projects raw deep features into a 128-d metric-learning embedding."""

    def __init__(self, input_dim: int, embed_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


def mine_quadruplets(embeddings: torch.Tensor, labels: torch.Tensor, margin: float = 1.0):
    """For each anchor in the batch, mines:
      - the hardest positive (largest same-class distance)
      - a semi-hard negative (just beyond the hardest-positive distance, within `margin`),
        falling back to the closest negative if none qualifies
      - a second, next-closest negative (for the quadruplet term)
    """
    dist = torch.cdist(embeddings, embeddings)

    anchors, positives, negatives1, negatives2 = [], [], [], []

    for i in range(len(labels)):
        anchor_label = labels[i]

        pos_mask = labels == anchor_label
        neg_mask = labels != anchor_label

        pos_indices = torch.where(pos_mask)[0]
        neg_indices = torch.where(neg_mask)[0]
        pos_indices = pos_indices[pos_indices != i]

        if len(pos_indices) == 0 or len(neg_indices) == 0:
            continue

        pos_dist = dist[i][pos_indices]
        neg_dist = dist[i][neg_indices]

        hard_pos = pos_indices[torch.argmax(pos_dist)]
        pos_max = pos_dist.max()

        semi_hard_mask = (neg_dist > pos_max) & (neg_dist < pos_max + margin)
        semi_hard = neg_indices[semi_hard_mask]

        if len(semi_hard) > 0:
            hard_neg1 = semi_hard[torch.randint(0, len(semi_hard), (1,))]
        else:
            hard_neg1 = neg_indices[torch.argmin(neg_dist)]

        sorted_neg = neg_indices[torch.argsort(neg_dist)]
        hard_neg2 = sorted_neg[1] if len(sorted_neg) > 1 else sorted_neg[0]

        anchors.append(i)
        positives.append(hard_pos.item())
        negatives1.append(hard_neg1.item())
        negatives2.append(hard_neg2.item())

    return anchors, positives, negatives1, negatives2


def quadruplet_loss(anchor, positive, negative1, negative2, margin1: float = 1.0, margin2: float = 0.3):
    d_ap = F.pairwise_distance(anchor, positive)
    d_an = F.pairwise_distance(anchor, negative1)
    d_pn = F.pairwise_distance(positive, negative2)

    loss1 = F.relu(d_ap - d_an + margin1)
    loss2 = F.relu(d_ap - d_pn + margin2)
    return (loss1 + loss2).mean()


def train_embedding_net(model, train_loader, device, epochs: int = 100, lr: float = 1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(epochs):
        model.train()
        total_loss, total_dap, total_dan, num_batches, dist_batches = 0.0, 0.0, 0.0, 0, 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for X_batch, y_batch in progress:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            embeddings = model(X_batch)

            a, p, n1, n2 = mine_quadruplets(embeddings, y_batch)
            if len(a) == 0:
                continue

            anchor, positive = embeddings[a], embeddings[p]
            negative1, negative2 = embeddings[n1], embeddings[n2]

            d_ap = F.pairwise_distance(anchor, positive)
            d_an = F.pairwise_distance(anchor, negative1)
            total_dap += d_ap.mean().item()
            total_dan += d_an.mean().item()
            dist_batches += 1

            loss = quadruplet_loss(anchor, positive, negative1, negative2)
            loss += 0.001 * torch.mean(torch.norm(embeddings, dim=1))  # light embedding-norm regularizer

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            progress.set_postfix(loss=loss.item())

        if num_batches:
            print(
                f"Epoch {epoch + 1}: loss={total_loss / num_batches:.4f} "
                f"d(a,p)={total_dap / dist_batches:.4f} d(a,n)={total_dan / dist_batches:.4f}"
            )

    return model


@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    all_embeddings, all_labels = [], []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        all_embeddings.append(model(X_batch).cpu())
        all_labels.append(y_batch)
    return torch.cat(all_embeddings).numpy(), torch.cat(all_labels).numpy()


def main():
    parser = argparse.ArgumentParser(description="Quadruplet-loss metric learning + SVM")
    parser.add_argument("--features", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--test_features", default=None)
    parser.add_argument("--test_labels", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results_out", default="results/results_quadruplet.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    X = np.load(args.features)
    y = np.load(args.labels)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.seed
    )

    train_loader = DataLoader(FeatureDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(FeatureDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)

    model = EmbeddingNet(X.shape[1], embed_dim=args.embed_dim).to(device)
    model = train_embedding_net(model, train_loader, device, epochs=args.epochs, lr=args.lr)

    X_train_emb, y_train_emb = extract_embeddings(model, train_loader, device)
    X_val_emb, y_val_emb = extract_embeddings(model, val_loader, device)

    svm = SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced")
    svm.fit(X_train_emb, y_train_emb)
    y_pred = svm.predict(X_val_emb)

    evaluate_predictions(y_val_emb, y_pred, save_path=args.results_out)

    if args.test_features and args.test_labels:
        X_test = np.load(args.test_features)
        y_test = np.load(args.test_labels)
        test_loader = DataLoader(FeatureDataset(X_test, y_test), batch_size=64, shuffle=False)
        X_test_emb, y_test_emb = extract_embeddings(model, test_loader, device)
        y_test_pred = svm.predict(X_test_emb)
        print("\n--- Held-out test set ---")
        evaluate_predictions(y_test_emb, y_test_pred)


if __name__ == "__main__":
    main()
