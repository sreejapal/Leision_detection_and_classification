"""Shared evaluation / reporting utilities used across the different
classification heads (SVM baseline, quadruplet embeddings, low-rank model).
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def evaluate_predictions(y_true, y_pred, save_path: str | None = None) -> dict:
    """Prints and returns a full classification report dict. Optionally
    dumps it to JSON at `save_path` (same schema as results/results.json)."""
    report = classification_report(y_true, y_pred, output_dict=True)
    report["accuracy"] = accuracy_score(y_true, y_pred)

    print(f"Accuracy:          {report['accuracy']:.4f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro F1:          {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"Weighted F1:       {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print()
    print(classification_report(y_true, y_pred))

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(
                {"accuracy": report["accuracy"], "classification_report": report},
                f,
                indent=4,
            )
        print(f"✅ Saved metrics to {save_path}")

    return report


def plot_confusion_matrix(y_true, y_pred, title: str = "Confusion Matrix", normalize: bool = False):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
    else:
        fmt = "d"

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()
