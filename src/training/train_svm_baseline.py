"""
Baseline classifier: Incremental PCA + RBF-kernel SVM.

This is the strongest, most stable model in the project (~75% test accuracy,
see README for full metrics). It also doubles as a 2D visualization tool:
pass --viz_2d to additionally fit a 2-component PCA purely for plotting the
SVM decision regions.

Usage
-----
    python train_svm_baseline.py --features data/features.npy --labels data/labels.npy \
        --test_features data/test_features.npy --test_labels data/test_labels.npy \
        --n_components 128 --results_out results/results.json
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

sys.path.append(str(Path(__file__).resolve().parent))
from evaluate import evaluate_predictions, plot_confusion_matrix  # noqa: E402


def load_or_split(features_path, labels_path, test_features_path, test_labels_path, test_size, seed):
    X = np.load(features_path)
    y = np.load(labels_path)
    print("Feature shape:", X.shape, "| Label shape:", y.shape)

    if test_features_path and test_labels_path:
        X_train, y_train = X, y
        X_test, y_test = np.load(test_features_path), np.load(test_labels_path)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )

    return X_train, X_test, y_train, y_test


def main():
    parser = argparse.ArgumentParser(description="Incremental PCA + SVM baseline")
    parser.add_argument("--features", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--test_features", default=None)
    parser.add_argument("--test_labels", default=None)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--n_components", type=int, default=128, help="PCA output dimensionality")
    parser.add_argument("--C", type=float, default=10.0, help="SVM regularization strength")
    parser.add_argument("--kernel", default="rbf")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results_out", default="results/results.json")
    parser.add_argument("--viz_2d", action="store_true", help="Also plot a 2D PCA decision-region figure")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_or_split(
        args.features, args.labels, args.test_features, args.test_labels, args.test_size, args.seed
    )

    print(f"Fitting Incremental PCA -> {args.n_components} components")
    ipca = IncrementalPCA(n_components=args.n_components, batch_size=max(256, args.n_components * 2))
    X_train_p = ipca.fit_transform(X_train)
    X_test_p = ipca.transform(X_test)
    print("Explained variance ratio (sum):", ipca.explained_variance_ratio_.sum())

    print(f"Training SVM (kernel={args.kernel}, C={args.C}) ...")
    svm = SVC(kernel=args.kernel, C=args.C, class_weight="balanced")
    svm.fit(X_train_p, y_train)

    y_pred = svm.predict(X_test_p)
    evaluate_predictions(y_test, y_pred, save_path=args.results_out)
    plot_confusion_matrix(y_test, y_pred, title="SVM Baseline — Confusion Matrix")

    if args.viz_2d:
        pca2 = PCA(n_components=2)
        X2_train = pca2.fit_transform(X_train)
        X2_test = pca2.transform(X_test)

        svm2 = SVC(kernel=args.kernel, C=args.C)
        svm2.fit(X2_train, y_train)

        import matplotlib.pyplot as plt

        h = 0.02
        x_min, x_max = X2_train[:, 0].min() - 1, X2_train[:, 0].max() + 1
        y_min, y_max = X2_train[:, 1].min() - 1, X2_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = svm2.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap="tab10")
        plt.scatter(X2_test[:, 0], X2_test[:, 1], c=y_test, cmap="tab10", edgecolors="k", s=10)
        plt.title("SVM Decision Boundary (2D PCA projection)")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")

    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
    main()
