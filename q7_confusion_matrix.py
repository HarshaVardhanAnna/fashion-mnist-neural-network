"""
Q7 — Confusion Matrix for the Best Fashion-MNIST Model

USAGE
─────
# Load best saved model
python q7_confusion.py --model_path best_model.npy

# OR retrain if no saved model exists
python q7_confusion.py --retrain
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import wandb

from src.neural_network import NeuralNetwork
from src.utils import load_data, CLASS_NAMES, get_batches
from src.optimizers import get_optimizer
from src.losses import get_loss

# Optional emoji labels
CLASS_ICONS = [
    "👕", "👖", "🧥", "👗", "🥼",
    "👡", "👔", "👟", "👜", "👢"
]

# Confusion Matrix
def compute_confusion_matrix(y_true, y_pred, n_classes=10):
    """
    CM[i,j]:
        true class i predicted as class j
    """
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    return cm

# Per-class Metrics
def compute_metrics(cm):
    n = cm.shape[0]

    precision = np.zeros(n)
    recall = np.zeros(n)
    f1 = np.zeros(n)

    for i in range(n):
        tp = cm[i, i]

        precision[i] = (
            tp / cm[:, i].sum()
            if cm[:, i].sum() > 0 else 0
        )

        recall[i] = (
            tp / cm[i, :].sum()
            if cm[i, :].sum() > 0 else 0
        )

        f1[i] = (
            2 * precision[i] * recall[i]
            / (precision[i] + recall[i] + 1e-12)
        )

    return precision, recall, f1

# Misclassification Analysis
def print_misclassification_analysis(cm):
    print("\nTop Misclassifications")
    print("─" * 70)

    errors = []

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                errors.append((cm[i, j], i, j))

    errors.sort(reverse=True)

    print(f"{'True':<15} {'Predicted':<15} {'Count':<10} {'Percent'}")
    print("─" * 70)

    for count, true_cls, pred_cls in errors[:10]:

        pct = (
            count / cm[true_cls, :].sum()
        ) * 100

        print(
            f"{CLASS_NAMES[true_cls]:<15}"
            f"{CLASS_NAMES[pred_cls]:<15}"
            f"{count:<10}"
            f"{pct:.2f}%"
        )

    print("\nPer-class Accuracy")
    print("─" * 70)

    for i in range(cm.shape[0]):
        acc = (
            cm[i, i] / cm[i, :].sum()
        ) * 100

        bar = "█" * int(acc / 5)

        print(
            f"{CLASS_NAMES[i]:<15}"
            f"{acc:>6.2f}%  {bar}"
        )

# Creative Plot
def plot_confusion_matrix(
    cm,
    test_acc,
    save_path="q7_confusion_matrix.png"
):
    n = cm.shape[0]

    # Row-normalized percentages
    cm_norm = (
        cm.astype(float)
        / (cm.sum(axis=1, keepdims=True) + 1e-12)
    )

    precision, recall, f1 = compute_metrics(cm)

    # Colormaps
    cmap_diag = LinearSegmentedColormap.from_list(
        "diag",
        ["#ffffff", "#1a7f5a"],
        N=256
    )

    cmap_off = LinearSegmentedColormap.from_list(
        "off",
        ["#ffffff", "#c0392b"],
        N=256
    )

    # RGBA image
    img_data = np.zeros((*cm_norm.shape, 4))

    for i in range(n):
        for j in range(n):

            v = cm_norm[i, j]

            if i == j:
                img_data[i, j] = cmap_diag(v)
            else:
                img_data[i, j] = cmap_off(min(v * 2, 1))

    # Layout
    fig = plt.figure(figsize=(16, 12), facecolor="#f8f9fa")

    gs = GridSpec(
        1,
        2,
        width_ratios=[4, 1],
        figure=fig,
        wspace=0.05
    )

    ax_cm = fig.add_subplot(gs[0])
    ax_side = fig.add_subplot(gs[1])

    # Main Heatmap
    ax_cm.imshow(img_data, aspect="auto")

    # Cell annotations
    for i in range(n):
        for j in range(n):

            pct = cm_norm[i, j] * 100
            cnt = cm[i, j]

            if cnt == 0:
                continue

            dark = pct > 50
            text_color = "white" if dark else "#2d3436"

            # Diagonal cells
            if i == j:
                label = f"{pct:.1f}%\n({cnt})"
                weight = "bold"

            else:
                label = f"{cnt}"
                weight = "normal"

            ax_cm.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=7.5,
                color=text_color,
                fontweight=weight
            )

    # Axes labels
    icon_labels = [
        f"{CLASS_ICONS[i]} {CLASS_NAMES[i]}"
        for i in range(n)
    ]

    ax_cm.set_xticks(range(n))
    ax_cm.set_yticks(range(n))

    ax_cm.set_xticklabels(
        icon_labels,
        rotation=40,
        ha="right",
        fontsize=9
    )

    ax_cm.set_yticklabels(
        icon_labels,
        fontsize=9
    )

    ax_cm.set_xlabel(
        "Predicted Class",
        fontsize=12
    )

    ax_cm.set_ylabel(
        "True Class",
        fontsize=12
    )

    ax_cm.set_title(
        f"Confusion Matrix — Fashion-MNIST\n"
        f"Test Accuracy: {test_acc*100:.2f}%",
        fontsize=14,
        pad=14,
        fontweight="bold"
    )

    # Grid lines
    for x in np.arange(-0.5, n, 1):
        ax_cm.axhline(x, color="white", linewidth=0.5)
        ax_cm.axvline(x, color="white", linewidth=0.5)

    # Highlight diagonal
    for i in range(n):
        rect = plt.Rectangle(
            (i - 0.5, i - 0.5),
            1,
            1,
            fill=False,
            edgecolor="#1a7f5a",
            linewidth=1.5
        )

        ax_cm.add_patch(rect)

    # Legend
    legend_items = [
        Patch(
            facecolor="#1a7f5a",
            label="Correct Predictions"
        ),
        Patch(
            facecolor="#c0392b",
            label="Misclassifications"
        ),
    ]

    ax_cm.legend(
        handles=legend_items,
        fontsize=8,
        loc="upper left"
    )

    # Sidebar Metrics
    y_pos = np.arange(n)
    bar_h = 0.25

    ax_side.barh(
        y_pos + bar_h,
        precision,
        bar_h,
        label="Precision",
        color="#3498db"
    )

    ax_side.barh(
        y_pos,
        recall,
        bar_h,
        label="Recall",
        color="#e67e22"
    )

    ax_side.barh(
        y_pos - bar_h,
        f1,
        bar_h,
        label="F1",
        color="#27ae60"
    )

    ax_side.set_xlim(0, 1.05)

    ax_side.set_yticks(y_pos)
    ax_side.set_yticklabels([])

    ax_side.set_xlabel("Score")
    ax_side.set_title("Per-class Metrics")

    ax_side.legend(fontsize=7)

    ax_side.grid(axis="x", alpha=0.3)

    # Save
    plt.savefig(
        save_path,
        dpi=180,
        bbox_inches="tight",
        facecolor=fig.get_facecolor()
    )

    plt.show()

    print(f"\nSaved → {save_path}")

# Train Best Model
def train_best_model():

    X_train, y_train, X_val, y_val, X_test, y_test = load_data(
        val_split=0.1
    )

    nn = NeuralNetwork(
        input_size=784,
        hidden_sizes=[128, 128, 128, 128],
        output_size=10,
        activation="relu",
        weight_init="xavier",
    )

    optimizer = get_optimizer("adam", lr=1e-3)

    loss_fn, loss_grad_fn = get_loss("cross_entropy")

    print("\nTraining best configuration...")

    for epoch in range(10):

        batch_losses = []

        for X_b, y_b in get_batches(
            X_train,
            y_train,
            batch_size=32
        ):

            y_pred, cache = nn.forward(X_b)

            loss = loss_fn(y_pred, y_b)

            g_out = loss_grad_fn(y_pred, y_b)

            grads = nn.backward(
                g_out,
                cache,
                weight_decay=0.0005
            )

            optimizer.update(
                nn.weights,
                nn.biases,
                grads
            )

            batch_losses.append(loss)

        val_acc = nn.accuracy(X_val, y_val)

        print(
            f"Epoch {epoch+1:>2}/10  "
            f"loss={np.mean(batch_losses):.4f}  "
            f"val_acc={val_acc:.4f}"
        )

    # Save model
    np.save(
        "best_model.npy",
        {
            "weights": nn.weights,
            "biases": nn.biases
        }
    )

    print("\nSaved best model → best_model.npy")

    return nn, X_test, y_test

# Load Saved Model
def load_saved_model(model_path):

    X_train, y_train, X_val, y_val, X_test, y_test = load_data(
        val_split=0.1
    )

    # Load checkpoint
    ckpt = np.load(model_path, allow_pickle=True)

    # Restore architecture
    layer_sizes = ckpt["layer_sizes"].tolist()

    nn = NeuralNetwork(
        input_size=layer_sizes[0],
        hidden_sizes=layer_sizes[1:-1],
        output_size=layer_sizes[-1],
        activation="tanh",      # use same activation as trained model
        weight_init="xavier"
    )

    # Restore weights/biases
    for i in range(nn.n_layers):
        nn.weights[i] = ckpt[f"W_{i}"]
        nn.biases[i]  = ckpt[f"b_{i}"]

    print(f"Model loaded ← {model_path}")

    return nn, X_test, y_test

# Main
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        default="best_model.npy"
    )

    parser.add_argument(
        "--retrain",
        action="store_true"
    )

    parser.add_argument(
        "--no_wandb",
        action="store_true"
    )

    args = parser.parse_args()

    # Load or train
    if (
        not args.retrain
        and os.path.exists(args.model_path)
    ):

        print(f"\nLoading saved model: {args.model_path}")

        nn, X_test, y_test = load_saved_model(
            args.model_path
        )

    else:

        nn, X_test, y_test = train_best_model()

    # Evaluate
    print("\nEvaluating on test set...")

    y_pred = nn.predict(X_test)

    test_acc = float(
        np.mean(y_pred == y_test)
    )

    print(
        f"\nTest Accuracy: "
        f"{test_acc*100:.2f}%"
    )

    # Confusion Matrix
    cm = compute_confusion_matrix(
        y_test,
        y_pred
    )

    # Analysis
    print_misclassification_analysis(cm)
    # Plot
    plot_confusion_matrix(
        cm,
        test_acc,
        save_path="q7_confusion_matrix.png"
    )

    # WandB Logging
    if not args.no_wandb:

        run = wandb.init(
            project="fashion-mnist-nn",
            name="q7-confusion-matrix",
            reinit=True
        )

        wandb.log({
            "test_accuracy": test_acc,
            "confusion_matrix_image": wandb.Image(
                "q7_confusion_matrix.png"
            ),
            "confusion_matrix_interactive":
                wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=y_test.tolist(),
                    preds=y_pred.tolist(),
                    class_names=CLASS_NAMES,
                )
        })

        wandb.finish()

        print("\nLogged confusion matrix to WandB.")