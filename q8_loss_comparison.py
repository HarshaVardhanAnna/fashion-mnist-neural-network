"""
Q8 — Compare cross-entropy vs squared-error loss on the same network.

Both models are initialised with the SAME random seed for a fair comparison.
Four plots are generated:
  (a) Training loss     (b) Validation loss
  (c) Training accuracy (d) Validation accuracy

Usage:
    python q8_loss_compare.py
"""
import numpy as np
import matplotlib.pyplot as plt
import wandb

from src.neural_network import NeuralNetwork
from src.optimizers     import get_optimizer, Nesterov
from src.losses         import get_loss
from src.utils          import load_data, get_batches


# Shared training config 
CONFIG = dict(
    epochs            = 10,
    num_hidden_layers = 4,
    hidden_size       = 128,
    activation        = "relu",
    weight_init       = "xavier",
    optimizer         = "adam",
    learning_rate     = 1e-3,
    batch_size        = 32,
    weight_decay      = 0.0005,
)


def train_model(loss_name: str,
                seed: int = 42) -> dict:
    """
    Train a model with the given loss function, recording metrics per epoch.
    Returns a dict of metric lists.
    """
    np.random.seed(seed)   # reproducible initialisation

    X_train, y_train, X_val, y_val, X_test, y_test = load_data(val_split=0.1)

    nn = NeuralNetwork(
        input_size   = 784,
        hidden_sizes = [CONFIG["hidden_size"]] * CONFIG["num_hidden_layers"],
        output_size  = 10,
        activation   = CONFIG["activation"],
        weight_init  = CONFIG["weight_init"],
    )

    optimizer             = get_optimizer(CONFIG["optimizer"],
                                          lr=CONFIG["learning_rate"])
    loss_fn, loss_grad_fn = get_loss(loss_name)
    is_nag                = isinstance(optimizer, Nesterov)

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
    }

    for epoch in range(CONFIG["epochs"]):
        batch_losses = []

        for X_b, y_b in get_batches(X_train, y_train, CONFIG["batch_size"]):
            if is_nag:
                W_save = [W.copy() for W in nn.weights]
                b_save = [b.copy() for b in nn.biases]
                optimizer.apply_lookahead(nn.weights, nn.biases)
                y_pred, cache = nn.forward(X_b)
                g = loss_grad_fn(y_pred, y_b)
                grads = nn.backward(g, cache, CONFIG["weight_decay"])
                for j in range(len(nn.weights)):
                    nn.weights[j][:] = W_save[j]
                    nn.biases[j][:] = b_save[j]
            else:
                y_pred, cache = nn.forward(X_b)
                g = loss_grad_fn(y_pred, y_b)
                grads = nn.backward(g, cache, CONFIG["weight_decay"])

            optimizer.update(nn.weights, nn.biases, grads)
            batch_losses.append(loss_fn(y_pred, y_b))

        # Epoch metrics
        train_loss = float(np.mean(batch_losses))
        train_acc  = nn.accuracy(X_train, y_train)
        y_vp, _    = nn.forward(X_val)
        val_loss   = loss_fn(y_vp, y_val)
        val_acc    = nn.accuracy(X_val, y_val)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(float(val_loss))
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"  [{loss_name}] Epoch {epoch+1:>2}  "
              f"train_loss={train_loss:.4f}  val_acc={val_acc:.3f}")

    history["test_acc"] = nn.accuracy(X_test, y_test)
    return history


def plot_comparison(hist_ce: dict,
                    hist_se: dict,
                    save_path: str = "q8_loss_comparison.png"):
    """
    Four-panel comparison figure with publication-quality styling.
    """
    epochs = list(range(1, CONFIG["epochs"] + 1))

    colours = {
        "cross_entropy": ("#2ecc71", "#27ae60"),   # (train, val)
        "squared_error": ("#e74c3c", "#c0392b"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(14, 9),
                             facecolor="#1a1a2e")
    fig.suptitle("Cross-Entropy vs Squared Error — Training Dynamics",
                 fontsize=14, color="white", y=0.98, fontweight="bold")

    panels = [
        ("train_loss", "Training Loss",     "Loss"),
        ("val_loss",   "Validation Loss",   "Loss"),
        ("train_acc",  "Training Accuracy", "Accuracy"),
        ("val_acc",    "Validation Accuracy","Accuracy"),
    ]

    for ax, (key, title, ylabel) in zip(axes.flatten(), panels):
        ax.set_facecolor("#16213e")
        ax.plot(epochs, hist_ce[key], color=colours["cross_entropy"][0],
                linewidth=2.2, marker="o", markersize=4,
                label="Cross-Entropy")
        ax.plot(epochs, hist_se[key], color=colours["squared_error"][0],
                linewidth=2.2, marker="s", markersize=4,
                label="Squared Error")
        ax.set_title(title,  color="white", fontsize=11)
        ax.set_xlabel("Epoch", color="#aaa", fontsize=9)
        ax.set_ylabel(ylabel,  color="#aaa", fontsize=9)
        ax.tick_params(colors="#aaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#555")
        ax.legend(fontsize=8, facecolor="#1a1a2e",
                  labelcolor="white", edgecolor="#555")
        ax.grid(alpha=0.15, color="white")

    # Annotate final test accuracy on val_acc panel
    ax_va = axes[1][1]
    ax_va.axhline(hist_ce["test_acc"], color=colours["cross_entropy"][1],
                  linewidth=1.2, linestyle="--", alpha=0.7)
    ax_va.axhline(hist_se["test_acc"], color=colours["squared_error"][1],
                  linewidth=1.2, linestyle="--", alpha=0.7)
    ax_va.text(CONFIG["epochs"] * 0.55, hist_ce["test_acc"] + 0.003,
               f"CE test={hist_ce['test_acc']:.3f}",
               color=colours["cross_entropy"][1], fontsize=8)
    ax_va.text(CONFIG["epochs"] * 0.55, hist_se["test_acc"] - 0.012,
               f"SE test={hist_se['test_acc']:.3f}",
               color=colours["squared_error"][1], fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print(f"Saved → {save_path}")


# Entry point
if __name__ == "__main__":
    run = wandb.init(project="fashion-mnist-nn", name="q8-loss-comparison")

    print("Training with Cross-Entropy...")
    hist_ce = train_model("cross_entropy", seed=42)

    print("\nTraining with Squared Error...")
    hist_se = train_model("squared_error", seed=42)   # same seed = same init

    plot_comparison(hist_ce, hist_se, "q8_loss_comparison.png")

    # Log to wandb
    for epoch in range(CONFIG["epochs"]):
        wandb.log({
            "epoch":              epoch + 1,
            "ce/train_loss":      hist_ce["train_loss"][epoch],
            "ce/val_loss":        hist_ce["val_loss"][epoch],
            "ce/val_accuracy":    hist_ce["val_acc"][epoch],
            "se/train_loss":      hist_se["train_loss"][epoch],
            "se/val_loss":        hist_se["val_loss"][epoch],
            "se/val_accuracy":    hist_se["val_acc"][epoch],
        })

    wandb.log({
        "ce/test_accuracy":  hist_ce["test_acc"],
        "se/test_accuracy":  hist_se["test_acc"],
        "comparison_plot":   wandb.Image("q8_loss_comparison.png"),
    })

    print(f"\nFinal Results:")
    print(f"  Cross-Entropy test accuracy : {hist_ce['test_acc']:.4f}")
    print(f"  Squared Error test accuracy : {hist_se['test_acc']:.4f}")

    wandb.finish()