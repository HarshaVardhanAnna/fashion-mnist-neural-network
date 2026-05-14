"""
Q1 — Plot one sample image per Fashion-MNIST class, log to wandb.
"""
import numpy as np
import matplotlib.pyplot as plt
import wandb
from keras.datasets import fashion_mnist
from src.utils import CLASS_NAMES


def plot_class_examples(save_path: str = "q1_samples.png"):
    (X_train, y_train), _ = fashion_mnist.load_data()

    fig, axes = plt.subplots(2, 5, figsize=(13, 6))
    fig.suptitle("Fashion-MNIST — One Sample per Class", fontsize=15, y=1.01)

    for cls in range(10):
        # First image in the training set belonging to this class
        idx = np.where(y_train == cls)[0][0]
        ax  = axes[cls // 5][cls % 5]
        ax.imshow(X_train[idx], cmap="gray")
        ax.set_title(CLASS_NAMES[cls], fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {save_path}")

    # Log to wandb
    run = wandb.init(project="fashion-mnist-nn", name="q1-class-samples")
    wandb.log({"class_samples": wandb.Image(save_path)})
    wandb.finish()


if __name__ == "__main__":
    plot_class_examples()