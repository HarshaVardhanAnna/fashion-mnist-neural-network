"""
Training script for Q3 / Q4.

Usage (standalone, no wandb):
    python train.py --optimizer adam --epochs 10 --learning_rate 1e-3

Usage (called by sweep.py with wandb):
    train(config=None, use_wandb=True)
"""
import argparse
import numpy as np
import wandb

from src.neural_network import NeuralNetwork
from src.optimizers      import get_optimizer, Nesterov
from src.losses          import get_loss
from src.utils           import load_data, get_batches


# Optimizer factory
def build_optimizer(name: str, lr: float) -> object:
    """
    Map optimizer name → initialised optimizer.
    Default hyper-parameters are well-known good values; they can be
    exposed as sweep parameters if needed.
    """
    defaults = {
        "sgd":      dict(lr=lr),
        "momentum": dict(lr=lr, beta=0.9),
        "nesterov": dict(lr=lr, beta=0.9),
        "rmsprop":  dict(lr=lr, rho=0.9,  eps=1e-8),
        "adam":     dict(lr=lr, beta1=0.9, beta2=0.999, eps=1e-8),
        "nadam":    dict(lr=lr, beta1=0.9, beta2=0.999, eps=1e-8),
    }
    if name not in defaults:
        raise ValueError(f"Unknown optimizer: {name}")
    return get_optimizer(name, **defaults[name])


# Core training function
def train(config=None, use_wandb: bool = True):
    """
    Train the network.

    Parameters
    ----------
    config    : argparse.Namespace or wandb.config
    use_wandb : bool — set False for local runs without wandb
    """
    if use_wandb:
        wandb.init(config=config)
        cfg = wandb.config
        # Meaningful run name (required by Q4)
        wandb.run.name = (
            f"hl_{cfg.num_hidden_layers}_"
            f"hs_{cfg.hidden_size}_"
            f"bs_{cfg.batch_size}_"
            f"ac_{cfg.activation}_"
            f"opt_{cfg.optimizer}_"
            f"lr_{cfg.learning_rate}"
        )
        wandb.save("train.py")
        wandb.save("src/*.py")
    else:
        cfg = config

    # Data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(val_split=0.1)

    # Model
    nn = NeuralNetwork(
        input_size   = 784,
        hidden_sizes = [cfg.hidden_size] * cfg.num_hidden_layers,
        output_size  = 10,
        activation   = cfg.activation,
        weight_init  = cfg.weight_init,
    )

    # Optimizer & Loss
    optimizer              = build_optimizer(cfg.optimizer, lr=cfg.learning_rate)
    loss_fn, loss_grad_fn  = get_loss(cfg.loss)
    is_nag                 = isinstance(optimizer, Nesterov)

    # Training Loop
    for epoch in range(cfg.epochs):
        batch_losses = []

        for X_b, y_b in get_batches(X_train, y_train, cfg.batch_size):

            if is_nag:
                # Nesterov lookahead
                # 1. Save current weights
                W_save = [W.copy() for W in nn.weights]
                b_save = [b.copy() for b in nn.biases]
                # 2. Shift to lookahead position
                optimizer.apply_lookahead(nn.weights, nn.biases)
                # 3. Compute gradients at lookahead
                y_pred, cache = nn.forward(X_b)
                g_out  = loss_grad_fn(y_pred, y_b)
                grads  = nn.backward(g_out, cache, cfg.weight_decay)
                # 4. Restore original weights before applying update
                for j in range(len(nn.weights)):
                    nn.weights[j][:] = W_save[j]
                    nn.biases[j][:]  = b_save[j]
            else:
                y_pred, cache = nn.forward(X_b)
                g_out  = loss_grad_fn(y_pred, y_b)
                grads  = nn.backward(g_out, cache, cfg.weight_decay)

            optimizer.update(nn.weights, nn.biases, grads)
            batch_losses.append(loss_fn(y_pred, y_b))

        # Epoch-level metrics
        train_loss = float(np.mean(batch_losses))
        train_acc  = nn.accuracy(X_train, y_train)

        y_val_pred, _ = nn.forward(X_val)
        val_loss = loss_fn(y_val_pred, y_val)
        val_acc  = nn.accuracy(X_val, y_val)

        print(f"Epoch {epoch+1:>3}/{cfg.epochs}  "
              f"loss={train_loss:.4f}  acc={train_acc:.3f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")

        if use_wandb:
            wandb.log({
                "epoch":        epoch + 1,
                "loss":         train_loss,
                "accuracy":     train_acc,
                "val_loss":     val_loss,
                "val_accuracy": val_acc,
            })

    # Final test accuracy
    test_acc = nn.accuracy(X_test, y_test)
    print(f"\nTest accuracy: {test_acc:.4f}")
    if use_wandb:
        wandb.log({"test_accuracy": test_acc})
        wandb.finish()

    return nn, test_acc


# CLI entry-point
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train Fashion-MNIST NN")
    p.add_argument("--epochs",            type=int,   default=10)
    p.add_argument("--num_hidden_layers", type=int,   default=3)
    p.add_argument("--hidden_size",       type=int,   default=128)
    p.add_argument("--activation",        type=str,   default="relu")
    p.add_argument("--optimizer",         type=str,   default="adam")
    p.add_argument("--learning_rate",     type=float, default=1e-3)
    p.add_argument("--batch_size",        type=int,   default=32)
    p.add_argument("--weight_decay",      type=float, default=0.0)
    p.add_argument("--weight_init",       type=str,   default="xavier")
    p.add_argument("--loss",              type=str,   default="cross_entropy")
    args = p.parse_args()

    train(config=args, use_wandb=False)