"""
Q4 / Q5 — WandB hyperparameter sweep.

Strategy: Bayesian optimisation with 25 runs.
Rationale: ~11,000 combinations in the grid; Bayesian search builds a
surrogate model after each run and focuses sampling on high-accuracy
regions. 25 runs gives enough coverage to find a strong configuration.
"""
import wandb
from train import train


SWEEP_CONFIG = {
    "method": "bayes",                  # Bayesian optimisation
    "metric": {
        "name": "val_accuracy",
        "goal": "maximize",
    },
    "parameters": {
        # Training duration
        "epochs":            {"values": [5, 10]},

        # Architecture
        "num_hidden_layers": {"values": [3, 4, 5]},
        "hidden_size":       {"values": [32, 64, 128]},
        "activation":        {"values": ["sigmoid", "tanh", "relu"]},
        "weight_init":       {"values": ["random", "xavier"]},

        # Optimisation
        "optimizer":         {"values": ["sgd", "momentum", "nesterov","rmsprop", "adam", "nadam"]},
        "learning_rate":     {"values": [1e-3, 1e-4]},
        "batch_size":        {"values": [16, 32, 64]},

        # Regularisation
        "weight_decay":      {"values": [0, 0.0005, 0.5]},

        # Loss
        "loss":              {"value":  "cross_entropy"},
    },
    # Early termination: kill runs that are clearly underperforming
    "early_terminate": {
        "type":   "hyperband",
        "min_iter": 3,
        "eta":      2,
    },
}


def sweep_agent():
    """Called once per sweep run by wandb.agent."""
    train(config=None, use_wandb=True)


if __name__ == "__main__":
    sweep_id = wandb.sweep(
        sweep=SWEEP_CONFIG,
        project="fashion-mnist-nn",
    )
    print(f"Sweep ID: {sweep_id}")
    # count=25 → run 25 configurations
    wandb.agent(sweep_id, function=sweep_agent, count=25)