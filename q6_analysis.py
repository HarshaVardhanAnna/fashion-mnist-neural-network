"""
Q6 — Fetch completed sweep runs from wandb, generate a custom parallel
coordinates plot and per-hyperparameter accuracy breakdowns, then print
automated insights you can use as a starting point for your written analysis.

Usage:
    python q6_analysis.py <wandb_project> <sweep_id>
    e.g. python q6_analysis.py fashion-mnist-nn abc12345
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import wandb


# 1. Fetch data from wandb API
def fetch_sweep_results(entity: str, project: str, sweep_id: str) -> pd.DataFrame:
    """Pull every finished run from a wandb sweep into a DataFrame."""
    api   = wandb.Api()
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")

    rows = []
    for run in sweep.runs:
        if run.state != "finished":
            continue
        row = dict(run.config)          # all hyperparameter values
        row.update({
            "val_accuracy":  run.summary.get("val_accuracy",  None),
            "val_loss":      run.summary.get("val_loss",      None),
            "test_accuracy": run.summary.get("test_accuracy", None),
            "run_name":      run.name,
        })
        rows.append(row)

    df = pd.DataFrame(rows).dropna(subset=["val_accuracy"])
    print(f"Loaded {len(df)} finished runs.")
    return df


# 2. Parallel coordinates (custom, matplotlib)
def plot_parallel_coords(df: pd.DataFrame, save_path: str = "q6_parallel_coords.png"):
    """
    Parallel coordinates plot where each line is one hyperparameter
    configuration and colour encodes validation accuracy.
    """
    param_cols = [
        "num_hidden_layers", "hidden_size", "learning_rate",
        "batch_size", "weight_decay", "optimizer",
        "activation", "weight_init",
    ]
    target     = "val_accuracy"
    cols       = [c for c in param_cols if c in df.columns] + [target]
    df_plot    = df[cols].copy()

    # Encode categoricals → integers for axis placement
    encodings = {}
    for col in df_plot.select_dtypes("object").columns:
        cats = sorted(df_plot[col].dropna().unique())
        encodings[col] = {c: i for i, c in enumerate(cats)}
        df_plot[col]   = df_plot[col].map(encodings[col])

    # Normalise every column to [0, 1] for uniform y-axis
    df_norm = (df_plot - df_plot.min()) / (df_plot.max() - df_plot.min() + 1e-9)

    cmap     = cm.RdYlGn
    norm_acc = Normalize(df_plot[target].min(), df_plot[target].max())

    fig, ax  = plt.subplots(figsize=(16, 7))
    n_cols   = len(cols)

    for idx, row in df_norm.iterrows():
        color = cmap(norm_acc(df_plot.loc[idx, target]))
        ax.plot(range(n_cols), [row[c] for c in cols],
                color=color, alpha=0.35, linewidth=1.3)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(cols, rotation=25, ha="right", fontsize=9)
    ax.set_yticks([])
    ax.set_title("Parallel Coordinates — Hyperparameter Sweep\n"
                 "(line colour: red = low val_accuracy → green = high)",
                 fontsize=12, pad=12)

    # Categorical tick annotations below each axis
    for j, col in enumerate(cols):
        if col in encodings:
            rev = {v: k for k, v in encodings[col].items()}
            label = "\n".join(rev[k] for k in sorted(rev))
            ax.text(j, -0.09, label, ha="center", va="top",
                    fontsize=6.5, transform=ax.get_xaxis_transform(),
                    color="#555")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_acc)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Validation Accuracy", shrink=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {save_path}")


# 3. Per-hyperparameter accuracy breakdown
def plot_accuracy_breakdowns(df: pd.DataFrame, save_path: str = "q6_breakdowns.png"):
    """
    Box-plots of val_accuracy split by each categorical hyperparameter.
    Reveals which choices consistently produce high accuracy.
    """
    cat_params = ["optimizer", "activation", "weight_init",
                  "num_hidden_layers", "batch_size", "learning_rate"]
    cat_params = [c for c in cat_params if c in df.columns]

    fig, axes  = plt.subplots(2, 3, figsize=(16, 9))
    axes       = axes.flatten()

    for ax, param in zip(axes, cat_params):
        groups = [df[df[param] == v]["val_accuracy"].values
                  for v in sorted(df[param].unique())]
        labels = [str(v) for v in sorted(df[param].unique())]
        bp = ax.boxplot(groups, labels=labels, patch_artist=True,
                        medianprops=dict(color="black", linewidth=2))
        for patch in bp["boxes"]:
            patch.set_facecolor("#4C9BE8")
            patch.set_alpha(0.6)
        ax.set_title(f"val_accuracy by {param}", fontsize=10)
        ax.set_xlabel(param)
        ax.set_ylabel("val_accuracy")
        ax.tick_params(axis="x", rotation=25)

    for ax in axes[len(cat_params):]:
        ax.set_visible(False)

    plt.suptitle("Validation Accuracy Breakdown by Hyperparameter", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {save_path}")


# 4. Print automated insights
def print_insights(df: pd.DataFrame):
    print("\n" + "═" * 60)
    print("  AUTOMATED INSIGHTS FOR Q6 REPORT")
    print("═" * 60)

    best = df.loc[df["val_accuracy"].idxmax()]
    print(f"\n★ Best val_accuracy  : {best['val_accuracy']:.4f}")
    print(f"  Best run name      : {best['run_name']}")
    print(f"  Best config        :")
    for k in ["optimizer", "activation", "num_hidden_layers",
              "hidden_size", "learning_rate", "batch_size",
              "weight_decay", "weight_init"]:
        if k in best:
            print(f"    {k:20s}: {best[k]}")

    for col in ["optimizer", "activation", "learning_rate",
                "weight_decay", "weight_init"]:
        if col not in df.columns:
            continue
        print(f"\n── Mean val_accuracy by {col}:")
        summary = (df.groupby(col)["val_accuracy"]
                     .agg(["mean", "std", "count"])
                     .sort_values("mean", ascending=False))
        print(summary.to_string())

    low = df[df["val_accuracy"] < 0.65]
    print(f"\n── Runs with val_accuracy < 65%: {len(low)} / {len(df)}")
    if not low.empty:
        for col in ["optimizer", "activation", "learning_rate"]:
            if col in low.columns:
                print(f"  Most common {col}: {low[col].mode()[0]}")

    print("\n── Top-5 configurations:")
    show_cols = ["run_name", "val_accuracy", "optimizer", "activation",
                 "learning_rate", "num_hidden_layers", "hidden_size"]
    show_cols = [c for c in show_cols if c in df.columns]
    print(df.nlargest(5, "val_accuracy")[show_cols].to_string(index=False))
    print("═" * 60)


# Entry point
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python q6_analysis.py <project> <sweep_id>")
        sys.exit(1)

    project  = sys.argv[1]
    sweep_id = sys.argv[2]
    api      = wandb.Api()
    entity   = api.default_entity

    df = fetch_sweep_results(entity, project, sweep_id)

    run = wandb.init(project=project, name="q6-analysis")
    plot_parallel_coords(df,       "q6_parallel_coords.png")
    plot_accuracy_breakdowns(df,   "q6_breakdowns.png")
    print_insights(df)

    wandb.log({
        "q6/parallel_coords":  wandb.Image("q6_parallel_coords.png"),
        "q6/breakdowns":       wandb.Image("q6_breakdowns.png"),
    })
    wandb.finish()