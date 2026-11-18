from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments" / "results" / "phase5"
OUT = ROOT / "reports" / "figures"

TEAL = "#137C78"
ORANGE = "#D18A45"
GREY = "#71858A"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelcolor": "#263238",
    "text.color": "#263238",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white"
})

def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", name)

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    trials = pd.read_csv(DATA / "trials.csv")
    trials = trials[trials["state"] == "COMPLETE"].copy()
    trials = trials.dropna(subset=["cv_top1"])
    trials = trials.sort_values("trial")

    if trials.empty:
        raise ValueError("No completed trials found")

    fig, ax = plt.subplots(figsize=(9, 4.5))

    ax.scatter(
        trials["trial"],
        trials["cv_top1"],
        color=TEAL,
        s=35,
        label="Completed trials"
    )

    ax.plot(
        trials["trial"],
        trials["cv_top1"].cummax(),
        color=ORANGE,
        linewidth=2,
        label="Best result so far"
    )

    ax.set_title("Bayesian Optimisation Progress")
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Cross-validation Top-1 accuracy")
    ax.grid(axis="y", alpha=0.15)
    ax.legend()

    save(fig, "hpo_history.png")

    importance = json.loads(
        (DATA / "param_importance.json").read_text(encoding="utf-8")
    )

    names = list(importance.keys())[::-1]
    values = [importance[name] for name in names]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.barh(names, values, color=TEAL)

    ax.set_title("Hyperparameter Importance")
    ax.set_xlabel("Relative importance")
    ax.grid(axis="x", alpha=0.15)

    save(fig, "hpo_param_importance.png")

    comp = pd.read_csv(DATA / "tuned_vs_untuned.csv")

    metrics = [
        ("cv_f1_macro_mean", "CV Macro-F1"),
        ("cv_top1_mean", "CV Top-1"),
        ("test_top1", "Test Top-1"),
        ("test_f1_macro", "Test Macro-F1")
    ]

    fig, axes = plt.subplots(
        1, 3,
        figsize=(15, 4.5),
        gridspec_kw={"width_ratios": [3, 1, 1]}
    )

    x = np.arange(len(metrics))
    width = 0.35

    colors = [GREY, TEAL]

    for i, row in comp.reset_index(drop=True).iterrows():
        values = [row[key] for key, _ in metrics]

        axes[0].bar(
            x + (i - 0.5) * width,
            values,
            width,
            color=colors[i],
            label=row["model"]
        )

    axes[0].set_xticks(x, [label for _, label in metrics])
    axes[0].set_ylim(0.85, 0.96)
    axes[0].set_title("Tuned vs Untuned XGBoost")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.15)

    axes[1].bar(
        comp["model"],
        comp["test_log_loss"],
        color=colors
    )
    axes[1].set_title("Test Log Loss")
    axes[1].tick_params(axis="x", rotation=25)

    axes[2].bar(
        comp["model"],
        comp["test_predict_ms_per_1k_points"],
        color=colors
    )
    axes[2].set_title("Inference Time")
    axes[2].set_ylabel("ms / 1,000 GPS points")
    axes[2].tick_params(axis="x", rotation=25)

    save(fig, "hpo_tuned_vs_untuned.png")

    print("All three HPO graphs regenerated successfully")

if __name__ == "__main__":
    main()