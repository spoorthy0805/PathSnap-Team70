from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results" / "phase6"
FIGURES = ROOT / "reports" / "figures"

TEAL = "#137C78"
ORANGE = "#D18A45"
GREY = "#71858A"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "text.color": "#263238",
    "axes.labelcolor": "#263238"
})

def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", name)

def comparison(folder, output):
    df = pd.read_csv(folder / "summary.csv")
    df = df[df["split"] == "student_test"]

    metrics = [
        ("top1", "Top-1"),
        ("continuity", "Continuity"),
        ("route_f1", "Route F1")
    ]

    x = np.arange(len(metrics))
    width = 0.35

    clf = df[df["method"] == "classifier_only"].iloc[0]
    hmm = df[df["method"] == "classifier+hmm"].iloc[0]

    fig, ax = plt.subplots(figsize=(9, 4.5))

    ax.bar(
        x - width / 2,
        [clf[key] for key, _ in metrics],
        width,
        label="Classifier",
        color=GREY
    )

    ax.bar(
        x + width / 2,
        [hmm[key] for key, _ in metrics],
        width,
        label="Classifier + HMM",
        color=TEAL
    )

    ax.set_xticks(x, [name for _, name in metrics])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Route Prediction Performance")
    ax.legend()
    ax.grid(axis="y", alpha=0.15)

    save(fig, output)

def per_trip(folder, output):
    df = pd.read_csv(folder / "per_trip.csv", dtype={"trip_id": str})
    df = df[df["split"] == "student_test"].copy()
    df = df.sort_values("trip_id")

    colors = [
        TEAL if value >= 0 else ORANGE
        for value in df["top1_gain"]
    ]

    fig, ax = plt.subplots(figsize=(10, 4.5))

    ax.bar(
        df["trip_id"].str[-2:],
        df["top1_gain"] * 100,
        color=colors
    )

    ax.axhline(0, color=GREY, linewidth=1)
    ax.set_xlabel("Trip ID")
    ax.set_ylabel("Top-1 improvement (percentage points)")
    ax.set_title("HMM Improvement Across Student-Test Trips")
    ax.grid(axis="y", alpha=0.15)

    save(fig, output)

def decoder_grid(folder, output):
    df = pd.read_csv(folder / "decoder_grid.csv")

    gap = 30
    subset = df[np.isclose(df["gap_s"], gap)].copy()

    if subset.empty:
        gap = float(df["gap_s"].iloc[0])
        subset = df[np.isclose(df["gap_s"], gap)].copy()

    pivot = subset.pivot_table(
        index="alpha",
        columns="beta",
        values="top1",
        aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    image = ax.imshow(
        pivot.values,
        cmap="YlGnBu",
        aspect="auto"
    )

    ax.set_xticks(
        range(len(pivot.columns)),
        [str(x) for x in pivot.columns]
    )

    ax.set_yticks(
        range(len(pivot.index)),
        [str(x) for x in pivot.index]
    )

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.values[i, j]

            if np.isfinite(value):
                ax.text(
                    j,
                    i,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    color="#263238",
                    fontsize=9
                )

    ax.set_xlabel("Beta")
    ax.set_ylabel("Alpha")
    ax.set_title(f"Decoder Parameter Comparison (gap = {gap:g}s)")

    fig.colorbar(image, ax=ax, label="Top-1 accuracy")

    save(fig, output)

def main():
    FIGURES.mkdir(parents=True, exist_ok=True)

    configs = [
        (
            "xgb_tuned",
            "hmm_vs_classifier.png",
            "hmm_per_trip.png",
            "hmm_decoder_grid.png"
        ),
        (
            "xgb_untuned",
            "hmm_xgb_untuned_vs_classifier.png",
            "hmm_xgb_untuned_per_trip.png",
            "hmm_xgb_untuned_decoder_grid.png"
        )
    ]

    for folder_name, comparison_name, trip_name, grid_name in configs:
        folder = RESULTS / folder_name

        comparison(folder, comparison_name)
        per_trip(folder, trip_name)
        decoder_grid(folder, grid_name)

    print("All six HMM graphs regenerated successfully")

if __name__ == "__main__":
    main()