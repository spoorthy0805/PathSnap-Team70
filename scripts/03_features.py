"""Phase 3 — build the candidate-level feature table and analyse the features.

One row per (GPS observation, candidate arc) with R/K from config. Target y = 1 if
the candidate is the observation's reference arc (from the Phase 2 alignment).

Outputs
  cache/features/<split>.parquet                     feature tables (not committed)
  experiments/results/feature_table_stats.json       rows, positives, coverage per split
  experiments/results/feature_univariate.csv         per-feature class means + univariate ROC-AUC (train)
  experiments/results/heuristic_baselines.csv        rule-based Top-1 references (no learning)
  reports/figures/feat_*.png
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathsnap.config import CACHE_DIR, FIGURES_DIR, RESULTS_DIR, SPLITS, UNRELIABLE_TRIPS_RULE_M  # noqa: E402
from pathsnap.features import FEATURES, build_features  # noqa: E402
from pathsnap.io import iter_trips  # noqa: E402
from pathsnap.labels import align_track_to_route  # noqa: E402
from pathsnap.network import RoadNetwork  # noqa: E402

FEAT_DIR = CACHE_DIR / "features"
for d in (FEAT_DIR, RESULTS_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)


def build_split(split: str) -> pd.DataFrame:
    frames = []
    for trip in iter_trips(split):
        net = RoadNetwork.from_trip(trip)
        f, c = build_features(trip, net)
        lab = align_track_to_route(trip, net)
        f.insert(0, "trip_id", trip.trip_id)
        f.insert(1, "point", c.point.astype(np.int32))
        f.insert(2, "arc", c.arc.astype(np.int32))
        f["y"] = (c.arc == lab.arc[c.point]).astype(np.int8)
        f["unreliable"] = bool(np.median(lab.dist) > UNRELIABLE_TRIPS_RULE_M)
        f["n_points_trip"] = len(trip.track)
        frames.append(f)
        print(f"[{split}] {trip.trip_id}: {len(f):,} rows, pos rate {f.y.mean():.3f}")
    return pd.concat(frames, ignore_index=True)


def table_stats(df: pd.DataFrame) -> dict:
    pts = df.groupby(["trip_id", "point"]).y.max()
    n_points = int(df.drop_duplicates("trip_id").n_points_trip.sum())
    return {"rows": len(df), "trips": int(df.trip_id.nunique()), "points_total": n_points,
            "points_with_candidates": int(len(pts)), "points_label_in_candidates": int(pts.sum()),
            "positive_rate": float(df.y.mean()), "mean_candidates_per_point": float(len(df) / len(pts))}


def top1_rule(df: pd.DataFrame, score: pd.Series) -> float:
    """Observation-level Top-1 over ALL GPS points (points with no candidates count as wrong)."""
    d = df.assign(s=score.values)
    best = d.loc[d.groupby(["trip_id", "point"]).s.idxmax()]
    return float(best.y.sum() / df.drop_duplicates("trip_id").n_points_trip.sum())


def main():
    t0 = time.time()
    tables = {s: build_split(s) for s in SPLITS}
    for s, df in tables.items():
        df.to_parquet(FEAT_DIR / f"{s}.parquet", index=False)

    stats = {s: table_stats(df) for s, df in tables.items()}
    stats["train_reliable"] = table_stats(tables["train"][~tables["train"].unreliable])
    (RESULTS_DIR / "feature_table_stats.json").write_text(json.dumps(stats, indent=2))

    tr = tables["train"][~tables["train"].unreliable]
    rows = []
    for col in FEATURES:
        x = tr[col].astype(float)
        auc = roc_auc_score(tr.y, x)
        rows.append({"feature": col, "mean_pos": x[tr.y == 1].mean(), "mean_neg": x[tr.y == 0].mean(),
                     "auc": auc, "auc_strength": abs(auc - 0.5) * 2})
    uni = pd.DataFrame(rows).sort_values("auc_strength", ascending=False)
    uni.to_csv(RESULTS_DIR / "feature_univariate.csv", index=False)

    base = []
    for s, df in tables.items():
        base.append({"split": s, "rule": "nearest arc (rank 0)", "top1": top1_rule(df, -df["rank"].astype(float))})
        base.append({"split": s, "rule": "nearest arc with heading agreement",
                     "top1": top1_rule(df, -df.dist + 20 * df.heading_cos)})
        base.append({"split": s, "rule": "5-point window mean distance + heading",
                     "top1": top1_rule(df, -df.dist_win5_mean + 20 * df.heading_cos)})
    base = pd.DataFrame(base)
    base.to_csv(RESULTS_DIR / "heuristic_baselines.csv", index=False)

    # ---- figures ----
    fig, ax = plt.subplots(figsize=(8, 7))
    u = uni.iloc[::-1]
    ax.barh(u.feature, u.auc_strength, color=np.where(u.auc > 0.5, "#1a7f37", "#d1242f"))
    ax.set(title="Univariate separability on train (|2·AUC−1|; green: higher→positive)", xlabel="|2·AUC − 1|")
    ax.grid(alpha=0.3, axis="x"); fig.tight_layout(); fig.savefig(FIGURES_DIR / "feat_univariate_auc.png", dpi=130)
    plt.close(fig)

    key = ["dist", "dist_minus_min", "heading_cos", "heading_cos_wide", "dist_win5_mean", "dist_prev1",
           "second_gap", "n_within_radius"]
    fig, axes = plt.subplots(2, 4, figsize=(17, 7))
    sample = tr.sample(min(len(tr), 400_000), random_state=0)
    for ax, col in zip(axes.ravel(), key):
        lo, hi = np.nanpercentile(sample[col], [0.5, 99.5])
        bins = np.linspace(lo, hi, 50)
        for yv, color, name in [(0, "#9aa0a6", "other candidate"), (1, "#1a7f37", "reference arc")]:
            ax.hist(sample.loc[sample.y == yv, col].clip(lo, hi), bins=bins, density=True, alpha=0.6,
                    color=color, label=name)
        ax.set_title(col)
    axes[0, 0].legend()
    fig.suptitle("Feature distributions by class (train, reliable trips)"); fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feat_class_distributions.png", dpi=130); plt.close(fig)

    corr = sample[FEATURES].astype(float).corr()
    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURES)), FEATURES, rotation=90); ax.set_yticks(range(len(FEATURES)), FEATURES)
    fig.colorbar(im); ax.set_title("Feature correlation (train sample)"); fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feat_correlation.png", dpi=130); plt.close(fig)

    print(json.dumps(stats, indent=2))
    print(uni[["feature", "auc", "auc_strength"]].round(3).to_string(index=False))
    print(base.round(4).to_string(index=False))
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
