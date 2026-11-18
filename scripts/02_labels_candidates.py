"""Phase 2 — per-observation reference labels and candidate-set miss-rate study.

For every train / student_test trip:
  1. align the GPS track to the reference route (monotone DP)  -> label arc per point
  2. generate candidate arcs within R_MAX metres                -> distance-ranked candidates
  3. record where the label arc sits among the candidates (rank, distance)

From (3) we compute the candidate miss rate for any radius R <= R_MAX and cap K
without regenerating candidates: the label is covered iff label_dist <= R and
label_rank < K.

Outputs
  cache/labels/<split>_<trip>.npz                      per-point labels (not committed)
  experiments/results/labels_trip_summary.csv          per-trip label quality + reliability flag
  experiments/results/candidate_miss_rate.csv          miss rate / candidate count grid
  reports/figures/cand_*.png, labels_*.png
  reports/maps/labels_trip_<id>.html                   point -> label-arc assignment map
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import folium  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathsnap.candidates import ArcIndex  # noqa: E402
from pathsnap.config import CACHE_DIR, FIGURES_DIR, MAPS_DIR, RESULTS_DIR, SPLITS  # noqa: E402
from pathsnap.io import iter_trips, load_trip  # noqa: E402
from pathsnap.labels import align_track_to_route, track_xy  # noqa: E402
from pathsnap.network import RoadNetwork  # noqa: E402
from pathsnap.viz import trip_map  # noqa: E402

R_MAX = 100.0
RADII = [15, 20, 25, 30, 40, 50, 60, 75, 100]
KS = [1, 2, 4, 6, 8, 10, 12, 16, 20, 32, None]
from pathsnap.config import UNRELIABLE_TRIPS_RULE_M as UNRELIABLE_MEDIAN_M  # noqa: E402

LABEL_DIR = CACHE_DIR / "labels"
for d in (LABEL_DIR, RESULTS_DIR, FIGURES_DIR, MAPS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def process(trip):
    net = RoadNetwork.from_trip(trip)
    pts = track_xy(trip, net)
    lab = align_track_to_route(trip, net, pts)
    cands = ArcIndex(net).query(pts, R_MAX)

    # rank of the label arc among this point's candidates (-1 if not within R_MAX)
    hit = cands.arc == lab.arc[cands.point]
    label_rank = np.full(len(pts), -1, dtype=np.int64)
    label_rank[cands.point[hit]] = cands.rank[hit]
    n_within = {r: np.bincount(cands.point[cands.dist <= r], minlength=len(pts)) for r in RADII}

    np.savez_compressed(LABEL_DIR / f"{trip.split}_{trip.trip_id}.npz", route_pos=lab.route_pos, arc=lab.arc,
                        dist=lab.dist, frac=lab.frac, label_rank=label_rank)
    summary = {
        "trip_id": trip.trip_id, "split": trip.split, "n_points": len(pts),
        "label_dist_median_m": float(np.median(lab.dist)),
        "label_dist_p95_m": float(np.percentile(lab.dist, 95)),
        "route_positions_used": float(len(np.unique(lab.route_pos)) / len(trip.route)),
        "unreliable": bool(np.median(lab.dist) > UNRELIABLE_MEDIAN_M),
    }
    per_point = pd.DataFrame({"trip_id": trip.trip_id, "split": trip.split, "label_dist": lab.dist,
                              "label_rank": label_rank, **{f"n_r{r}": n_within[r] for r in RADII}})
    return summary, per_point


def miss_grid(pp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, subset), g in [((s, "all"), pp[pp.split == s]) for s in SPLITS] + \
                              [((s, "reliable"), pp[(pp.split == s) & ~pp.unreliable]) for s in SPLITS]:
        for r in RADII:
            for k in KS:
                covered = (g.label_dist <= r) & (g.label_rank >= 0)
                if k is not None:
                    covered &= g.label_rank < k
                n = g[f"n_r{r}"] if k is None else np.minimum(g[f"n_r{r}"], k)
                rows.append({"split": split, "subset": subset, "radius_m": r, "k": k or "all",
                             "miss_rate": float(1 - covered.mean()), "no_candidate_rate": float((n == 0).mean()),
                             "mean_candidates": float(n.mean()), "p95_candidates": float(np.percentile(n, 95))})
    return pd.DataFrame(rows)


def figures(grid: pd.DataFrame, pp: pd.DataFrame):
    g = grid[(grid.split == "train") & (grid.subset == "reliable")]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for k in [2, 4, 8, 12, 16, "all"]:
        s = g[g.k == k]
        ax[0].plot(s.radius_m, s.miss_rate * 100, marker="o", label=f"K={k}")
        ax[1].plot(s.radius_m, s.mean_candidates, marker="o", label=f"K={k}")
    ax[0].set(title="Candidate miss rate (train, reliable trips)", xlabel="radius (m)", ylabel="miss rate (%)")
    ax[0].set_yscale("log"); ax[0].grid(alpha=0.3); ax[0].legend()
    ax[1].set(title="Mean candidates per GPS point (train)", xlabel="radius (m)", ylabel="candidates")
    ax[1].grid(alpha=0.3); ax[1].legend()
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "cand_miss_rate_vs_radius.png", dpi=130); plt.close(fig)

    piv = g.pivot(index="radius_m", columns="k", values="miss_rate") * 100
    piv = piv[[c for c in [1, 2, 4, 6, 8, 10, 12, 16, 20, 32, "all"] if c in piv.columns]]
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(np.log10(piv.values + 1e-2), cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(piv.columns)), piv.columns); ax.set_yticks(range(len(piv.index)), piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", fontsize=8, color="w")
    ax.set(xlabel="K (max candidates)", ylabel="radius (m)", title="Miss rate % (train, reliable trips)")
    fig.colorbar(im, label="log10 miss %"); fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cand_miss_rate_heatmap.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for s in SPLITS:
        x = pp[(pp.split == s) & ~pp.unreliable]
        ax[0].hist(np.clip(x.label_dist, 0, 60), bins=60, alpha=0.6, density=True, label=s)
        lr = x.label_rank[x.label_rank >= 0]
        ax[1].hist(np.clip(lr, 0, 20), bins=np.arange(22) - 0.5, alpha=0.6, density=True, label=s)
    ax[0].set(title="GPS point → label arc distance (reliable trips, m, clipped 60)", xlabel="m")
    ax[1].set(title="Rank of label arc among distance-sorted candidates (clipped 20)", xlabel="rank")
    for a in ax:
        a.legend(); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "labels_dist_and_rank.png", dpi=130); plt.close(fig)


def label_map(trip_id: str, split: str, n_lines: int = 600):
    trip = load_trip(trip_id, split)
    net = RoadNetwork.from_trip(trip)
    lab = np.load(LABEL_DIR / f"{split}_{trip_id}.npz")
    m = trip_map(trip, net, title=f"Trip {trip_id}: GPS → label-arc assignment")
    a, b = net.a_xy[lab["arc"]], net.b_xy[lab["arc"]]
    proj = a + lab["frac"][:, None] * (b - a)
    plon, plat = net.proj.to_lonlat(proj[:, 0], proj[:, 1])
    fg = folium.FeatureGroup("Point → label arc", show=True)
    step = max(1, len(plon) // n_lines)
    for i in range(0, len(plon), step):
        folium.PolyLine([(trip.track.lat.iat[i], trip.track.lon.iat[i]), (plat[i], plon[i])], color="#bf8700",
                        weight=1.5, tooltip=f"pt {i} → arc {lab['arc'][i]} ({lab['dist'][i]:.1f} m)").add_to(fg)
    fg.add_to(m)
    m.save(str(MAPS_DIR / f"labels_trip_{trip_id}.html"))


def main():
    t0 = time.time()
    summaries, pps = [], []
    for split in SPLITS:
        for trip in iter_trips(split):
            s, pp = process(trip)
            summaries.append(s); pps.append(pp)
            print(f"[{split}] {trip.trip_id}: label dist median {s['label_dist_median_m']:.1f} m"
                  f"{'  UNRELIABLE' if s['unreliable'] else ''}")
    summ = pd.DataFrame(summaries)
    summ.to_csv(RESULTS_DIR / "labels_trip_summary.csv", index=False)
    pp = pd.concat(pps, ignore_index=True).merge(summ[["trip_id", "unreliable"]], on="trip_id")

    grid = miss_grid(pp)
    grid.to_csv(RESULTS_DIR / "candidate_miss_rate.csv", index=False)
    figures(grid, pp)
    label_map("00000095", "train")

    show = grid[(grid.subset == "reliable") & grid.k.isin([4, 8, 12, 16, "all"]) & grid.radius_m.isin([30, 40, 50, 75])]
    print(show.pivot_table(index=["split", "radius_m"], columns="k", values="miss_rate").mul(100).round(2))
    print(show.pivot_table(index=["split", "radius_m"], columns="k", values="mean_candidates").round(1))
    print("unreliable trips:", summ[summ.unreliable].trip_id.tolist())
    print(json.dumps({"elapsed_s": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
