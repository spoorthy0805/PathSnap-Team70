"""Phase 6 — HMM + Viterbi route reconstruction on top of the tuned XGBoost scores.

Protocol
1. Out-of-fold (OOF) candidate probabilities on the 78 reliable training trips, using the tuned
   XGBoost config and the same 5 trip-grouped folds as Phases 4–5 (each trip is scored by a model
   that never saw it).
2. Decoder parameters (emission weight alpha, transition scale beta, gap threshold) are chosen by
   grid search on the training OOF predictions only (objective: observation Top-1).
3. student_test is decoded once with the chosen parameters using the final tuned model's
   probabilities (Phase 5), and compared with classifier-only predictions.

Outputs (experiments/results/phase6/)
  <model>/decoder_grid.csv          Top-1 / continuity for every decoder setting (train OOF)
  decoder_params.json       chosen parameters
  per_trip.csv              classifier-only vs HMM per trip (train OOF + student_test)
  summary.csv               pooled comparison per split
reports/figures/hmm_*.png, reports/maps/hmm_trip_<id>.html
cache/preds/xgb_tuned_oof_train.parquet
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathsnap.config import CACHE_DIR, FIGURES_DIR, MAPS_DIR, RESULTS_DIR, SEED  # noqa: E402
from pathsnap.experiment import cv_splits, load_tables  # noqa: E402
from pathsnap.features import FEATURES  # noqa: E402
from pathsnap.hmm import (RouteGraph, build_lattice, classifier_only, reconstruct_route, route_overlap,  # noqa: E402
                          viterbi)
from pathsnap.io import load_trip  # noqa: E402
from pathsnap.labels import track_xy  # noqa: E402
from pathsnap.models import ZOO  # noqa: E402
from pathsnap.metrics import route_continuity  # noqa: E402
from pathsnap.network import RoadNetwork  # noqa: E402
from pathsnap.viz import trip_map  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

OUT = RESULTS_DIR / "phase6"  # per-model subfolder set in main()
FIG_PREFIX = "hmm_"
GRID = {"alpha": [0.5, 1.0, 2.0], "beta": [2.0, 5.0, 10.0, 25.0], "gap_s": [30.0]}
GAP_GRID = [10.0, 30.0, 120.0]
UNREACHABLE_PENALTY = 30.0


def oof_predictions(train: pd.DataFrame, params: dict, model: str) -> np.ndarray:
    path = CACHE_DIR / "preds" / f"{model}_oof_train.parquet"
    if path.exists():
        return pd.read_parquet(path).p.to_numpy()
    X, y = train[FEATURES].to_numpy(np.float32), train.y.to_numpy()
    p = np.zeros(len(train))
    for k, (tr, va) in enumerate(cv_splits(train)):
        est = XGBClassifier(**params, tree_method="hist", eval_metric="logloss", random_state=SEED, n_jobs=-1)
        p[va] = est.fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
        print(f"  OOF fold {k} done", flush=True)
    train[["trip_id", "point", "arc", "y"]].assign(p=p).to_parquet(path)
    return p


class TripCase:
    """Everything needed to decode and score one trip, built once."""

    def __init__(self, trip_id: str, split: str, frame: pd.DataFrame):
        trip = load_trip(trip_id, split)
        self.trip_id, self.split, self.trip = trip_id, split, trip
        self.net = RoadNetwork.from_trip(trip)
        self.graph = RouteGraph(self.net)
        self.frame = frame.reset_index(drop=True)
        self.n_points = len(trip.track)
        self.lat = build_lattice(self.frame, track_xy(trip, self.net), trip.track.t.to_numpy(), self.graph)

    def score(self, rows: np.ndarray, with_route: bool = False) -> dict:
        y = self.frame.y.to_numpy()[rows]
        arcs = self.frame.arc.to_numpy()[rows]
        out = {"correct": int(y.sum()), "n_points": self.n_points,
               "continuity": route_continuity(self.lat.points, arcs, self.net.u, self.net.v),
               "n_pairs": max(len(arcs) - 1, 0)}
        if with_route:
            route, gaps = reconstruct_route(arcs, self.graph)
            out |= route_overlap(route, self.trip.route, self.net.length) | {"route_gaps": gaps, "route": route}
        return out


def pooled(rows: list[dict]) -> dict:
    d = pd.DataFrame(rows)
    out = {"top1": d.correct.sum() / d.n_points.sum(),
           "continuity": (d.continuity * d.n_pairs).sum() / d.n_pairs.sum()}
    for c in ["route_precision", "route_recall", "route_f1"]:
        if c in d:
            out[c] = d[c].mean()  # trip-averaged
    return out


def main():
    global OUT, FIG_PREFIX
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["xgb_tuned", "xgb_untuned"], default="xgb_tuned")
    ap.add_argument("--maps", action="store_true", help="write Folium comparison maps")
    args = ap.parse_args()
    OUT = OUT / args.model
    FIG_PREFIX = "hmm_" if args.model == "xgb_tuned" else f"hmm_{args.model}_"
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    if args.model == "xgb_tuned":
        params_m = json.loads((RESULTS_DIR / "phase5" / "best_params.json").read_text())["params"]
    else:
        params_m = dict(ZOO["xgb"].config)
    train, test = load_tables()
    print(f"OOF predictions ({args.model})...", flush=True)
    train["p"] = oof_predictions(train, params_m, args.model)
    test_pred = pd.read_parquet(CACHE_DIR / "preds" / f"{args.model}_student_test.parquet")
    assert (test_pred[["trip_id", "point", "arc"]].to_numpy() == test[["trip_id", "point", "arc"]].to_numpy()).all()
    test["p"] = test_pred.p.to_numpy()

    print("building lattices...", flush=True)
    t0 = time.time()
    cases = {"train": [TripCase(t, "train", g) for t, g in train.groupby("trip_id", sort=True)],
             "student_test": [TripCase(t, "student_test", g) for t, g in test.groupby("trip_id", sort=True)]}
    lattice_s = time.time() - t0
    print(f"  lattices built in {lattice_s:.1f}s", flush=True)

    # ---- decoder grid search on train OOF ----
    grid_rows = []
    combos = list(itertools.product(GRID["alpha"], GRID["beta"], GRID["gap_s"]))
    for alpha, beta, gap in combos:
        res = [c.score(viterbi(c.lat, alpha, beta, UNREACHABLE_PENALTY, gap)[0]) for c in cases["train"]]
        grid_rows.append({"alpha": alpha, "beta": beta, "gap_s": gap, **pooled(res)})
        print(f"  alpha {alpha} beta {beta} gap {gap}: top1 {grid_rows[-1]['top1']:.4f}", flush=True)
    g0 = max(grid_rows, key=lambda r: r["top1"])
    for gap in GAP_GRID:
        if gap == g0["gap_s"]:
            continue
        res = [c.score(viterbi(c.lat, g0["alpha"], g0["beta"], UNREACHABLE_PENALTY, gap)[0]) for c in cases["train"]]
        grid_rows.append({"alpha": g0["alpha"], "beta": g0["beta"], "gap_s": gap, **pooled(res)})
        print(f"  alpha {g0['alpha']} beta {g0['beta']} gap {gap}: top1 {grid_rows[-1]['top1']:.4f}", flush=True)
    grid = pd.DataFrame(grid_rows)
    grid.to_csv(OUT / "decoder_grid.csv", index=False)
    bestd = grid.loc[grid.top1.idxmax()]
    params = {"alpha": float(bestd.alpha), "beta": float(bestd.beta), "gap_s": float(bestd.gap_s),
              "unreachable_penalty": UNREACHABLE_PENALTY, "route_limit_m": 1000.0,
              "classifier": args.model, "selected_on": "train OOF (78 trips), objective = observation Top-1"}
    (OUT / "decoder_params.json").write_text(json.dumps(params, indent=2))
    print("chosen decoder params:", params, flush=True)

    # ---- classifier-only vs HMM ----
    per_trip, summary = [], []
    for split, cs in cases.items():
        res_c, res_h = [], []
        for c in cs:
            t0 = time.perf_counter()
            rows_h, diag = viterbi(c.lat, params["alpha"], params["beta"], UNREACHABLE_PENALTY, params["gap_s"])
            dec_ms = (time.perf_counter() - t0) * 1000
            sc = c.score(classifier_only(c.lat), with_route=True)
            sh = c.score(rows_h, with_route=True)
            c.routes = {"classifier": sc.pop("route"), "hmm": sh.pop("route")}
            res_c.append(sc); res_h.append(sh)
            per_trip.append({"split": split, "trip_id": c.trip_id, "n_points": c.n_points,
                             "top1_classifier": sc["correct"] / c.n_points, "top1_hmm": sh["correct"] / c.n_points,
                             "continuity_classifier": sc["continuity"], "continuity_hmm": sh["continuity"],
                             "route_f1_classifier": sc["route_f1"], "route_f1_hmm": sh["route_f1"],
                             "route_gaps_classifier": sc["route_gaps"], "route_gaps_hmm": sh["route_gaps"],
                             "hmm_breaks": diag["breaks"], "hmm_unreachable_steps": diag["unreachable_steps"],
                             "decode_ms": dec_ms})
        for name, res in [("classifier_only", res_c), ("classifier+hmm", res_h)]:
            summary.append({"split": split, "method": name, **pooled(res),
                            "route_gaps_total": int(sum(r["route_gaps"] for r in res))})
    per_trip = pd.DataFrame(per_trip)
    per_trip["top1_gain"] = per_trip.top1_hmm - per_trip.top1_classifier
    per_trip.to_csv(OUT / "per_trip.csv", index=False)
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / "summary.csv", index=False)
    print(summary.round(4).to_string(index=False))
    print(per_trip[per_trip.split == "student_test"][["trip_id", "top1_classifier", "top1_hmm", "route_f1_classifier",
                                                      "route_f1_hmm", "hmm_breaks"]].round(4).to_string(index=False))

    figures(grid, summary, per_trip)
    if not args.maps:
        print(f"done in {time.time() - t_start:.1f}s (lattices {lattice_s:.1f}s)")
        return
    test_cases = {c.trip_id: c for c in cases["student_test"]}
    pt = per_trip[per_trip.split == "student_test"]
    for tid in {pt.loc[pt.top1_gain.idxmax()].trip_id, pt.loc[pt.top1_gain.idxmin()].trip_id,
                pt.loc[pt.top1_hmm.idxmin()].trip_id}:
        c = test_cases[tid]
        trip_map(c.trip, c.net, predicted_arcs=c.routes["hmm"], classifier_arcs=c.routes["classifier"],
                 predicted_label="HMM route", title=f"Trip {tid}: classifier-only vs HMM") \
            .save(str(MAPS_DIR / f"hmm_trip_{tid}.html"))
    print(f"done in {time.time() - t_start:.1f}s (lattices {lattice_s:.1f}s)")


def figures(grid: pd.DataFrame, summary: pd.DataFrame, per_trip: pd.DataFrame):
    # stage 1: alpha x beta at the initial gap_s; stage 2: gap_s sweep at the best (alpha, beta)
    g = grid[grid.gap_s == grid.gap_s.mode()[0]].drop_duplicates(["alpha", "beta"])
    piv = g.pivot(index="alpha", columns="beta", values="top1")
    best = grid.loc[grid.top1.idxmax()]
    sweep = grid[(grid.alpha == best.alpha) & (grid.beta == best.beta)].sort_values("gap_s")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.4, 1]})
    im = ax.imshow(piv.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(piv.shape[1]), piv.columns); ax.set_yticks(range(piv.shape[0]), piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.4f}", ha="center", va="center", color="w", fontsize=9)
    ax.set(xlabel="beta (transition scale, m)", ylabel="alpha (emission weight)",
           title=f"Stage 1: alpha x beta (gap_s = {g.gap_s.iloc[0]:.0f} s)")
    fig.colorbar(im, ax=ax)
    ax2.plot(sweep.gap_s, sweep.top1, "o-")
    for x, y in zip(sweep.gap_s, sweep.top1):
        ax2.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    ax2.set(xlabel="gap_s (s)", ylabel="Top-1", title=f"Stage 2: gap_s (alpha {best.alpha:g}, beta {best.beta:g})")
    ax2.grid(alpha=0.3)
    fig.suptitle("Decoder grid search: Top-1 on train OOF"); fig.tight_layout()
    fig.savefig(FIGURES_DIR / (FIG_PREFIX + "decoder_grid.png"), dpi=130); plt.close(fig)

    metrics = ["top1", "continuity", "route_f1"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    for ax, split in zip(axes, ["train", "student_test"]):
        s = summary[summary.split == split].set_index("method")
        x = np.arange(len(metrics)); w = 0.38
        for i, (meth, color) in enumerate([("classifier_only", "#bf8700"), ("classifier+hmm", "#d1242f")]):
            vals = [s.loc[meth, m] for m in metrics]
            ax.bar(x + (i - 0.5) * w, vals, w, label=meth, color=color)
            for j, v in enumerate(vals):
                ax.text(j + (i - 0.5) * w, v + 0.003, f"{v:.3f}", ha="center", fontsize=8)
        ax.set_xticks(x, ["Top-1", "Continuity", "Route F1 (len-weighted)"]); ax.set_ylim(0.7, 1.02)
        ax.set_title(f"{'train (OOF, 78 trips)' if split == 'train' else 'student_test (10 trips)'}")
        ax.legend(loc="lower right"); ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Classifier-only vs classifier + HMM/Viterbi"); fig.tight_layout()
    fig.savefig(FIGURES_DIR / (FIG_PREFIX + "vs_classifier.png"), dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    for ax, split in zip(axes, ["train", "student_test"]):
        d = per_trip[per_trip.split == split].sort_values("top1_classifier")
        ax.scatter(d.top1_classifier, d.top1_hmm, color="#1f6feb")
        lo = min(d.top1_classifier.min(), d.top1_hmm.min()) - 0.02
        ax.plot([lo, 1], [lo, 1], ls="--", color="#9aa0a6")
        ax.set(xlabel="classifier-only Top-1", ylabel="HMM Top-1", title=f"Per trip ({split}); above line = HMM helps")
        ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / (FIG_PREFIX + "per_trip.png"), dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
