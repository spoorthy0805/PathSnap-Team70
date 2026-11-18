#!/usr/bin/env python3
"""Automated dataset validation and data-quality report for PathSnap.

This script re-reads the raw trip files directly (independently of the main
pipeline) and checks every structural and value-level property the rest of the
project relies on. It writes a machine-readable report to
``experiments/results/data_quality_report.json`` and renders three summary
figures to ``reports/figures/``.

Per-trip checks
---------------
* required files present (``.track``, ``.nodes``, ``.arcs``; ``.route`` optional)
* correct column counts and numeric parsing
* ``.track``  : finite values, longitude/latitude within valid bounds,
                strictly increasing timestamps, sampling-gap statistics
* ``.nodes``  : finite values, coordinates within valid bounds
* ``.arcs``   : integer node indices within ``[0, n_nodes)``, self-loops
* ``.route``  : arc ids within ``[0, n_arcs)``, contiguity of the arc chain
* isolated nodes (nodes never referenced by an arc)

The report also cross-references the committed EDA outputs
(``labels_trip_summary.csv``) to list the trips flagged as unreliable by the
training rule (median point->label distance > 50 m).

Usage
-----
    python scripts/validate_dataset.py                 # from the repo root
    python scripts/validate_dataset.py --no-figures
    python scripts/validate_dataset.py --data-root data --results-dir experiments/results
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---- validation thresholds (documented, not tuned) ------------------------
LON_BOUNDS = (-180.0, 180.0)
LAT_BOUNDS = (-90.0, 90.0)
GAP_THRESHOLD_S = 30.0          # a sampling gap larger than this is flagged
UNRELIABLE_RULE_M = 50.0        # matches src/pathsnap/config.UNRELIABLE_TRIPS_RULE_M
PARTITIONS = ("train", "student_test")


def _read_matrix(path: Path, expected_cols: int, integer: bool = False):
    """Read a whitespace-separated file; return (array, error_or_None)."""
    if not path.exists():
        return None, "missing"
    try:
        frame = pd.read_csv(path, sep=r"\s+", header=None, dtype=float)
    except Exception as exc:  # malformed / unreadable
        return None, f"unreadable ({exc.__class__.__name__})"
    if frame.shape[1] != expected_cols:
        return None, f"expected {expected_cols} columns, found {frame.shape[1]}"
    arr = frame.to_numpy()
    if integer:
        if not np.allclose(arr, np.round(arr)):
            return None, "non-integer values"
        arr = np.round(arr).astype(np.int64)
    return arr, None


def validate_trip(trip_dir: Path, trip_id: str, partition: str) -> dict:
    """Run all checks for one trip and return a per-trip record."""
    rec: dict = {"trip_id": trip_id, "split": partition, "errors": [], "warnings": []}
    base = trip_dir / trip_id

    track, err = _read_matrix(base.with_suffix(".track"), 3)
    nodes, err_n = _read_matrix(base.with_suffix(".nodes"), 2)
    arcs, err_a = _read_matrix(base.with_suffix(".arcs"), 2, integer=True)
    for name, e in (("track", err), ("nodes", err_n), ("arcs", err_a)):
        if e:
            rec["errors"].append(f"{name}: {e}")
    if track is None or nodes is None or arcs is None:
        rec["status"] = "fail"
        return rec

    # --- track checks ------------------------------------------------------
    lon, lat, t = track[:, 0], track[:, 1], track[:, 2]
    rec["n_points"] = int(len(track))
    rec["nan_values"] = int(np.count_nonzero(~np.isfinite(track)))
    rec["invalid_coords"] = int(np.count_nonzero(
        (lon < LON_BOUNDS[0]) | (lon > LON_BOUNDS[1])
        | (lat < LAT_BOUNDS[0]) | (lat > LAT_BOUNDS[1])
    ))
    dt_steps = np.diff(t)
    rec["nonmonotonic_t"] = int(np.count_nonzero(dt_steps <= 0))
    rec["duplicate_t"] = int(np.count_nonzero(dt_steps == 0))
    rec["dt_frac_1s"] = float(np.mean(np.isclose(dt_steps, 1.0))) if len(dt_steps) else 1.0
    rec["max_gap_s"] = float(dt_steps.max()) if len(dt_steps) else 0.0
    rec["gaps_gt_30s"] = int(np.count_nonzero(dt_steps > GAP_THRESHOLD_S))

    # --- nodes checks ------------------------------------------------------
    rec["n_nodes"] = int(len(nodes))
    rec["nan_values"] += int(np.count_nonzero(~np.isfinite(nodes)))
    rec["invalid_coords"] += int(np.count_nonzero(
        (nodes[:, 0] < LON_BOUNDS[0]) | (nodes[:, 0] > LON_BOUNDS[1])
        | (nodes[:, 1] < LAT_BOUNDS[0]) | (nodes[:, 1] > LAT_BOUNDS[1])
    ))

    # --- arcs checks -------------------------------------------------------
    rec["n_arcs"] = int(len(arcs))
    out_of_range = int(np.count_nonzero((arcs < 0) | (arcs >= len(nodes))))
    rec["arc_index_out_of_range"] = out_of_range
    rec["self_loops"] = int(np.count_nonzero(arcs[:, 0] == arcs[:, 1]))
    referenced = np.unique(arcs[(arcs >= 0) & (arcs < len(nodes))])
    rec["isolated_nodes"] = int(len(nodes) - len(referenced))

    # --- route checks ------------------------------------------------------
    route, err_r = _read_matrix(base.with_suffix(".route"), 1, integer=True)
    if err_r == "missing":
        rec["has_route"] = False
    elif route is None:
        rec["has_route"] = False
        rec["errors"].append(f"route: {err_r}")
    else:
        route = route.ravel()
        rec["has_route"] = True
        rec["route_len"] = int(len(route))
        rec["route_arc_out_of_range"] = int(np.count_nonzero((route < 0) | (route >= len(arcs))))
        valid = route[(route >= 0) & (route < len(arcs))]
        if len(valid) >= 2:
            seg = arcs[valid]
            shares = [bool(set(seg[i]) & set(seg[i + 1])) for i in range(len(seg) - 1)]
            rec["route_contiguity"] = float(np.mean(shares))
            rec["route_breaks"] = int(np.count_nonzero(~np.array(shares)))
        else:
            rec["route_contiguity"] = None
            rec["route_breaks"] = 0

    # --- grade the trip ----------------------------------------------------
    hard = (rec["nan_values"] + rec["invalid_coords"] + rec["nonmonotonic_t"]
            + rec["arc_index_out_of_range"] + rec.get("route_arc_out_of_range", 0))
    if rec["errors"] or hard > 0:
        rec["status"] = "fail"
    elif rec["gaps_gt_30s"] or rec.get("route_breaks", 0):
        rec["status"] = "warn"
    else:
        rec["status"] = "pass"
    return rec


def validate_dataset(data_root: Path) -> list[dict]:
    records = []
    for partition in PARTITIONS:
        part_dir = data_root / partition
        if not part_dir.is_dir():
            continue
        for trip_dir in sorted(p for p in part_dir.iterdir() if p.is_dir()):
            records.append(validate_trip(trip_dir, trip_dir.name, partition))
    return records


def summarise(records: list[dict], results_dir: Path) -> dict:
    """Aggregate per-trip records, cross-reference label reliability, grade."""
    labels_csv = results_dir / "labels_trip_summary.csv"
    unreliable: list[str] = []
    if labels_csv.exists():
        labels = pd.read_csv(labels_csv)
        if "unreliable" in labels.columns:
            unreliable = labels.loc[labels["unreliable"].astype(str) == "True", "trip_id"]
            unreliable = [f"{int(t):08d}" for t in unreliable]

    report: dict = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "validator": "scripts/validate_dataset.py",
        "thresholds": {
            "lon_bounds": LON_BOUNDS, "lat_bounds": LAT_BOUNDS,
            "gap_threshold_s": GAP_THRESHOLD_S, "unreliable_rule_m": UNRELIABLE_RULE_M,
        },
        "partitions": {},
        "totals": {}, "flags": {}, "status": "pass",
    }

    hard_keys = ["nan_values", "invalid_coords", "nonmonotonic_t",
                 "arc_index_out_of_range", "route_arc_out_of_range"]
    for partition in PARTITIONS:
        recs = [r for r in records if r["split"] == partition]
        if not recs:
            continue
        report["partitions"][partition] = {
            "trips": len(recs),
            "points": int(sum(r.get("n_points", 0) for r in recs)),
            "nodes": int(sum(r.get("n_nodes", 0) for r in recs)),
            "arcs": int(sum(r.get("n_arcs", 0) for r in recs)),
            "trips_passed": sum(r["status"] == "pass" for r in recs),
            "trips_warned": sum(r["status"] == "warn" for r in recs),
            "trips_failed": sum(r["status"] == "fail" for r in recs),
            "checks": {k: int(sum(r.get(k, 0) for r in recs)) for k in hard_keys},
            "isolated_nodes": int(sum(r.get("isolated_nodes", 0) for r in recs)),
            "trips_with_gaps_gt_30s": sum(bool(r.get("gaps_gt_30s", 0)) for r in recs),
            "trips_with_route_breaks": sum(bool(r.get("route_breaks", 0)) for r in recs),
        }

    report["totals"] = {
        "trips": len(records),
        "points": int(sum(r.get("n_points", 0) for r in records)),
        "checks": {k: int(sum(r.get(k, 0) for r in records)) for k in hard_keys},
    }
    report["flags"] = {
        "trips_with_gaps_gt_30s": [r["trip_id"] for r in records if r.get("gaps_gt_30s")],
        "trips_with_route_breaks": [r["trip_id"] for r in records if r.get("route_breaks")],
        "trips_failed": [r["trip_id"] for r in records if r["status"] == "fail"],
        "unreliable_for_training": unreliable,
    }

    if any(v for v in report["totals"]["checks"].values()) or report["flags"]["trips_failed"]:
        report["status"] = "fail"
    elif report["flags"]["trips_with_gaps_gt_30s"] or report["flags"]["trips_with_route_breaks"]:
        report["status"] = "pass_with_warnings"
    report["trips"] = records
    return report


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def render_figures(report: dict, results_dir: Path, figures_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    figures_dir.mkdir(parents=True, exist_ok=True)
    inv = pd.read_csv(results_dir / "eda_inventory.csv")
    ink, grid = "#1d1d1f", "#d2d2d7"

    # 1. Quality summary -----------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle("PathSnap — dataset quality summary", fontweight="bold")

    checks = report["totals"]["checks"]
    names = ["NaN\nvalues", "Invalid\ncoords", "Non-monotonic\ntime",
             "Arc idx\nout-of-range", "Route idx\nout-of-range"]
    ax[0, 0].bar(names, list(checks.values()), color="#30d158")
    ax[0, 0].set_title("Integrity violations (lower is better)")
    ax[0, 0].set_ylim(0, max(1, max(checks.values()) * 1.2))
    for i, v in enumerate(checks.values()):
        ax[0, 0].text(i, v, str(v), ha="center", va="bottom")

    for part, colour in (("train", "#0071e3"), ("student_test", "#ff9f0a")):
        d = inv[inv["split"] == part]
        ax[0, 1].scatter(d["track_km"], d["gps_to_route_median_m"], s=18, alpha=.7,
                         color=colour, label=part)
    ax[0, 1].set(title="GPS→route median offset vs trip length",
                 xlabel="track length (km)", ylabel="median offset (m)")
    ax[0, 1].legend(frameon=False)

    ax[1, 0].hist(inv["dt_frac_1s"], bins=20, color="#5e9ced", edgecolor="white")
    ax[1, 0].set(title="Fraction of 1 Hz sampling steps per trip",
                 xlabel="fraction of steps == 1 s", ylabel="trips")

    miss = pd.read_csv(results_dir / "candidate_miss_rate.csv")
    for part, colour in (("train", "#0071e3"), ("student_test", "#ff9f0a")):
        d = miss[(miss["split"] == part) & (miss["subset"] == "all") & (miss["k"] == "12")]
        if d.empty:
            d = miss[(miss["split"] == part) & (miss["subset"] == "all")]
            d = d[d["k"].astype(str) == "12"]
        d = d.sort_values("radius_m")
        ax[1, 1].plot(d["radius_m"].astype(float), d["miss_rate"].astype(float) * 100,
                      marker="o", ms=3, color=colour, label=part)
    ax[1, 1].set(title="Candidate miss rate vs search radius (k=12)",
                 xlabel="radius (m)", ylabel="miss rate (%)")
    ax[1, 1].legend(frameon=False)
    for a in ax.ravel():
        a.grid(axis="y", color=grid, lw=.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(figures_dir / "dataset_quality_summary.png", bbox_inches="tight")
    plt.close(fig)

    # 2. Trip sampling distribution -----------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    fig.suptitle("PathSnap — trip sampling & size distribution", fontweight="bold")
    ax[0].hist(inv["n_points"], bins=25, color="#0071e3", edgecolor="white")
    ax[0].set(title="GPS observations per trip", xlabel="points", ylabel="trips")
    ax[1].hist(np.clip(inv["dt_max"], 0, 300), bins=25, color="#ff9f0a", edgecolor="white")
    ax[1].set(title="Largest sampling gap per trip (clipped 300 s)",
              xlabel="max gap (s)", ylabel="trips")
    ax[2].hist(inv["speed_median_ms"] * 3.6, bins=25, color="#30d158", edgecolor="white")
    ax[2].set(title="Median speed per trip", xlabel="km/h", ylabel="trips")
    for a in ax.ravel():
        a.grid(axis="y", color=grid, lw=.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(figures_dir / "trip_sampling_distribution.png", bbox_inches="tight")
    plt.close(fig)

    # 3. Label distribution analysis ----------------------------------------
    lab = pd.read_csv(results_dir / "labels_trip_summary.csv")
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    fig.suptitle("PathSnap — reference-label quality", fontweight="bold")
    ax[0].hist(np.clip(lab["label_dist_median_m"], 0, 60), bins=25,
               color="#0071e3", edgecolor="white")
    ax[0].axvline(UNRELIABLE_RULE_M, color="#ff3b30", ls="--", lw=1.2,
                  label=f"unreliable > {UNRELIABLE_RULE_M:g} m")
    ax[0].set(title="Median point→label distance per trip", xlabel="metres", ylabel="trips")
    ax[0].legend(frameon=False)
    ax[1].hist(lab["route_positions_used"], bins=20, color="#ff9f0a", edgecolor="white")
    ax[1].set(title="Fraction of route positions used by labels",
              xlabel="fraction", ylabel="trips")
    rel = lab["unreliable"].astype(str).value_counts()
    ax[2].bar(["reliable", "unreliable"],
              [int(rel.get("False", 0)), int(rel.get("True", 0))],
              color=["#30d158", "#ff3b30"])
    ax[2].set(title="Trips by label reliability", ylabel="trips")
    for a in ax.ravel():
        a.grid(axis="y", color=grid, lw=.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(figures_dir / "label_distribution_analysis.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate the PathSnap dataset.")
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--data-root", type=Path, default=root / "data")
    ap.add_argument("--results-dir", type=Path, default=root / "experiments" / "results")
    ap.add_argument("--figures-dir", type=Path, default=root / "reports" / "figures")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    print(f"Validating dataset under {args.data_root} ...")
    records = validate_dataset(args.data_root)
    report = summarise(records, args.results_dir)

    out = args.results_dir / "data_quality_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"  {report['totals']['trips']} trips, "
          f"{report['totals']['points']:,} GPS observations")
    print(f"  integrity violations: {report['totals']['checks']}")
    print(f"  status: {report['status']}")
    print(f"  report -> {out}")

    if not args.no_figures:
        try:
            render_figures(report, args.results_dir, args.figures_dir)
            print(f"  figures -> {args.figures_dir}")
        except FileNotFoundError as exc:
            print(f"  figures skipped (missing EDA csv): {exc}")


if __name__ == "__main__":
    main()
