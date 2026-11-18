"""Phase 1 — dataset inventory, data-quality checks, plots and Folium maps.

Uses only the train and student_test partitions.

Outputs
  experiments/results/eda_inventory.csv   one row per trip
  experiments/results/eda_summary.json    partition-level aggregates
  reports/figures/eda_*.png               plots
  reports/maps/eda_trip_<id>.html         interactive Folium maps
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathsnap.config import FIGURES_DIR, MAPS_DIR, RESULTS_DIR, SPLITS  # noqa: E402
from pathsnap.geo import haversine  # noqa: E402
from pathsnap.io import iter_trips, trip_dir  # noqa: E402
from pathsnap.network import RoadNetwork, min_distance_to_arcs  # noqa: E402
from pathsnap.viz import trip_map  # noqa: E402

for d in (FIGURES_DIR, MAPS_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def trip_stats(trip) -> tuple[dict, dict]:
    tr = trip.track
    lon, lat, t = tr.lon.values, tr.lat.values, tr.t.values
    dt = np.diff(t)
    step = haversine(lon[:-1], lat[:-1], lon[1:], lat[1:])
    speed = np.divide(step, dt, out=np.full_like(step, np.nan), where=dt > 0)
    net = RoadNetwork.from_trip(trip)
    px, py = net.proj.to_xy(lon, lat)
    pts = np.column_stack([px, py])

    d = trip_dir(trip.trip_id, trip.split)
    row = {
        "trip_id": trip.trip_id,
        "split": trip.split,
        "files": len(list(d.iterdir())),
        "size_mb": round(sum(f.stat().st_size for f in d.iterdir()) / 1e6, 2),
        "n_points": len(tr),
        "nan_values": int(tr.isna().sum().sum()),
        "invalid_coords": int(((np.abs(lat) > 90) | (np.abs(lon) > 180) | ((lat == 0) & (lon == 0))).sum()),
        "duration_s": float(t[-1] - t[0]),
        "nonmonotonic_t": int((dt < 0).sum()),
        "duplicate_t": int((dt == 0).sum()),
        "dt_median": float(np.median(dt)),
        "dt_frac_1s": float(np.mean(dt == 1)),
        "dt_max": float(dt.max()),
        "gaps_gt_5s": int((dt > 5).sum()),
        "gaps_gt_30s": int((dt > 30).sum()),
        "stationary_steps": int((step < 0.5).sum()),
        "track_km": float(step.sum() / 1000),
        "speed_median_ms": float(np.nanmedian(speed)),
        "speed_p95_ms": float(np.nanpercentile(speed, 95)),
        "speed_max_ms": float(np.nanmax(speed)),
        "jumps_gt_70ms": int(np.nansum(speed > 70)),
        "center_lon": float(np.median(lon)),
        "center_lat": float(np.median(lat)),
        "extent_x_km": float(np.ptp(px) / 1000),
        "extent_y_km": float(np.ptp(py) / 1000),
        "n_nodes": net.n_nodes,
        "n_arcs": net.n_arcs,
        **net.connectivity_stats(),
    }

    if trip.route is not None:
        r = trip.route
        valid = (r >= 0) & (r < net.n_arcs)
        rv = r[valid]
        contiguous = net.v[rv[:-1]] == net.u[rv[1:]]
        d_route = min_distance_to_arcs(pts, net, np.unique(rv))
        row.update({
            "route_len": len(r),
            "route_invalid_ids": int((~valid).sum()),
            "route_repeat_arcs": int(len(rv) - len(np.unique(rv))),
            "route_contiguity": float(contiguous.mean()) if len(rv) > 1 else np.nan,
            "route_breaks": int((~contiguous).sum()),
            "route_km": float(net.length[rv].sum() / 1000),
            "gps_to_route_median_m": float(np.median(d_route)),
            "gps_to_route_p95_m": float(np.percentile(d_route, 95)),
            "gps_to_route_max_m": float(d_route.max()),
            "points_per_route_arc": len(tr) / len(r),
        })
    else:
        d_route = np.array([])
    extras = {"dt": dt, "speed": speed[np.isfinite(speed)], "d_route": d_route, "arc_len": net.length}
    return row, extras


def plots(inv: pd.DataFrame, pooled: dict):
    split_colors = {"train": "#1f6feb", "student_test": "#d1242f"}

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for s, g in inv.groupby("split"):
        ax[0].hist(g.n_points, bins=30, alpha=0.6, label=s, color=split_colors[s])
        ax[1].hist(g.track_km, bins=30, alpha=0.6, label=s, color=split_colors[s])
        ax[2].hist(g.n_arcs, bins=30, alpha=0.6, label=s, color=split_colors[s])
    for a, t in zip(ax, ["GPS points per trip", "Track length (km)", "Road arcs per trip"]):
        a.set_title(t); a.legend()
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "eda_trip_sizes.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    dt = pooled["train"]["dt"]
    vals, counts = np.unique(np.clip(dt, 0, 11), return_counts=True)
    ax[0].bar(vals, counts, color="#1f6feb"); ax[0].set_yscale("log")
    ax[0].set_title("Sampling interval (train, s; 11 = ≥11)")
    ax[1].hist(np.clip(pooled["train"]["speed"], 0, 60), bins=60, color="#1f6feb")
    ax[1].set_title("Point-to-point speed (train, m/s, clipped 60)")
    for s in SPLITS:
        ax[2].hist(np.clip(pooled[s]["d_route"], 0, 100), bins=100, alpha=0.6, density=True,
                   label=s, color=split_colors[s])
    ax[2].set_title("GPS distance to reference route (m, clipped 100)"); ax[2].legend()
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "eda_sampling_speed_noise.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for s, g in inv.groupby("split"):
        ax.scatter(g.center_lon, g.center_lat, s=np.sqrt(g.n_points) * 2, alpha=0.7, label=s, color=split_colors[s])
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.set_title("Trip locations (size ∝ √points)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "eda_trip_locations.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].scatter(inv.n_nodes, inv.n_arcs, c=inv.split.map(split_colors), alpha=0.7)
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("nodes"); ax[0].set_ylabel("arcs")
    ax[0].set_title("Road-network size")
    ax[1].scatter(inv.weak_components, inv.oneway_arc_frac, c=inv.split.map(split_colors), alpha=0.7)
    ax[1].set_xscale("log"); ax[1].set_xlabel("weakly connected components"); ax[1].set_ylabel("one-way arc fraction")
    ax[1].set_title("Road-network connectivity")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "eda_network.png", dpi=130); plt.close(fig)

    miss = inv[["split", "nan_values", "invalid_coords", "nonmonotonic_t", "duplicate_t", "route_invalid_ids"]]
    miss = miss.groupby("split").sum().T
    fig, ax = plt.subplots(figsize=(7, 3.5))
    miss.plot.barh(ax=ax, color=[split_colors[c] for c in miss.columns])
    ax.set_title("Missing / invalid value counts per partition"); ax.set_xlabel("count")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "eda_missing_values.png", dpi=130); plt.close(fig)


def main():
    t0 = time.time()
    rows, pooled = [], {}
    map_trips = []
    for split in SPLITS:
        acc = {"dt": [], "speed": [], "d_route": [], "arc_len": []}
        for trip in iter_trips(split):
            row, ex = trip_stats(trip)
            rows.append(row)
            for k in acc:
                acc[k].append(ex[k])
            print(f"[{split}] {trip.trip_id}: {row['n_points']} pts, {row['n_arcs']} arcs, "
                  f"gps→route median {row.get('gps_to_route_median_m', float('nan')):.1f} m")
        pooled[split] = {k: np.concatenate(v) for k, v in acc.items()}
    inv = pd.DataFrame(rows)
    inv.to_csv(RESULTS_DIR / "eda_inventory.csv", index=False)

    summary = {}
    for s, g in inv.groupby("split"):
        summary[s] = {
            "trips": int(len(g)),
            "points_total": int(g.n_points.sum()),
            "points_per_trip": g.n_points.describe().round(1).to_dict(),
            "track_km_total": round(float(g.track_km.sum()), 1),
            "dt_frac_1s_pooled": float(np.mean(pooled[s]["dt"] == 1)),
            "trips_with_gaps_gt_30s": int((g.gaps_gt_30s > 0).sum()),
            "trips_with_jumps_gt_70ms": int((g.jumps_gt_70ms > 0).sum()),
            "nan_values": int(g.nan_values.sum()),
            "invalid_coords": int(g.invalid_coords.sum()),
            "nonmonotonic_t": int(g.nonmonotonic_t.sum()),
            "isolated_nodes_total": int(g.isolated_nodes.sum()),
            "trips_multi_weak_components": int((g.weak_components > 1).sum()),
            "route_contiguity_mean": float(g.route_contiguity.mean()),
            "trips_with_route_breaks": int((g.route_breaks > 0).sum()),
            "gps_to_route_median_m_pooled": float(np.median(pooled[s]["d_route"])),
            "gps_to_route_p95_m_pooled": float(np.percentile(pooled[s]["d_route"], 95)),
            "gps_to_route_p99_m_pooled": float(np.percentile(pooled[s]["d_route"], 99)),
            "arc_len_median_m_pooled": float(np.median(pooled[s]["arc_len"])),
        }
    (RESULTS_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    plots(inv, pooled)

    # Folium maps (train): a typical trip, the trip with the worst p95 GPS→route distance, the trip with
    # the largest systematic GPS→route offset (median), and the trip with the largest time gap;
    # plus one student_test trip.
    tr = inv[inv.split == "train"]
    picks = {
        tr.iloc[(tr.n_points - tr.n_points.median()).abs().argsort().iloc[0]].trip_id: "train",
        tr.loc[tr.gps_to_route_p95_m.idxmax()].trip_id: "train",
        tr.loc[tr.gps_to_route_median_m.idxmax()].trip_id: "train",
        tr.loc[tr.dt_max.idxmax()].trip_id: "train",
        inv[inv.split == "student_test"].iloc[0].trip_id: "student_test",
    }
    from pathsnap.io import load_trip
    for tid, split in picks.items():
        trip_map(load_trip(tid, split)).save(str(MAPS_DIR / f"eda_trip_{tid}.html"))
        map_trips.append(tid)

    print(json.dumps(summary, indent=2))
    print(f"maps: {map_trips}  |  done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
