"""Candidate-level features: one row per (GPS observation, candidate arc).

Everything here is computed from the GPS track and the road network only — no
reference-route information — so the same code runs at inference time in the API.
Absolute coordinates and timestamps are deliberately excluded so models cannot
memorise locations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .candidates import ArcIndex, Candidates
from .config import CAND_K, CAND_RADIUS_M
from .geo import angle_diff, bearing_xy
from .io import Trip
from .labels import track_xy
from .network import RoadNetwork, point_segment_distance_paired

MIN_MOVE_M = 2.0  # displacement below this gives no reliable heading

FEATURES = [
    # geometry of the candidate relative to the point
    "dist", "rank", "dist_minus_min", "dist_ratio", "frac", "at_endpoint",
    # candidate-set context for the point
    "n_cands", "n_within_radius", "min_dist", "second_gap",
    # arc properties
    "arc_len_log", "two_way", "out_deg_end", "in_deg_start",
    # direction agreement between GPS motion and arc direction
    "heading_cos", "heading_diff", "heading_cos_wide", "heading_valid",
    # motion
    "speed", "dt_prev", "dt_next", "gap_flag",
    # temporal consistency: do neighbouring GPS points also lie near this arc?
    "dist_prev1", "dist_next1", "dist_win5_mean", "dist_win5_max",
]


def _heading(pts: np.ndarray, w: int):
    n = len(pts)
    i = np.arange(n)
    lo, hi = np.clip(i - w, 0, n - 1), np.clip(i + w, 0, n - 1)
    d = pts[hi] - pts[lo]
    valid = np.hypot(d[:, 0], d[:, 1]) >= MIN_MOVE_M
    return bearing_xy(d[:, 0], d[:, 1]), valid


def _neighbour_dist(pts, net, point, arc, offset):
    j = np.clip(point + offset, 0, len(pts) - 1)
    d, _ = point_segment_distance_paired(pts[j], net.a_xy[arc], net.b_xy[arc])
    return d


def build_features(trip: Trip, net: RoadNetwork | None = None, index: ArcIndex | None = None,
                   radius: float = CAND_RADIUS_M, k: int = CAND_K) -> tuple[pd.DataFrame, Candidates]:
    """Return (feature frame aligned with candidates, candidates)."""
    net = net or RoadNetwork.from_trip(trip)
    index = index or ArcIndex(net)
    pts = track_xy(trip, net)
    full = index.query(pts, radius)
    n_within = np.bincount(full.point, minlength=len(pts))
    c = full.subset(k=k)
    p, a = c.point, c.arc

    counts = c.counts()
    off = c.offsets()
    has = counts > 0
    min_dist = np.full(len(pts), np.nan)
    min_dist[has] = c.dist[off[:-1][has]]
    has2 = counts > 1
    second = np.full(len(pts), np.nan)
    second[has2] = c.dist[off[:-1][has2] + 1]

    # arc topology
    n = net.n_nodes
    keys = net.u.astype(np.int64) * n + net.v
    two_way = np.isin(net.v.astype(np.int64) * n + net.u, keys)
    out_deg = np.bincount(net.u, minlength=n)
    in_deg = np.bincount(net.v, minlength=n)

    # motion
    t = trip.track.t.to_numpy()
    dt = np.diff(t)
    dt_prev = np.concatenate([[np.nan], dt])
    dt_next = np.concatenate([dt, [np.nan]])
    step = np.hypot(*np.diff(pts, axis=0).T)
    lo, hi = np.clip(np.arange(len(pts)) - 1, 0, len(pts) - 1), np.clip(np.arange(len(pts)) + 1, 0, len(pts) - 1)
    span_t = np.maximum(t[hi] - t[lo], 1e-6)
    span_d = np.concatenate([[0], np.cumsum(step)])
    speed = (span_d[hi] - span_d[lo]) / span_t

    h1, v1 = _heading(pts, 1)
    h3, v3 = _heading(pts, 3)
    diff1 = angle_diff(h1[p], net.bearing[a])
    diff3 = angle_diff(h3[p], net.bearing[a])
    valid1, valid3 = v1[p], v3[p]

    d_prev1 = _neighbour_dist(pts, net, p, a, -1)
    d_next1 = _neighbour_dist(pts, net, p, a, +1)
    win = np.column_stack([c.dist, d_prev1, d_next1,
                           _neighbour_dist(pts, net, p, a, -2), _neighbour_dist(pts, net, p, a, +2)])

    f = pd.DataFrame({
        "dist": c.dist,
        "rank": c.rank,
        "dist_minus_min": c.dist - min_dist[p],
        "dist_ratio": (c.dist + 1.0) / (min_dist[p] + 1.0),
        "frac": c.frac,
        "at_endpoint": ((c.frac <= 1e-6) | (c.frac >= 1 - 1e-6)).astype(np.int8),
        "n_cands": counts[p],
        "n_within_radius": n_within[p],
        "min_dist": min_dist[p],
        "second_gap": np.nan_to_num(second[p] - min_dist[p], nan=radius),
        "arc_len_log": np.log1p(net.length[a]),
        "two_way": two_way[a].astype(np.int8),
        "out_deg_end": out_deg[net.v[a]],
        "in_deg_start": in_deg[net.u[a]],
        "heading_cos": np.where(valid1, np.cos(np.radians(diff1)), 0.0),
        "heading_diff": np.where(valid1, diff1, 90.0),
        "heading_cos_wide": np.where(valid3, np.cos(np.radians(diff3)), 0.0),
        "heading_valid": valid1.astype(np.int8),
        "speed": speed[p],
        "dt_prev": np.nan_to_num(dt_prev[p], nan=1.0),
        "dt_next": np.nan_to_num(dt_next[p], nan=1.0),
        "gap_flag": ((np.nan_to_num(dt_prev[p], nan=1.0) > 5) | (np.nan_to_num(dt_next[p], nan=1.0) > 5)).astype(np.int8),
        "dist_prev1": d_prev1,
        "dist_next1": d_next1,
        "dist_win5_mean": win.mean(1),
        "dist_win5_max": win.max(1),
    })
    f = f.astype({col: np.float32 for col in FEATURES if f[col].dtype == np.float64})
    return f[FEATURES], c
