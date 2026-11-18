"""Per-observation reference labels.

`.route` is an ordered list of arc ids, not one label per GPS point. We derive a
label for each GPS observation by a **monotone alignment** of the track to the
route: point i is assigned route position j(i), with j non-decreasing in i
(the vehicle never goes backwards along its own route), minimising the total
point-to-arc distance

    DP[i, j] = d(i, j) + min_{k <= j} DP[i-1, k]

which is O(P * R) with a running prefix-minimum. Working on route *positions*
(not arc ids) handles routes that revisit the same arc. Skipping positions is
allowed (short arcs traversed between two 1 Hz fixes get no point).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io import Trip
from .network import RoadNetwork, point_segment_distance, point_segment_distance_paired


@dataclass
class TrackLabels:
    route_pos: np.ndarray   # (P,) index into trip.route
    arc: np.ndarray         # (P,) arc id label
    dist: np.ndarray        # (P,) distance (m) from point to its label arc
    frac: np.ndarray        # (P,) projection fraction along label arc


def track_xy(trip: Trip, net: RoadNetwork) -> np.ndarray:
    x, y = net.proj.to_xy(trip.track.lon.values, trip.track.lat.values)
    return np.column_stack([x, y])


def align_track_to_route(trip: Trip, net: RoadNetwork, pts: np.ndarray | None = None) -> TrackLabels:
    if trip.route is None:
        raise ValueError(f"trip {trip.trip_id} has no route")
    pts = track_xy(trip, net) if pts is None else pts
    route = trip.route
    a, b = net.a_xy[route], net.b_xy[route]
    P, R = len(pts), len(route)
    back = np.empty((P, R), dtype=np.int32 if R >= 32767 else np.int16)

    d0, _ = point_segment_distance(pts[:1], a, b)
    prev = d0[0]
    back[0] = np.arange(R)
    idx = np.arange(R)
    for i in range(1, P):
        # running prefix argmin of prev
        best_k = np.maximum.accumulate(np.where(prev == np.minimum.accumulate(prev), idx, 0))
        di, _ = point_segment_distance(pts[i:i + 1], a, b)
        prev = di[0] + prev[best_k]
        back[i] = best_k

    pos = np.empty(P, dtype=np.int64)
    pos[-1] = int(np.argmin(prev))
    for i in range(P - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]

    d, t = point_segment_distance_paired(pts, a[pos], b[pos])
    return TrackLabels(pos, route[pos], d, t)
