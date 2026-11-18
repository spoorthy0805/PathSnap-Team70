"""Road-network geometry and connectivity for a single trip."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .geo import LocalProjection, bearing_xy
from .io import Trip


@dataclass
class RoadNetwork:
    proj: LocalProjection
    node_xy: np.ndarray   # (N, 2) metres
    u: np.ndarray         # (A,) from-node per arc
    v: np.ndarray         # (A,) to-node per arc
    a_xy: np.ndarray      # (A, 2) arc start
    b_xy: np.ndarray      # (A, 2) arc end
    length: np.ndarray    # (A,) metres
    bearing: np.ndarray   # (A,) degrees from north

    @property
    def n_nodes(self) -> int:
        return len(self.node_xy)

    @property
    def n_arcs(self) -> int:
        return len(self.u)

    @classmethod
    def from_trip(cls, trip: Trip, proj: LocalProjection | None = None) -> "RoadNetwork":
        if proj is None:
            proj = LocalProjection.for_track(trip.track.lon.values, trip.track.lat.values)
        nx, ny = proj.to_xy(trip.nodes.lon.values, trip.nodes.lat.values)
        node_xy = np.column_stack([nx, ny])
        u = trip.arcs.u.to_numpy()
        v = trip.arcs.v.to_numpy()
        a_xy, b_xy = node_xy[u], node_xy[v]
        d = b_xy - a_xy
        return cls(proj, node_xy, u, v, a_xy, b_xy, np.hypot(d[:, 0], d[:, 1]), bearing_xy(d[:, 0], d[:, 1]))

    # ---------- connectivity ----------
    def _adj(self):
        n = self.n_nodes
        return coo_matrix((np.ones(self.n_arcs), (self.u, self.v)), shape=(n, n)).tocsr()

    def connectivity_stats(self) -> dict:
        adj = self._adj()
        deg = np.bincount(self.u, minlength=self.n_nodes) + np.bincount(self.v, minlength=self.n_nodes)
        n_weak, weak_lbl = connected_components(adj, directed=True, connection="weak")
        n_strong, strong_lbl = connected_components(adj, directed=True, connection="strong")
        pairs = set(zip(self.u.tolist(), self.v.tolist()))
        two_way = np.fromiter(((b, a) in pairs for a, b in zip(self.u.tolist(), self.v.tolist())), bool, self.n_arcs)
        return {
            "isolated_nodes": int((deg == 0).sum()),
            "weak_components": int(n_weak),
            "largest_weak_frac": float(np.bincount(weak_lbl).max() / self.n_nodes),
            "strong_components": int(n_strong),
            "largest_strong_frac": float(np.bincount(strong_lbl).max() / self.n_nodes),
            "oneway_arc_frac": float(1 - two_way.mean()),
            "self_loops": int((self.u == self.v).sum()),
            "zero_length_arcs": int((self.length < 1e-6).sum()),
            "arc_len_median_m": float(np.median(self.length)),
            "arc_len_p95_m": float(np.percentile(self.length, 95)),
        }


def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Distance from points p (P,2) to segments a->b (S,2), broadcast to (P,S).

    Returns (dist, t) where t in [0,1] is the projection fraction along each segment.
    """
    ab = b - a                                     # (S,2)
    ab2 = np.maximum((ab ** 2).sum(-1), 1e-12)     # (S,)
    ap = p[:, None, :] - a[None, :, :]             # (P,S,2)
    t = np.clip((ap * ab[None]).sum(-1) / ab2, 0.0, 1.0)
    proj = a[None] + t[..., None] * ab[None]
    return np.hypot(*(p[:, None, :] - proj).transpose(2, 0, 1)), t


def point_segment_distance_paired(p: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Row-wise distance from p[i] to segment a[i]->b[i]. Returns (dist, t), each (N,)."""
    ab = b - a
    ab2 = np.maximum((ab ** 2).sum(-1), 1e-12)
    t = np.clip(((p - a) * ab).sum(-1) / ab2, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return np.hypot(*(p - proj).T), t


def min_distance_to_arcs(p: np.ndarray, net: RoadNetwork, arc_ids: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """For each point, the distance to the nearest of the given arcs (metres)."""
    a, b = net.a_xy[arc_ids], net.b_xy[arc_ids]
    out = np.empty(len(p))
    for i in range(0, len(p), chunk):
        d, _ = point_segment_distance(p[i:i + chunk], a, b)
        out[i:i + chunk] = d.min(1)
    return out
