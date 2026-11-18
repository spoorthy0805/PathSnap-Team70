"""Candidate road segments for each GPS observation.

Arcs are straight segments between two nodes, but some are hundreds of metres
long, so indexing only arc midpoints would miss long arcs passing close to a
point. We split every arc into pieces of at most `piece_len` metres, index the
piece midpoints in a KD-tree, gather all pieces within `radius + piece_len/2`
of each point, map them back to arc ids, and compute the exact point-to-arc
distance. Each point keeps the arcs within `radius`, nearest first, capped at `k`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .network import RoadNetwork, point_segment_distance_paired


@dataclass
class Candidates:
    """Flat candidate table, sorted by (point, rank)."""
    point: np.ndarray   # (C,) GPS observation index
    arc: np.ndarray     # (C,) arc id
    dist: np.ndarray    # (C,) metres from point to arc
    frac: np.ndarray    # (C,) projection fraction along the arc [0, 1]
    rank: np.ndarray    # (C,) 0 = nearest arc for that point
    n_points: int

    def __len__(self) -> int:
        return len(self.point)

    def counts(self) -> np.ndarray:
        return np.bincount(self.point, minlength=self.n_points)

    def offsets(self) -> np.ndarray:
        """CSR-style offsets: candidates of point i are [off[i], off[i+1])."""
        return np.concatenate([[0], np.cumsum(self.counts())])

    def subset(self, radius: float | None = None, k: int | None = None) -> "Candidates":
        m = np.ones(len(self), bool)
        if radius is not None:
            m &= self.dist <= radius
        if k is not None:
            m &= self.rank < k
        return Candidates(self.point[m], self.arc[m], self.dist[m], self.frac[m], self.rank[m], self.n_points)


class ArcIndex:
    def __init__(self, net: RoadNetwork, piece_len: float = 20.0):
        self.net = net
        self.piece_len = piece_len
        n_pieces = np.maximum(1, np.ceil(net.length / piece_len).astype(int))
        arc_of_piece = np.repeat(np.arange(net.n_arcs), n_pieces)
        # midpoint fraction of piece j of n: (j + 0.5) / n
        start = np.concatenate([[0], np.cumsum(n_pieces)[:-1]])
        j = np.arange(len(arc_of_piece)) - np.repeat(start, n_pieces)
        f = (j + 0.5) / np.repeat(n_pieces, n_pieces)
        a, b = net.a_xy[arc_of_piece], net.b_xy[arc_of_piece]
        self.arc_of_piece = arc_of_piece
        self.tree = cKDTree(a + f[:, None] * (b - a))

    def query(self, pts: np.ndarray, radius: float, k: int | None = None) -> Candidates:
        hits = self.tree.query_ball_point(pts, r=radius + self.piece_len / 2, return_sorted=False)
        lens = np.fromiter((len(h) for h in hits), int, len(hits))
        if lens.sum() == 0:
            e = np.empty(0, int)
            return Candidates(e, e, np.empty(0), np.empty(0), e, len(pts))
        point = np.repeat(np.arange(len(pts)), lens)
        arc = self.arc_of_piece[np.concatenate([np.asarray(h, int) for h in hits if h])]

        # unique (point, arc)
        key = point.astype(np.int64) * self.net.n_arcs + arc
        key = np.unique(key)
        point, arc = key // self.net.n_arcs, key % self.net.n_arcs

        dist, frac = point_segment_distance_paired(pts[point], self.net.a_xy[arc], self.net.b_xy[arc])
        keep = dist <= radius
        point, arc, dist, frac = point[keep], arc[keep], dist[keep], frac[keep]

        order = np.lexsort((arc, dist, point))       # by point, then distance, then arc id (tie-break)
        point, arc, dist, frac = point[order], arc[order], dist[order], frac[order]
        start = np.searchsorted(point, np.arange(len(pts)))
        rank = np.arange(len(point)) - start[point]
        cands = Candidates(point, arc, dist, frac, rank, len(pts))
        return cands.subset(k=k) if k is not None else cands
