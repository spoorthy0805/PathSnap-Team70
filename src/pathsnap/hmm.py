"""HMM + Viterbi route reconstruction on top of classifier candidate scores.

Hidden states   at observation t: the candidate arcs of GPS point t (≤ K per point),
                each carrying its projection fraction along the arc.
Emissions       log e_t(c) = alpha * log max(p_c, eps), p_c = classifier P(candidate is the true arc).
Transitions     Newson & Krumm (2009): a vehicle's network distance between consecutive
                matched positions should be close to the straight-line GPS displacement,
                log a(c -> c') = -|d_route(c, c') - d_gc(t, t+1)| / beta
                d_route is the directed along-network distance from the projected position on
                arc c to the projected position on arc c' (bounded Dijkstra, ≤ ROUTE_LIMIT_M).
                Unreachable pairs (one-way violation, disconnected component, beyond the bound)
                get log a = -unreachable_penalty instead of -inf, so decoding never dead-ends.
Gaps            if the time gap between two observations exceeds gap_s, the transition is made
                uniform (log a = 0): the chain is effectively split and each side is decoded on
                its own evidence. Points without any candidate are skipped (no state).
Decoding        Viterbi: delta_t(c') = max_c [delta_{t-1}(c) + log a(c -> c')] + log e_t(c').
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .network import RoadNetwork

ROUTE_LIMIT_M = 1000.0
EPS = 1e-6


class RouteGraph:
    """Directed road graph with cached, distance-bounded Dijkstra from any node."""

    def __init__(self, net: RoadNetwork, limit_m: float = ROUTE_LIMIT_M):
        self.net = net
        self.limit = limit_m
        order = np.argsort(net.u, kind="stable")
        indptr = np.searchsorted(net.u[order], np.arange(net.n_nodes + 1))
        self._adj = [list(zip(net.v[order[indptr[n]:indptr[n + 1]]].tolist(),
                              net.length[order[indptr[n]:indptr[n + 1]]].tolist(),
                              order[indptr[n]:indptr[n + 1]].tolist()))
                     for n in range(net.n_nodes)]
        self._cache: dict[int, tuple[dict, dict]] = {}

    def from_node(self, src: int) -> tuple[dict, dict]:
        """(dist[node], pred_arc[node]) for all nodes within the bound."""
        hit = self._cache.get(src)
        if hit is not None:
            return hit
        dist, pred = {src: 0.0}, {}
        heap = [(0.0, src)]
        while heap:
            d, n = heapq.heappop(heap)
            if d > dist.get(n, np.inf):
                continue
            for m, w, a in self._adj[n]:
                nd = d + w
                if nd <= self.limit and nd < dist.get(m, np.inf):
                    dist[m] = nd
                    pred[m] = a
                    heapq.heappush(heap, (nd, m))
        self._cache[src] = (dist, pred)
        return dist, pred

    def path_arcs(self, src_node: int, dst_node: int) -> list[int] | None:
        dist, pred = self.from_node(src_node)
        if dst_node not in dist:
            return None
        arcs, n = [], dst_node
        while n != src_node:
            a = pred[n]
            arcs.append(a)
            n = int(self.net.u[a])
        return arcs[::-1]


@dataclass
class Lattice:
    """Decoder inputs for one trip, independent of decoder parameters (precompute once)."""
    points: np.ndarray          # (T,) GPS indices that have ≥1 candidate, ascending
    cand_arc: list[np.ndarray]  # per step: candidate arc ids
    cand_row: list[np.ndarray]  # per step: row indices into the candidate frame
    log_p: list[np.ndarray]     # per step: log classifier probability
    dev: list[np.ndarray]       # per transition t-1 -> t: |d_route - d_gc| (inf = unreachable)
    dt: np.ndarray              # (T,) time since previous step (dt[0] = 0)


def build_lattice(cands: pd.DataFrame, pts: np.ndarray, t: np.ndarray, graph: RouteGraph) -> Lattice:
    """cands: one trip, columns point, arc, frac, p (any order)."""
    net = graph.net
    c = cands.reset_index(drop=True)
    c = c.assign(_row=np.arange(len(c))).sort_values(["point", "p"], ascending=[True, False], kind="stable")
    points = c.point.unique()
    groups = [g for _, g in c.groupby("point", sort=True)]
    cand_arc = [g.arc.to_numpy() for g in groups]
    cand_row = [g._row.to_numpy() for g in groups]
    frac = [g.frac.to_numpy() for g in groups]
    log_p = [np.log(np.maximum(g.p.to_numpy(), EPS)) for g in groups]
    L = net.length
    dev, dts = [], [0.0]
    for k in range(1, len(points)):
        i, j = points[k - 1], points[k]
        dts.append(float(t[j] - t[i]))
        d_gc = float(np.hypot(*(pts[j] - pts[i])))
        a_prev, f_prev, a_cur, f_cur = cand_arc[k - 1], frac[k - 1], cand_arc[k], frac[k]
        m = np.full((len(a_prev), len(a_cur)), np.inf)
        for x, (ap, fp) in enumerate(zip(a_prev, f_prev)):
            dist, _ = graph.from_node(int(net.v[ap]))
            rem = (1.0 - fp) * L[ap]
            for y, (ac, fc) in enumerate(zip(a_cur, f_cur)):
                if ac == ap:
                    d_route = abs(fc - fp) * L[ap]
                else:
                    sp = dist.get(int(net.u[ac]))
                    if sp is None:
                        continue
                    d_route = rem + sp + fc * L[ac]
                m[x, y] = abs(d_route - d_gc)
        dev.append(m)
    return Lattice(points, cand_arc, cand_row, log_p, dev, np.array(dts))


def viterbi(lat: Lattice, alpha: float = 1.0, beta: float = 5.0, unreachable_penalty: float = 30.0,
            gap_s: float = 30.0) -> tuple[np.ndarray, dict]:
    """Return (chosen candidate row per lattice step, diagnostics)."""
    T = len(lat.points)
    if T == 0:
        return np.empty(0, int), {"breaks": 0, "unreachable_steps": 0}
    delta = alpha * lat.log_p[0]
    back = [None]
    breaks = unreach_steps = 0
    for k in range(1, T):
        if lat.dt[k] > gap_s:
            logA = np.zeros_like(lat.dev[k - 1])
            breaks += 1
        else:
            dv = lat.dev[k - 1]
            finite = np.isfinite(dv)
            logA = np.where(finite, -np.where(finite, dv, 0.0) / beta, -unreachable_penalty)
            if not finite.any():
                unreach_steps += 1
        s = delta[:, None] + logA
        bp = s.argmax(0)
        delta = s[bp, np.arange(s.shape[1])] + alpha * lat.log_p[k]
        delta -= delta.max()  # keep numbers small; argmax unaffected
        back.append(bp)
    idx = np.empty(T, int)
    idx[-1] = int(delta.argmax())
    for k in range(T - 1, 0, -1):
        idx[k - 1] = back[k][idx[k]]
    rows = np.array([lat.cand_row[k][idx[k]] for k in range(T)])
    return rows, {"breaks": breaks, "unreachable_steps": unreach_steps}


def classifier_only(lat: Lattice) -> np.ndarray:
    """Argmax-probability candidate per step (rows were sorted by p descending)."""
    return np.array([r[0] for r in lat.cand_row])


def reconstruct_route(point_arcs: np.ndarray, graph: RouteGraph) -> tuple[list[int], int]:
    """Collapse per-point arcs into a connected arc sequence, filling short gaps with shortest paths.

    Returns (route arcs, number of disconnections that could not be bridged).
    """
    net = graph.net
    seq = [int(a) for k, a in enumerate(point_arcs) if k == 0 or a != point_arcs[k - 1]]
    if not seq:
        return [], 0
    route, gaps = [seq[0]], 0
    for a in seq[1:]:
        prev = route[-1]
        if net.v[prev] != net.u[a]:
            fill = graph.path_arcs(int(net.v[prev]), int(net.u[a]))
            if fill is None:
                gaps += 1
            else:
                route.extend(fill)
        route.append(a)
    return route, gaps


def route_overlap(pred_route: list[int], ref_route: np.ndarray, length: np.ndarray) -> dict:
    """Length-weighted precision / recall / F1 between predicted and reference arc sets (direction-aware)."""
    P, R = set(pred_route), set(int(a) for a in ref_route)
    inter = sum(length[a] for a in P & R)
    lp, lr = sum(length[a] for a in P), sum(length[a] for a in R)
    prec = inter / lp if lp else 0.0
    rec = inter / lr if lr else 0.0
    return {"route_precision": prec, "route_recall": rec,
            "route_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}
