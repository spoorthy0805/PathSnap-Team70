"""Evaluation metrics shared by model experiments, HMM experiments and the API.

Conventions
- Candidate-level metrics (precision, recall, macro-F1, log loss) score the binary
  "is this candidate the reference arc?" decision; threshold 0.5 unless stated.
- Observation-level Top-1 accuracy: for every GPS point pick the highest-scoring
  candidate; the denominator is ALL GPS points, so points with no candidates or
  whose reference arc is missing from the candidate set count as errors.
- Route continuity: share of consecutive predicted points whose arcs are identical
  or directly connected (head of previous arc == tail of next arc).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, log_loss, precision_score, recall_score


def candidate_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    yhat = (p >= threshold).astype(int)
    return {
        "precision_macro": precision_score(y, yhat, average="macro", zero_division=0),
        "recall_macro": recall_score(y, yhat, average="macro", zero_division=0),
        "f1_macro": f1_score(y, yhat, average="macro", zero_division=0),
        "f1_pos": f1_score(y, yhat, zero_division=0),
        "log_loss": log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]),
    }


def _top1_positions(df: pd.DataFrame, score: np.ndarray) -> np.ndarray:
    """Positional indices (into df) of the highest-scoring candidate per (trip_id, point)."""
    d = pd.DataFrame({"trip_id": df.trip_id.to_numpy(), "point": df.point.to_numpy(), "score": score})
    d = d.sort_values(["trip_id", "point", "score"], ascending=[True, True, False], kind="stable")
    return d.drop_duplicates(["trip_id", "point"], keep="first").index.to_numpy()


def top1_choice(df: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    """Highest-scoring candidate per (trip_id, point), sorted by trip and point."""
    pos = _top1_positions(df, score)
    out = df.iloc[pos][["trip_id", "point", "arc", "y"]].assign(score=np.asarray(score)[pos])
    return out.sort_values(["trip_id", "point"], kind="stable").reset_index(drop=True)


def top1_accuracy(df: pd.DataFrame, score: np.ndarray, n_points_total: int) -> float:
    return float(df.y.to_numpy()[_top1_positions(df, score)].sum() / n_points_total)


def top1_decision_f1(df: pd.DataFrame, score: np.ndarray) -> float:
    """Macro-F1 when exactly the argmax candidate of each point is predicted positive."""
    yhat = np.zeros(len(df), dtype=int)
    yhat[_top1_positions(df, score)] = 1
    return f1_score(df.y, yhat, average="macro")


def route_continuity(points: np.ndarray, arcs: np.ndarray, arc_u: np.ndarray, arc_v: np.ndarray) -> float:
    """points/arcs: predicted arc per point, ordered by point index (one trip)."""
    if len(arcs) < 2:
        return float("nan")
    a, b = arcs[:-1], arcs[1:]
    ok = (a == b) | (arc_v[a] == arc_u[b])
    return float(ok.mean())


def continuity_by_trip(chosen: pd.DataFrame, arc_tables: dict) -> float:
    """Point-weighted continuity over trips. arc_tables[trip_id] = (u, v)."""
    num = den = 0
    for tid, g in chosen.groupby("trip_id", sort=False):
        u, v = arc_tables[tid]
        c = route_continuity(g.point.to_numpy(), g.arc.to_numpy(), u, v)
        if np.isfinite(c):
            num += c * (len(g) - 1)
            den += len(g) - 1
    return float(num / den) if den else float("nan")


def binary_confusion(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return confusion_matrix(y, (p >= threshold).astype(int), labels=[0, 1])
