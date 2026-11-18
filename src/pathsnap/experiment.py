"""Shared experiment plumbing: data tables, CV folds and evaluation (Phases 4–6)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .config import CACHE_DIR, DATA_DIR, SEED
from .metrics import candidate_metrics, continuity_by_trip, top1_accuracy, top1_choice, top1_decision_f1

N_FOLDS = 5
EXCLUDED_TRIPS = ["00000033", "00000054"]  # unreliable reference routes (Phase 2 rule)


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reliable-trip training table and student_test table (Phase 3 cache)."""
    feat = CACHE_DIR / "features"
    train = pd.read_parquet(feat / "train.parquet")
    test = pd.read_parquet(feat / "student_test.parquet")
    return train[~train.unreliable].reset_index(drop=True), test


def cv_splits(train: pd.DataFrame):
    """Deterministic trip-grouped folds; identical across phases for fair comparison."""
    gkf = GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    return list(gkf.split(np.zeros(len(train)), train.y.to_numpy(), train.trip_id.to_numpy()))


def arc_tables(df: pd.DataFrame, split: str) -> dict:
    out = {}
    for tid in df.trip_id.unique():
        a = pd.read_csv(DATA_DIR / split / tid / f"{tid}.arcs", sep=r"\s+", header=None).to_numpy()
        out[tid] = (a[:, 0], a[:, 1])
    return out


def n_points(df: pd.DataFrame) -> int:
    return int(df.drop_duplicates("trip_id").n_points_trip.sum())


def evaluate(df: pd.DataFrame, p: np.ndarray, arcs: dict | None) -> dict:
    m = candidate_metrics(df.y.to_numpy(), p)
    m["top1"] = top1_accuracy(df, p, n_points(df))
    m["f1_macro_top1"] = top1_decision_f1(df, p)
    if arcs is not None:
        m["continuity"] = continuity_by_trip(top1_choice(df, p), arcs)
    return m
