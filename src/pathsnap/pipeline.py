"""End-to-end inference: raw trip files -> matched arc per GPS point -> connected route.

Prediction uses ONLY the trip's .track, .nodes and .arcs files. Reference routes are
read separately by `evaluate_prediction`, which the API calls server-side after the
prediction is complete, so ground truth never reaches the model or the UI.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .config import ARTIFACTS_DIR
from .features import build_features
from .hmm import RouteGraph, build_lattice, reconstruct_route, route_overlap, viterbi
from .io import Trip, load_trip
from .labels import align_track_to_route, track_xy
from .metrics import route_continuity
from .network import RoadNetwork

MODEL_FILE = "xgb_final.json"
CONFIG_FILE = "pipeline_config.json"


@dataclass
class Prediction:
    trip_id: str
    n_points: int
    point_arc: np.ndarray        # (n_points,) predicted arc per GPS point, -1 if no candidate
    route: list[int]             # connected arc sequence
    route_gaps: int              # disconnections that could not be bridged
    hmm_breaks: int
    timings_ms: dict = field(default_factory=dict)


class PathSnapModel:
    def __init__(self, artifacts_dir: Path = ARTIFACTS_DIR):
        artifacts_dir = Path(artifacts_dir)
        self.config = json.loads((artifacts_dir / CONFIG_FILE).read_text())
        self.clf = XGBClassifier()
        self.clf.load_model(str(artifacts_dir / MODEL_FILE))
        self.features = self.config["features"]

    def predict(self, trip: Trip) -> tuple[Prediction, RoadNetwork]:
        cfg = self.config
        t = {}
        t0 = time.perf_counter()
        net = RoadNetwork.from_trip(trip)
        feats, cands = build_features(trip, net, radius=cfg["candidates"]["radius_m"], k=cfg["candidates"]["k"])
        t["features"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        p = self.clf.predict_proba(feats[self.features].to_numpy(np.float32))[:, 1] if len(feats) else np.empty(0)
        t["classifier"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        frame = pd.DataFrame({"point": cands.point, "arc": cands.arc, "frac": cands.frac, "p": p})
        dec = cfg["decoder"]
        graph = RouteGraph(net, dec["route_limit_m"])
        lat = build_lattice(frame, track_xy(trip, net), trip.track.t.to_numpy(), graph)
        rows, diag = viterbi(lat, dec["alpha"], dec["beta"], dec["unreachable_penalty"], dec["gap_s"])
        point_arc = np.full(len(trip.track), -1, dtype=np.int64)
        point_arc[lat.points] = frame.arc.to_numpy()[rows]
        route, gaps = reconstruct_route(frame.arc.to_numpy()[rows], graph)
        t["hmm"] = (time.perf_counter() - t0) * 1000
        t["total"] = sum(t.values())
        return Prediction(trip.trip_id, len(trip.track), point_arc, route, gaps, diag["breaks"], t), net


def predict_trip_dir(model: PathSnapModel, trip_id: str, split_dir: Path) -> tuple[Prediction, Trip, RoadNetwork]:
    """Load a trip WITHOUT its reference route and predict it."""
    trip = load_trip(trip_id, split_dir.name, data_dir=split_dir.parent, with_route=False)
    pred, net = model.predict(trip)
    return pred, trip, net


def evaluate_prediction(pred: Prediction, trip_id: str, split_dir: Path, net: RoadNetwork) -> dict:
    """Server-side scoring against the reference route (never sent to the model or UI)."""
    trip = load_trip(trip_id, split_dir.name, data_dir=split_dir.parent, with_route=True)
    if trip.route is None:
        return {"available": False}
    lab = align_track_to_route(trip, net)
    matched = pred.point_arc >= 0
    pts_idx = np.flatnonzero(matched)
    return {
        "available": True,
        "top1_accuracy": float((pred.point_arc == lab.arc).mean()),
        "points_matched": int(matched.sum()),
        "route_continuity": route_continuity(pts_idx, pred.point_arc[matched], net.u, net.v),
        **route_overlap(pred.route, trip.route, net.length),
    }
