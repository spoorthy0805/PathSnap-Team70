"""Readers for the raw PathSnap trip files.

Each trip directory <split>/<id>/ holds four header-less, whitespace-separated files:
  <id>.track  lon lat t        (GPS observations, t in seconds)
  <id>.nodes  lon lat          (row index = node_id)
  <id>.arcs   from to          (row index = arc_id, directed)
  <id>.route  arc_id           (ordered reference route)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR


@dataclass
class Trip:
    trip_id: str
    split: str
    track: pd.DataFrame   # columns: lon, lat, t
    nodes: pd.DataFrame   # columns: lon, lat   (index = node_id)
    arcs: pd.DataFrame    # columns: u, v       (index = arc_id)
    route: np.ndarray | None  # arc ids, or None if not supplied

    @property
    def n_points(self) -> int:
        return len(self.track)


def _read(path: Path, names: list[str], dtype) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=dtype, engine="c")


def list_trips(split: str, data_dir: Path = DATA_DIR) -> list[str]:
    root = Path(data_dir) / split
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def trip_dir(trip_id: str, split: str, data_dir: Path = DATA_DIR) -> Path:
    return Path(data_dir) / split / trip_id


def load_trip(trip_id: str, split: str, data_dir: Path = DATA_DIR, with_route: bool = True) -> Trip:
    d = trip_dir(trip_id, split, data_dir)
    track = _read(d / f"{trip_id}.track", ["lon", "lat", "t"], np.float64)
    nodes = _read(d / f"{trip_id}.nodes", ["lon", "lat"], np.float64)
    arcs = _read(d / f"{trip_id}.arcs", ["u", "v"], np.int64)
    route = None
    route_path = d / f"{trip_id}.route"
    if with_route and route_path.exists():
        route = _read(route_path, ["arc"], np.int64)["arc"].to_numpy()
    return Trip(trip_id, split, track, nodes, arcs, route)


def iter_trips(split: str, data_dir: Path = DATA_DIR, with_route: bool = True):
    for tid in list_trips(split, data_dir):
        yield load_trip(tid, split, data_dir, with_route)
