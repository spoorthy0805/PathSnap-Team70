"""Geodesy helpers.

Trips are worldwide, so each trip gets its own azimuthal-equidistant projection
centred on the track. Within a few tens of km this keeps distance error far
below GPS noise, and lets us use fast planar geometry in metres.
"""
from __future__ import annotations

import numpy as np
from pyproj import Transformer

EARTH_R = 6_371_008.8  # mean Earth radius (m)


def haversine(lon1, lat1, lon2, lat2):
    """Great-circle distance in metres (vectorised)."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


class LocalProjection:
    """lon/lat <-> local metric x/y (metres) centred at (lon0, lat0)."""

    def __init__(self, lon0: float, lat0: float):
        self.lon0, self.lat0 = float(lon0), float(lat0)
        proj = f"+proj=aeqd +lat_0={self.lat0} +lon_0={self.lon0} +datum=WGS84 +units=m +no_defs"
        self._fwd = Transformer.from_crs("EPSG:4326", proj, always_xy=True)
        self._inv = Transformer.from_crs(proj, "EPSG:4326", always_xy=True)

    @classmethod
    def for_track(cls, lon: np.ndarray, lat: np.ndarray) -> "LocalProjection":
        return cls(np.median(lon), np.median(lat))

    def to_xy(self, lon, lat):
        x, y = self._fwd.transform(np.asarray(lon), np.asarray(lat))
        return np.asarray(x), np.asarray(y)

    def to_lonlat(self, x, y):
        lon, lat = self._inv.transform(np.asarray(x), np.asarray(y))
        return np.asarray(lon), np.asarray(lat)


def bearing_xy(dx, dy):
    """Bearing in degrees clockwise from north, in [0, 360)."""
    return np.degrees(np.arctan2(dx, dy)) % 360.0


def angle_diff(a, b):
    """Smallest absolute difference between two bearings, in [0, 180]."""
    d = np.abs((np.asarray(a) - np.asarray(b)) % 360.0)
    return np.minimum(d, 360.0 - d)
