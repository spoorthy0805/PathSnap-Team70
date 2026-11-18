"""Folium map builders shared by EDA, the report and the UI."""
from __future__ import annotations

import folium
import numpy as np
from scipy.spatial import cKDTree

from .io import Trip
from .network import RoadNetwork

COLORS = {"gps": "#1f6feb", "network": "#9aa0a6", "reference": "#1a7f37", "predicted": "#d1242f",
          "classifier": "#bf8700"}

_LEGEND = """
<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;padding:8px 12px;
border:1px solid #ccc;border-radius:6px;font:13px sans-serif;line-height:1.6">
<b>{title}</b><br>{rows}</div>"""


def _legend(title: str, layers: list[str]) -> folium.Element:
    rows = "".join(
        f'<span style="display:inline-block;width:14px;height:4px;background:{COLORS[k]};'
        f'margin-right:6px;vertical-align:middle"></span>{k.capitalize()}<br>' for k in layers)
    return folium.Element(_LEGEND.format(title=title, rows=rows))


def arcs_near_track(trip: Trip, net: RoadNetwork, radius_m: float = 150.0, max_arcs: int = 6000) -> np.ndarray:
    tx, ty = net.proj.to_xy(trip.track.lon.values, trip.track.lat.values)
    tree = cKDTree(np.column_stack([tx, ty]))
    mid = (net.a_xy + net.b_xy) / 2
    d, _ = tree.query(mid, k=1)
    ids = np.flatnonzero(d <= radius_m + net.length / 2)
    return ids[np.argsort(d[ids])][:max_arcs]


def _arc_lines(trip: Trip, arc_ids) -> list[list[tuple[float, float]]]:
    lon, lat = trip.nodes.lon.values, trip.nodes.lat.values
    u, v = trip.arcs.u.values[arc_ids], trip.arcs.v.values[arc_ids]
    return [[(lat[a], lon[a]), (lat[b], lon[b])] for a, b in zip(u, v)]


def _route_polyline(trip: Trip, arc_ids) -> list[tuple[float, float]]:
    """Chain consecutive arcs into one polyline (breaks are drawn as straight jumps)."""
    lon, lat = trip.nodes.lon.values, trip.nodes.lat.values
    u, v = trip.arcs.u.values, trip.arcs.v.values
    pts: list[tuple[float, float]] = []
    for a in arc_ids:
        s, e = (lat[u[a]], lon[u[a]]), (lat[v[a]], lon[v[a]])
        if not pts or pts[-1] != s:
            pts.append(s)
        pts.append(e)
    return pts


def trip_map(trip: Trip, net: RoadNetwork | None = None, predicted_arcs=None, show_reference: bool = True,
             show_network: bool = True, max_gps_points: int = 3000, title: str | None = None,
             classifier_arcs=None, predicted_label: str = "Predicted route") -> folium.Map:
    net = net or RoadNetwork.from_trip(trip)
    tr = trip.track
    m = folium.Map(tiles="OpenStreetMap", control_scale=True)
    m.fit_bounds([[tr.lat.min(), tr.lon.min()], [tr.lat.max(), tr.lon.max()]], padding=(20, 20))
    layers = []

    if show_network:
        fg = folium.FeatureGroup("Road network (near track)", show=True)
        for line in _arc_lines(trip, arcs_near_track(trip, net)):
            folium.PolyLine(line, color=COLORS["network"], weight=1.5, opacity=0.6).add_to(fg)
        fg.add_to(m)
        layers.append("network")

    if show_reference and trip.route is not None:
        fg = folium.FeatureGroup("Reference route", show=True)
        folium.PolyLine(_route_polyline(trip, trip.route), color=COLORS["reference"], weight=6, opacity=0.7,
                        tooltip="Reference route").add_to(fg)
        fg.add_to(m)
        layers.append("reference")

    if classifier_arcs is not None:
        fg = folium.FeatureGroup("Classifier-only route", show=False)
        folium.PolyLine(_route_polyline(trip, classifier_arcs), color=COLORS["classifier"], weight=3,
                        opacity=0.9, dash_array="6 4", tooltip="Classifier-only route").add_to(fg)
        fg.add_to(m)
        layers.append("classifier")

    if predicted_arcs is not None:
        fg = folium.FeatureGroup(predicted_label, show=True)
        folium.PolyLine(_route_polyline(trip, predicted_arcs), color=COLORS["predicted"], weight=3.5,
                        opacity=0.9, tooltip=predicted_label).add_to(fg)
        fg.add_to(m)
        layers.append("predicted")

    fg = folium.FeatureGroup("GPS observations", show=True)
    step = max(1, len(tr) // max_gps_points)
    for row in tr.iloc[::step].itertuples():
        folium.CircleMarker((row.lat, row.lon), radius=2, color=COLORS["gps"], fill=True, fill_opacity=0.8,
                            weight=0, tooltip=f"t={row.t:.0f}s").add_to(fg)
    folium.Marker((tr.lat.iloc[0], tr.lon.iloc[0]), tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(fg)
    folium.Marker((tr.lat.iloc[-1], tr.lon.iloc[-1]), tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(fg)
    fg.add_to(m)
    layers.append("gps")

    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(_legend(title or f"Trip {trip.trip_id} ({trip.split})", layers))
    return m
