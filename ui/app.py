"""PathSnap evaluator UI. The UI reads predictions only from the API."""
from __future__ import annotations
import os
import folium
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

DEFAULT_API = os.environ.get("PATHSNAP_API", "http://localhost:8000")
TIMEOUT_S = 120
GPS_COLOR, ROUTE_COLOR = "#2563eb", "#dc2626"
st.set_page_config(page_title="PathSnap", page_icon="🗺️", layout="wide")


def api_get(base: str, path: str) -> dict:
    response = requests.get(f"{base.rstrip('/')}{path}", timeout=10)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"API {path} returned an unexpected JSON shape")
    return payload


def api_predict(base: str, trip_id: str) -> dict:
    response = requests.post(f"{base.rstrip('/')}/predict", json={"trip_id": trip_id}, timeout=TIMEOUT_S)
    response.raise_for_status()
    payload = response.json()
    required = {"trip_id", "n_points", "points_matched", "route_arcs", "route_gaps",
                "gps", "route_segments", "metrics", "timings_ms"}
    missing = required.difference(payload) if isinstance(payload, dict) else required
    if missing:
        raise ValueError(f"API returned an invalid prediction response; missing: {sorted(missing)}")
    return payload


def build_map(result: dict) -> folium.Map:
    gps, segments = result["gps"], result["route_segments"]
    points = gps + [point for segment in segments for point in segment]
    if not points:
        raise ValueError("Prediction response contains no GPS or route coordinates")
    map_view = folium.Map(location=points[0], zoom_start=13, tiles="OpenStreetMap", control_scale=True)
    map_view.fit_bounds([[min(p[0] for p in points), min(p[1] for p in points)],
                         [max(p[0] for p in points), max(p[1] for p in points)]], padding=(20, 20))
    gps_layer = folium.FeatureGroup(name=f"GPS observations ({result['n_points']} pts)")
    for lat, lon in gps:
        folium.CircleMarker([lat, lon], radius=2, color=GPS_COLOR, fill=True, fill_opacity=0.7,
                            weight=0).add_to(gps_layer)
    route_layer = folium.FeatureGroup(name=f"Predicted route ({result['route_arcs']} arcs)")
    for segment in segments:
        folium.PolyLine(segment, color=ROUTE_COLOR, weight=5, opacity=0.8).add_to(route_layer)
    if gps:
        folium.Marker(gps[0], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(route_layer)
        folium.Marker(gps[-1], tooltip="End", icon=folium.Icon(color="black", icon="stop")).add_to(route_layer)
    gps_layer.add_to(map_view)
    route_layer.add_to(map_view)
    folium.LayerControl(collapsed=False).add_to(map_view)
    return map_view


def pct(value) -> str:
    return "—" if value is None else f"{100 * value:.2f}%"


st.sidebar.title("PathSnap")
api = st.sidebar.text_input("API URL (Laptop 1)", DEFAULT_API)
try:
    health = api_get(api, "/health")
    trips = api_get(api, "/trips")["trips"]
    st.sidebar.success(f"Connected · {health['classifier']} + HMM · {health['n_trips']} trips")
    with st.sidebar.expander("Decoder settings"):
        st.json(health["decoder"])
except (requests.RequestException, ValueError, KeyError) as exc:
    st.sidebar.error(f"Cannot use API: {exc}")
    st.title("PathSnap — map matching")
    st.info("Start the API on Laptop 1 and set its URL in the sidebar.")
    st.stop()

st.title("PathSnap — map matching & route reconstruction")
st.caption("The API runs candidate generation, XGBoost and HMM/Viterbi and scores predictions server-side. "
           "The reference route is never sent to the browser.")
if not trips:
    st.warning("No evaluation trips are available. Configure PATHSNAP_EVAL_DIR on the API server.")
    st.stop()

single_tab, all_tab = st.tabs(["Single trip", "All trips"])
with single_tab:
    left, right = st.columns([3, 1])
    trip_id = left.selectbox("Evaluation trip", trips, index=0)
    run = right.button("Match trip", type="primary", use_container_width=True)
    if run:
        with st.spinner(f"Running pipeline on trip {trip_id}…"):
            try:
                st.session_state["result"] = api_predict(api, trip_id)
            except (requests.RequestException, ValueError, KeyError) as exc:
                st.error(f"Prediction failed: {exc}")
    result = st.session_state.get("result")
    if result:
        metrics = result["metrics"]
        st.subheader(f"Trip {result['trip_id']}")
        columns = st.columns(6)
        columns[0].metric("Top-1 accuracy", pct(metrics.get("top1_accuracy")))
        columns[1].metric("Route F1", pct(metrics.get("route_f1")))
        columns[2].metric("Route precision", pct(metrics.get("route_precision")))
        columns[3].metric("Route recall", pct(metrics.get("route_recall")))
        columns[4].metric("Route continuity", pct(metrics.get("route_continuity")))
        columns[5].metric("Total time", f"{result['timings_ms']['total']:.0f} ms")
        st.caption(f"{result['points_matched']}/{result['n_points']} GPS points matched · "
                   f"{result['route_arcs']} route arcs · {result['route_gaps']} unbridged gaps")
        if not metrics.get("available", True):
            st.info("No reference route is available on the server; only the prediction is shown.")
        st_folium(build_map(result), height=600, use_container_width=True, returned_objects=[],
                  key=f"map_{result['trip_id']}")

with all_tab:
    st.write("Runs every trip through the API one by one.")
    if st.button("Evaluate all trips"):
        rows = []
        progress = st.progress(0.0)
        for index, trip in enumerate(trips):
            try:
                result = api_predict(api, trip)
                rows.append({"trip_id": trip, "n_points": result["n_points"],
                             **{key: result["metrics"].get(key) for key in
                                ["top1_accuracy", "route_precision", "route_recall", "route_f1", "route_continuity"]},
                             "route_gaps": result["route_gaps"], "ms": result["timings_ms"]["total"]})
            except (requests.RequestException, ValueError, KeyError) as exc:
                rows.append({"trip_id": trip, "error": str(exc)})
            progress.progress((index + 1) / len(trips))
        st.session_state["all"] = pd.DataFrame(rows)
    frame = st.session_state.get("all")
    if frame is not None:
        if "top1_accuracy" in frame:
            valid = frame.dropna(subset=["top1_accuracy"])
            if not valid.empty:
                pooled = (valid.top1_accuracy * valid.n_points).sum() / valid.n_points.sum()
                first, second, third = st.columns(3)
                first.metric("Pooled Top-1", pct(pooled))
                second.metric("Mean route F1", pct(valid.route_f1.mean()))
                third.metric("Trips succeeded", f"{len(valid)}/{len(frame)}")
        st.dataframe(frame, use_container_width=True, hide_index=True)
