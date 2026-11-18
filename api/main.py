"""PathSnap FastAPI backend (Laptop 1).

Endpoints:
  GET /health, GET /trips, POST /predict {"trip_id"}
Ground truth is read only on the API server; the reference route is never returned.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from pathsnap.pipeline import PathSnapModel, evaluate_prediction  # noqa: E402
from pathsnap.io import load_trip  # noqa: E402

EVAL_DIR = Path(os.environ.get("PATHSNAP_EVAL_DIR", REPO / "data" / "eval_test")).resolve()
MAX_GPS_POINTS = 2500
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("pathsnap")
app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
MODEL = PathSnapModel()
log.info("model loaded: %s | eval dir: %s", MODEL.config["classifier"]["type"], EVAL_DIR)


class PredictRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    trip_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")


def available_trips() -> list[str]:
    if not EVAL_DIR.is_dir():
        return []
    return sorted(p.name for p in EVAL_DIR.iterdir() if p.is_dir() and (p / f"{p.name}.track").exists())


@app.middleware("http")
async def log_requests(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    client = request.client.host if request.client else "?"
    log.info("%s %s from %s -> %d (%.0f ms)", request.method, request.url.path, client,
             response.status_code, (time.perf_counter() - started) * 1000)
    return response


@app.get("/health")
def health():
    return {"status": "ok", "eval_dir": EVAL_DIR.name, "eval_dir_exists": EVAL_DIR.is_dir(),
            "n_trips": len(available_trips()), "classifier": MODEL.config["classifier"]["type"],
            "decoder": MODEL.config["decoder"]}


@app.get("/trips")
def trips():
    return {"trips": available_trips()}


@app.post("/predict")
def predict(req: PredictRequest):
    if req.trip_id not in available_trips():
        raise HTTPException(status_code=404, detail=f"unknown trip_id {req.trip_id!r}")
    log.info("predict trip %s: running pipeline", req.trip_id)
    try:
        trip = load_trip(req.trip_id, EVAL_DIR.name, data_dir=EVAL_DIR.parent, with_route=False)
        pred, net = MODEL.predict(trip)
        metrics = evaluate_prediction(pred, req.trip_id, EVAL_DIR, net)
    except FileNotFoundError as exc:
        log.exception("required input missing for trip %s", req.trip_id)
        raise HTTPException(status_code=500, detail="Trip data is incomplete on the API server") from exc
    except Exception as exc:
        log.exception("prediction failed for trip %s", req.trip_id)
        raise HTTPException(status_code=500, detail="Prediction failed; inspect the API server log") from exc

    log.info("predict trip %s: %d points, %d route arcs, %.0f ms", req.trip_id, pred.n_points,
             len(pred.route), pred.timings_ms["total"])
    lon, lat = trip.nodes.lon.to_numpy(), trip.nodes.lat.to_numpy()
    segments, current = [], []
    for arc in pred.route:
        start, end = int(net.u[arc]), int(net.v[arc])
        if current and current[-1] != [lat[start], lon[start]]:
            segments.append(current)
            current = []
        if not current:
            current.append([lat[start], lon[start]])
        current.append([lat[end], lon[end]])
    if current:
        segments.append(current)
    step = max(1, len(trip.track) // MAX_GPS_POINTS)
    gps = trip.track.iloc[::step][["lat", "lon"]].round(7).to_numpy().tolist()
    return {"trip_id": req.trip_id, "n_points": pred.n_points,
            "points_matched": int((pred.point_arc >= 0).sum()), "route_arcs": len(pred.route),
            "route_gaps": pred.route_gaps, "gps": gps,
            "route_segments": [[[round(x, 7) for x in point] for point in segment] for segment in segments],
            "metrics": metrics, "timings_ms": {key: round(value, 1) for key, value in pred.timings_ms.items()}}
