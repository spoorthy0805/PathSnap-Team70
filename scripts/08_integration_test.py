"""Phase 10 HTTP integration test for all trips served by the PathSnap API."""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
import pandas as pd
import requests

OUT = Path(__file__).resolve().parents[1] / "experiments" / "results" / "phase10"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://localhost:8000")
    args = parser.parse_args()
    base = args.api.rstrip("/")
    try:
        health_response = requests.get(f"{base}/health", timeout=10)
        health_response.raise_for_status()
        health = health_response.json()
        trips_response = requests.get(f"{base}/trips", timeout=10)
        trips_response.raise_for_status()
        trips = trips_response.json()["trips"]
    except (requests.RequestException, ValueError, KeyError) as exc:
        print(f"API preflight failed: {exc}", file=sys.stderr)
        sys.exit(2)
    print(f"API ok: {health['classifier']} + HMM, eval dir {health['eval_dir']}, {len(trips)} trips")
    if not trips:
        print("No trips available. Configure PATHSNAP_EVAL_DIR to a directory containing <trip_id>/<trip_id>.track.",
              file=sys.stderr)
        sys.exit(2)

    rows = []
    required = {"trip_id", "n_points", "points_matched", "route_arcs", "route_gaps",
                "gps", "route_segments", "metrics", "timings_ms"}
    for trip_id in trips:
        started = time.perf_counter()
        try:
            response = requests.post(f"{base}/predict", json={"trip_id": trip_id}, timeout=120)
        except requests.RequestException as exc:
            rows.append({"trip_id": trip_id, "status": "request_error", "error": str(exc)})
            print(f"{trip_id}: request error: {exc}")
            continue
        round_trip_ms = (time.perf_counter() - started) * 1000
        if response.status_code != 200:
            rows.append({"trip_id": trip_id, "status": response.status_code})
            print(f"{trip_id}: HTTP {response.status_code}")
            continue
        try:
            result = response.json()
            missing = required.difference(result) if isinstance(result, dict) else required
            assert not missing, f"response missing keys: {sorted(missing)}"
            assert "route" not in result and "reference" not in str(result.keys()), "ground truth must not be returned"
            assert isinstance(result["gps"], list) and isinstance(result["route_segments"], list)
            assert result["n_points"] > 0 and result["points_matched"] <= result["n_points"]
        except (ValueError, AssertionError, TypeError, KeyError) as exc:
            rows.append({"trip_id": trip_id, "status": "invalid_response", "error": str(exc)})
            print(f"{trip_id}: invalid response: {exc}")
            continue
        metrics = result["metrics"]
        rows.append({"trip_id": trip_id, "status": 200, "n_points": result["n_points"],
                     "points_matched": result["points_matched"],
                     **{key: value for key, value in metrics.items() if key != "available"},
                     "route_arcs": result["route_arcs"], "route_gaps": result["route_gaps"],
                     "server_ms": result["timings_ms"]["total"], "round_trip_ms": round(round_trip_ms, 1)})
        detail = (f"top1 {metrics['top1_accuracy']:.4f} route F1 {metrics['route_f1']:.4f}"
                  if metrics.get("available") else "prediction only (reference route unavailable)")
        print(f"{trip_id}: {detail} ({round_trip_ms:.0f} ms round trip)")

    try:
        unknown = requests.post(f"{base}/predict", json={"trip_id": "does-not-exist"}, timeout=10).status_code
    except requests.RequestException as exc:
        print(f"Unknown-trip check failed: {exc}", file=sys.stderr)
        unknown = 0
    print(f"Unknown trip -> HTTP {unknown} (expected 404)")
    frame = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "integration.csv", index=False)
    successful = frame[frame.status == 200]
    if successful.empty:
        print(f"0/{len(frame)} trips succeeded")
        sys.exit(1)
    if "top1_accuracy" in successful and successful.top1_accuracy.notna().any():
        pooled = (successful.top1_accuracy * successful.n_points).sum() / successful.n_points.sum()
        summary = f"pooled Top-1 {pooled:.4f}; mean route F1 {successful.route_f1.mean():.4f}"
    else:
        summary = "prediction contract passed; reference-route metrics unavailable"
    print(f"{len(successful)}/{len(frame)} trips succeeded; {summary}")
    sys.exit(0 if len(successful) == len(frame) and unknown == 404 else 1)


if __name__ == "__main__":
    main()
