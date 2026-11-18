"""Phase 7 — export the final inference artifacts and verify they reproduce the results.

Final pipeline (Phase 6 decision): untuned XGBoost (Phase 4 config) + HMM decoder.

Writes
  artifacts/xgb_final.json          XGBoost model (native JSON, version-portable)
  artifacts/pipeline_config.json    features, candidate settings, decoder params, provenance
  experiments/results/phase7/verification.csv   raw-file end-to-end run on student_test
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathsnap.config import ARTIFACTS_DIR, CAND_K, CAND_RADIUS_M, DATA_DIR, RESULTS_DIR, SEED  # noqa: E402
from pathsnap.experiment import EXCLUDED_TRIPS, load_tables  # noqa: E402
from pathsnap.features import FEATURES  # noqa: E402
from pathsnap.io import list_trips  # noqa: E402
from pathsnap.models import ZOO  # noqa: E402
from pathsnap.pipeline import CONFIG_FILE, MODEL_FILE, PathSnapModel, evaluate_prediction, predict_trip_dir  # noqa: E402


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "phase7"
    out.mkdir(parents=True, exist_ok=True)

    train, _ = load_tables()
    params = dict(ZOO["xgb"].config)
    t0 = time.time()
    clf = XGBClassifier(**params, tree_method="hist", eval_metric="logloss", random_state=SEED, n_jobs=-1)
    clf.fit(train[FEATURES].to_numpy(np.float32), train.y.to_numpy())
    fit_s = time.time() - t0
    clf.save_model(str(ARTIFACTS_DIR / MODEL_FILE))

    decoder = json.loads((RESULTS_DIR / "phase6" / "xgb_untuned" / "decoder_params.json").read_text())
    config = {
        "pipeline": "candidates -> XGBoost candidate scoring -> HMM/Viterbi",
        "features": FEATURES,
        "candidates": {"radius_m": CAND_RADIUS_M, "k": CAND_K, "piece_len_m": 20.0},
        "classifier": {"type": "XGBClassifier", "params": params, "tree_method": "hist", "seed": SEED,
                       "trained_on": f"{train.trip_id.nunique()} reliable train trips ({len(train):,} rows)",
                       "excluded_trips": EXCLUDED_TRIPS, "fit_seconds": round(fit_s, 1)},
        "decoder": {k: decoder[k] for k in ["alpha", "beta", "gap_s", "unreachable_penalty", "route_limit_m"]},
        "versions": {"xgboost": xgboost.__version__, "scikit-learn": sklearn.__version__},
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (ARTIFACTS_DIR / CONFIG_FILE).write_text(json.dumps(config, indent=2))
    print(f"saved {MODEL_FILE} ({(ARTIFACTS_DIR / MODEL_FILE).stat().st_size / 1e6:.1f} MB) and {CONFIG_FILE}")

    # ---- verify: raw files -> prediction -> server-side score, exactly as the API does ----
    model = PathSnapModel()
    split_dir = DATA_DIR / "student_test"
    rows = []
    for tid in list_trips("student_test"):
        pred, trip, net = predict_trip_dir(model, tid, split_dir)
        m = evaluate_prediction(pred, tid, split_dir, net)
        rows.append({"trip_id": tid, "n_points": pred.n_points, **{k: v for k, v in m.items() if k != "available"},
                     "route_gaps": pred.route_gaps, **{f"ms_{k}": round(v, 1) for k, v in pred.timings_ms.items()}})
        print(f"{tid}: top1 {m['top1_accuracy']:.4f}  route F1 {m['route_f1']:.4f}  {pred.timings_ms['total']:.0f} ms")
    df = pd.DataFrame(rows)
    df.to_csv(out / "verification.csv", index=False)
    pooled = (df.top1_accuracy * df.n_points).sum() / df.n_points.sum()
    print(f"pooled student_test Top-1 {pooled:.4f} (Phase 6 reported 0.9552); mean route F1 {df.route_f1.mean():.4f}")


if __name__ == "__main__":
    main()
