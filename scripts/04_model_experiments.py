"""Phase 4 — train and compare candidate-scoring models.

Protocol
- Fit / CV data: the 78 reliable training trips (00000033 and 00000054 excluded, see Phase 2).
- 5-fold GroupKFold grouped by trip (shuffled, seed 42): no trip appears in both train and validation.
- Final model per family: refit on all 78 trips with the same config, scored once on the 10 student_test trips.
- Training metrics on the training folds are recorded to diagnose over/underfitting.

Usage
  python scripts/04_model_experiments.py                 # all models
  python scripts/04_model_experiments.py --models lgbm,xgb

Outputs (experiments/results/phase4/)
  cv_folds.csv            one row per (model, fold): train & validation metrics, timings
  comparison.csv          one row per model: CV mean±std, student_test metrics, timings
  histories/<model>.json  per-iteration train/valid log loss (iterative models only)
  confusion/<model>.json  student_test candidate-level confusion matrix (threshold 0.5)
  configs.json            exact model configurations + protocol
cache/preds/<model>_student_test.parquet   per-candidate probabilities (input to Phase 6)
cache/models/<model>.joblib                fitted final models
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathsnap.config import CACHE_DIR, RESULTS_DIR, SEED  # noqa: E402
from pathsnap.experiment import EXCLUDED_TRIPS, N_FOLDS, arc_tables, cv_splits, evaluate, load_tables, n_points  # noqa: E402
from pathsnap.features import FEATURES  # noqa: E402
from pathsnap.metrics import binary_confusion  # noqa: E402
from pathsnap.models import ZOO, fit_model  # noqa: E402

OUT = RESULTS_DIR / "phase4"
TRAIN_DIAG_TRIPS = 12  # training-fold trips used to measure training fit


def run_model(name: str, train: pd.DataFrame, test: pd.DataFrame, arcs_tr: dict, arcs_te: dict):
    zm = ZOO[name]
    rng = np.random.default_rng(SEED)
    X, y, groups = train[FEATURES].to_numpy(np.float32), train.y.to_numpy(), train.trip_id.to_numpy()
    fold_rows, histories = [], {}

    for k, (tr_idx, va_idx) in enumerate(cv_splits(train)):
        fit_idx = tr_idx
        if zm.max_train_rows and len(tr_idx) > zm.max_train_rows:
            fit_idx = np.sort(rng.choice(tr_idx, zm.max_train_rows, replace=False))
        t0 = time.perf_counter()
        est, hist = fit_model(zm, X[fit_idx], y[fit_idx], X[va_idx], y[va_idx])
        t_fit = time.perf_counter() - t0
        t0 = time.perf_counter()
        p_va = est.predict_proba(X[va_idx])[:, 1]
        t_pred = time.perf_counter() - t0
        va = train.iloc[va_idx]
        mv = evaluate(va, p_va, arcs_tr)

        diag_trips = rng.choice(np.unique(groups[tr_idx]), min(TRAIN_DIAG_TRIPS, len(np.unique(groups[tr_idx]))),
                                replace=False)
        dmask = np.isin(groups, diag_trips)
        dtr = train[dmask]
        mt = evaluate(dtr, est.predict_proba(X[dmask])[:, 1], None)

        fold_rows.append({"model": name, "fold": k, "n_fit_rows": len(fit_idx), "n_val_trips": va.trip_id.nunique(),
                          "fit_s": t_fit, "predict_s": t_pred,
                          **{f"val_{a}": b for a, b in mv.items()},
                          **{f"train_{a}": b for a, b in mt.items() if a in ("top1", "log_loss", "f1_macro")}})
        if hist is not None:
            histories[f"fold{k}"] = hist
        print(f"  [{name}] fold {k}: top1 {mv['top1']:.4f}  f1 {mv['f1_macro']:.4f}  logloss {mv['log_loss']:.4f}"
              f"  (train top1 {mt['top1']:.4f})  fit {t_fit:.1f}s", flush=True)

    # ---- final fit on all reliable training trips, score student_test ----
    fit_idx = np.arange(len(X))
    if zm.max_train_rows and len(fit_idx) > zm.max_train_rows:
        fit_idx = np.sort(rng.choice(fit_idx, zm.max_train_rows, replace=False))
    t0 = time.perf_counter()
    est, _ = fit_model(zm, X[fit_idx], y[fit_idx])
    t_fit = time.perf_counter() - t0
    Xte = test[FEATURES].to_numpy(np.float32)
    t0 = time.perf_counter()
    p_te = est.predict_proba(Xte)[:, 1]
    t_pred = time.perf_counter() - t0
    mte = evaluate(test, p_te, arcs_te)
    cm = binary_confusion(test.y.to_numpy(), p_te)

    (CACHE_DIR / "preds").mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / "models").mkdir(parents=True, exist_ok=True)
    test[["trip_id", "point", "arc", "y"]].assign(p=p_te).to_parquet(CACHE_DIR / "preds" / f"{name}_student_test.parquet")
    joblib.dump(est, CACHE_DIR / "models" / f"{name}.joblib")
    (OUT / "confusion").mkdir(parents=True, exist_ok=True)
    (OUT / "confusion" / f"{name}.json").write_text(json.dumps({"labels": [0, 1], "matrix": cm.tolist()}))
    if histories:
        (OUT / "histories").mkdir(parents=True, exist_ok=True)
        (OUT / "histories" / f"{name}.json").write_text(json.dumps(histories))

    folds = pd.DataFrame(fold_rows)
    summary = {"model": name, "family": zm.family, "config": json.dumps(zm.config, default=str), "notes": zm.notes}
    for col in ["val_top1", "val_f1_macro", "val_log_loss", "val_f1_macro_top1", "val_continuity",
                "train_top1", "train_log_loss"]:
        summary[f"cv_{col[4:] if col.startswith('val_') else col}_mean"] = folds[col].mean()
        summary[f"cv_{col[4:] if col.startswith('val_') else col}_std"] = folds[col].std()
    summary.update({f"test_{a}": b for a, b in mte.items()})
    summary.update({"final_fit_s": t_fit, "test_predict_s": t_pred,
                    "test_predict_ms_per_1k_points": t_pred * 1e6 / n_points(test),
                    "cv_fit_s_mean": folds.fit_s.mean()})
    print(f"  [{name}] student_test: top1 {mte['top1']:.4f}  f1 {mte['f1_macro']:.4f}  logloss {mte['log_loss']:.4f}"
          f"  continuity {mte['continuity']:.4f}", flush=True)
    return folds, summary


def upsert(path: Path, new: pd.DataFrame, models: list[str]):
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[~old.model.isin(models)], new], ignore_index=True)
    order = {m: i for i, m in enumerate(ZOO)}
    new = new.sort_values(by="model", key=lambda s: s.map(order), kind="stable")
    new.to_csv(path, index=False)
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(ZOO))
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    OUT.mkdir(parents=True, exist_ok=True)

    train, test = load_tables()
    print(f"train rows {len(train):,} ({train.trip_id.nunique()} trips) | test rows {len(test):,}", flush=True)
    arcs_tr, arcs_te = arc_tables(train, "train"), arc_tables(test, "student_test")

    configs = {"protocol": {"cv": f"GroupKFold(n_splits={N_FOLDS}, shuffle=True, random_state={SEED}) by trip",
                            "train_trips": int(train.trip_id.nunique()), "excluded_trips": EXCLUDED_TRIPS,
                            "features": FEATURES, "threshold": 0.5, "seed": SEED},
               "models": {m: {"family": z.family, "config": z.config, "max_train_rows": z.max_train_rows,
                              "notes": z.notes} for m, z in ZOO.items()}}
    (OUT / "configs.json").write_text(json.dumps(configs, indent=2, default=str))

    all_folds, summaries = [], []
    for m in models:
        print(f"== {m} ({ZOO[m].family})", flush=True)
        folds, summ = run_model(m, train, test, arcs_tr, arcs_te)
        all_folds.append(folds)
        summaries.append(summ)
        upsert(OUT / "cv_folds.csv", folds, [m])
        upsert(OUT / "comparison.csv", pd.DataFrame([summ]), [m])

    comp = pd.read_csv(OUT / "comparison.csv")
    cols = ["model", "family", "cv_f1_macro_mean", "cv_top1_mean", "cv_log_loss_mean", "test_top1", "test_f1_macro",
            "test_log_loss", "test_continuity", "cv_fit_s_mean", "test_predict_ms_per_1k_points"]
    print(comp[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
