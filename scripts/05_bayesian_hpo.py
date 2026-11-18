"""Phase 5 — Bayesian hyperparameter optimisation of the selected model (XGBoost).

Search
- Optuna TPE sampler (Bayesian, seed 42) with a median pruner.
- Objective: MAXIMISE mean validation observation Top-1 over the same 5 trip-grouped
  folds used in Phase 4 (78 reliable training trips). Log loss and macro-F1 are logged
  per trial. student_test is never touched during the search.
- After the search: refit the best configuration on all 78 trips, evaluate once on
  student_test, and compare with the untuned Phase 4 XGBoost.

Usage
  python scripts/05_bayesian_hpo.py --trials 40          # run / resume the search, then evaluate
  python scripts/05_bayesian_hpo.py --trials 0           # only evaluate the best trial so far

Outputs (experiments/results/phase5/)
  optuna_xgb.db                  Optuna study (full search record, resumable)
  trials.csv                     every trial: params, CV Top-1/F1/log loss, duration, state
  search_space.json              search space, objective, sampler, pruner
  best_params.json               best configuration
  tuned_vs_untuned.csv           same metrics + partitions for both models
reports/figures/hpo_*.png
cache/models/xgb_tuned.joblib, cache/preds/xgb_tuned_student_test.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathsnap.config import CACHE_DIR, FIGURES_DIR, RESULTS_DIR, SEED  # noqa: E402
from pathsnap.experiment import arc_tables, cv_splits, evaluate, load_tables, n_points  # noqa: E402
from pathsnap.features import FEATURES  # noqa: E402
from pathsnap.metrics import candidate_metrics, top1_accuracy  # noqa: E402
from pathsnap.models import ZOO  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

OUT = RESULTS_DIR / "phase5"
STUDY = "xgb_top1"

SPACE = {
    "n_estimators": ("int", 200, 1200, 100),
    "learning_rate": ("float_log", 0.01, 0.3),
    "max_depth": ("int", 3, 10, 1),
    "min_child_weight": ("float_log", 1.0, 50.0),
    "subsample": ("float", 0.5, 1.0),
    "colsample_bytree": ("float", 0.4, 1.0),
    "gamma": ("float", 0.0, 5.0),
    "reg_lambda": ("float_log", 1e-3, 10.0),
    "reg_alpha": ("float_log", 1e-4, 1.0),
}


def suggest(trial: optuna.Trial) -> dict:
    p = {}
    for name, spec in SPACE.items():
        kind = spec[0]
        if kind == "int":
            p[name] = trial.suggest_int(name, spec[1], spec[2], step=spec[3])
        elif kind == "float":
            p[name] = trial.suggest_float(name, spec[1], spec[2])
        else:
            p[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
    return p


def make_xgb(params: dict) -> XGBClassifier:
    return XGBClassifier(**params, tree_method="hist", eval_metric="logloss", random_state=SEED, n_jobs=-1)


def run_search(n_trials: int, train: pd.DataFrame, folds) -> optuna.Study:
    X, y = train[FEATURES].to_numpy(np.float32), train.y.to_numpy()
    val_frames = [train.iloc[va] for _, va in folds]
    val_points = [n_points(f) for f in val_frames]

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial)
        top1s, lls, f1s = [], [], []
        for k, (tr, va) in enumerate(folds):
            est = make_xgb(params).fit(X[tr], y[tr])
            p = est.predict_proba(X[va])[:, 1]
            top1s.append(top1_accuracy(val_frames[k], p, val_points[k]))
            cm = candidate_metrics(y[va], p)
            lls.append(cm["log_loss"]); f1s.append(cm["f1_macro"])
            trial.report(float(np.mean(top1s)), step=k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("cv_log_loss", float(np.mean(lls)))
        trial.set_user_attr("cv_f1_macro", float(np.mean(f1s)))
        trial.set_user_attr("cv_top1_std", float(np.std(top1s, ddof=1)))
        trial.set_user_attr("fold_top1", [float(v) for v in top1s])
        return float(np.mean(top1s))

    storage = f"sqlite:///{OUT / 'optuna_xgb.db'}"
    study = optuna.create_study(study_name=STUDY, storage=storage, direction="maximize", load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=10),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
    if not any(t.user_attrs.get("baseline") for t in study.trials):
        # Trial 0 = untuned Phase 4 configuration, so the search starts from the known baseline.
        base = {k: v for k, v in ZOO["xgb"].config.items()}
        base.update({"min_child_weight": 1.0, "gamma": 0.0, "reg_lambda": 1.0, "reg_alpha": 1e-4})
        study.enqueue_trial(base, user_attrs={"baseline": True})
    done = len([t for t in study.trials if t.state.is_finished()])
    remaining = max(0, n_trials - done)
    if remaining:
        study.optimize(objective, n_trials=remaining, show_progress_bar=False,
                       callbacks=[lambda s, t: print(f"trial {t.number:3d} {t.state.name:8s} "
                                                     f"value {t.value if t.value is not None else float('nan'):.4f} "
                                                     f"best {s.best_value:.4f}", flush=True)])
    return study


def cv_eval(params: dict, train: pd.DataFrame, folds, arcs) -> dict:
    X, y = train[FEATURES].to_numpy(np.float32), train.y.to_numpy()
    rows = []
    for tr, va in folds:
        p = make_xgb(params).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
        rows.append(evaluate(train.iloc[va], p, arcs))
    d = pd.DataFrame(rows)
    return {f"cv_{c}_mean": d[c].mean() for c in d} | {f"cv_{c}_std": d[c].std() for c in d}


def final_eval(name: str, params: dict, train, test, arcs_te) -> dict:
    X, y = train[FEATURES].to_numpy(np.float32), train.y.to_numpy()
    t0 = time.perf_counter()
    est = make_xgb(params).fit(X, y)
    fit_s = time.perf_counter() - t0
    Xte = test[FEATURES].to_numpy(np.float32)
    est.predict_proba(Xte[:1000])  # warm-up
    t0 = time.perf_counter()
    p = est.predict_proba(Xte)[:, 1]
    pred_s = time.perf_counter() - t0
    (CACHE_DIR / "models").mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / "preds").mkdir(parents=True, exist_ok=True)
    joblib.dump(est, CACHE_DIR / "models" / f"{name}.joblib")
    test[["trip_id", "point", "arc", "y"]].assign(p=p).to_parquet(CACHE_DIR / "preds" / f"{name}_student_test.parquet")
    m = evaluate(test, p, arcs_te)
    return {f"test_{k}": v for k, v in m.items()} | {"final_fit_s": fit_s,
                                                      "test_predict_ms_per_1k_points": pred_s * 1e6 / n_points(test)}


def figures(study: optuna.Study, comp: pd.DataFrame):
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    num = [t.number for t in trials]
    val = [t.value for t in trials]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.scatter(num, val, color="#1f6feb", label="completed trial")
    pruned = [t.number for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    if pruned:
        ax.scatter(pruned, [min(val)] * len(pruned), marker="x", color="#9aa0a6", label="pruned trial (at y-min)")
    ax.plot(num, np.maximum.accumulate(val), color="#d1242f", label="best so far")
    ax.set(title="Bayesian optimisation history (TPE): CV Top-1", xlabel="trial", ylabel="mean CV Top-1")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "hpo_history.png", dpi=130); plt.close(fig)

    try:
        imp = optuna.importance.get_param_importances(study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=SEED))
        fig, ax = plt.subplots(figsize=(7, 4))
        ks = list(imp)[::-1]
        ax.barh(ks, [imp[k] for k in ks], color="#1f6feb")
        ax.set(title="Hyperparameter importance (fANOVA on CV Top-1)", xlabel="importance")
        ax.grid(alpha=0.3, axis="x"); fig.tight_layout(); fig.savefig(FIGURES_DIR / "hpo_param_importance.png", dpi=130)
        plt.close(fig)
        (OUT / "param_importance.json").write_text(json.dumps(imp, indent=2))
    except Exception as e:  # importance needs enough completed trials
        print("param importance skipped:", e)

    metrics = [("cv_f1_macro_mean", "CV macro-F1"), ("cv_top1_mean", "CV Top-1"), ("test_top1", "Test Top-1"),
               ("test_f1_macro", "Test macro-F1")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [3, 1, 1]})
    x = np.arange(len(metrics)); w = 0.38
    for i, (_, r) in enumerate(comp.iterrows()):
        axes[0].bar(x + (i - 0.5) * w, [r[m] for m, _ in metrics], w, label=r.model,
                    color=["#9aa0a6", "#1f6feb"][i])
        for j, (m, _) in enumerate(metrics):
            axes[0].text(j + (i - 0.5) * w, r[m] + 0.002, f"{r[m]:.4f}", ha="center", fontsize=8)
    axes[0].set_xticks(x, [lbl for _, lbl in metrics]); axes[0].set_ylim(0.85, 0.96); axes[0].legend()
    axes[0].set_title("Untuned vs tuned XGBoost"); axes[0].grid(alpha=0.3, axis="y")
    axes[1].bar(comp.model, comp.test_log_loss, color=["#9aa0a6", "#1f6feb"]); axes[1].set_title("Test log loss")
    axes[2].bar(comp.model, comp.test_predict_ms_per_1k_points, color=["#9aa0a6", "#1f6feb"])
    axes[2].set_title("Inference ms / 1k GPS points")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "hpo_tuned_vs_untuned.png", dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    train, test = load_tables()
    folds = cv_splits(train)
    (OUT / "search_space.json").write_text(json.dumps({
        "model": "XGBClassifier(tree_method='hist')", "space": SPACE,
        "objective": "maximise mean observation Top-1 over 5 GroupKFold(by trip, shuffle, seed 42) folds, 78 trips",
        "sampler": "TPESampler(seed=42, n_startup_trials=10)", "pruner": "MedianPruner(n_startup_trials=8, n_warmup_steps=1)",
        "trial_0": "untuned Phase 4 configuration (enqueued)", "student_test_used_in_search": False}, indent=2))

    t0 = time.time()
    study = run_search(args.trials, train, folds)
    search_s = time.time() - t0
    rows = []
    for t in study.trials:
        rows.append({"trial": t.number, "state": t.state.name, "cv_top1": t.value,
                     "cv_log_loss": t.user_attrs.get("cv_log_loss"), "cv_f1_macro": t.user_attrs.get("cv_f1_macro"),
                     "cv_top1_std": t.user_attrs.get("cv_top1_std"), "baseline": t.user_attrs.get("baseline", False),
                     "duration_s": t.duration.total_seconds() if t.duration else None, **t.params})
    trials = pd.DataFrame(rows)
    trials.to_csv(OUT / "trials.csv", index=False)
    best = study.best_trial
    best_params = dict(best.params)
    (OUT / "best_params.json").write_text(json.dumps({"trial": best.number, "cv_top1": best.value,
                                                      "params": best_params}, indent=2))
    print(f"best trial {best.number}: CV Top-1 {best.value:.4f}  params {best_params}")

    arcs_tr, arcs_te = arc_tables(train, "train"), arc_tables(test, "student_test")
    untuned = {k: v for k, v in ZOO["xgb"].config.items()}
    comp = []
    for name, params in [("xgb_untuned", untuned), ("xgb_tuned", best_params)]:
        r = {"model": name, "params": json.dumps(params)}
        r |= cv_eval(params, train, folds, arcs_tr)
        r |= final_eval(name, params, train, test, arcs_te)
        comp.append(r)
        print(f"{name}: CV top1 {r['cv_top1_mean']:.4f} f1 {r['cv_f1_macro_mean']:.4f} | test top1 {r['test_top1']:.4f} "
              f"f1 {r['test_f1_macro']:.4f} logloss {r['test_log_loss']:.4f}", flush=True)
    comp = pd.DataFrame(comp)
    comp["search_wallclock_s_this_run"] = [search_s, search_s]
    comp["n_trials_total"] = len(study.trials)
    comp["n_trials_complete"] = int((trials.state == "COMPLETE").sum())
    comp["n_trials_pruned"] = int((trials.state == "PRUNED").sum())
    comp["search_total_trial_s"] = float(trials.duration_s.sum())
    comp.to_csv(OUT / "tuned_vs_untuned.csv", index=False)
    figures(study, comp)


if __name__ == "__main__":
    main()
