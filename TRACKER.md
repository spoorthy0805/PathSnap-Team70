# PathSnap — Progress Tracker

Update this file in the same commit as the work. Status: ⬜ todo · 🟡 in progress · ✅ done · ⛔ blocked

## Team

| Member    | SRN           | Focus                                               |
| --------- | ------------- | --------------------------------------------------- |
| Spoorthy  | PES1UG24AM280 | setup, data pipeline                                |
| Sreekar   | PES1UG24AM282 | Dataset Documentation                               |
| Praneetha | PES1UG24AM281 | pipeline automation, modeling                       |
| SriRam    | PES1UG24AM283 | Documentation, Validation, Frontend-API integration |

## Decisions log

| Date       | Decision                                                                                                                                               | Why                                                                                                                                                                                                                                                                                          |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| 2026-11-18 | Per-trip azimuthal-equidistant projection                                                                                                              | trips span several countries; need metres, not degrees                                                                                                                                                                                                                                       |
| 2026-11-18 | Raw data not committed; `data/` holds symlinks/local copies                                                                                            | size, and eval data must stay on Laptop 1 only                                                                                                                                                                                                                                               |
| 2026-11-18 | Per-point labels = monotone DP alignment of track to route positions                                                                                   | `.route` has no per-point labels; monotonicity stops points snapping to a later/earlier pass of the same road; handles loops (00000056)                                                                                                                                                      |
| 2026-11-18 | Candidates: arcs within **R = 50 m**, nearest **K = 12**                                                                                               | train-reliable miss 1.57%, student_test 0.21%, ~7 candidates/point; R=75 only saves ~0.5 pp train miss for +20% candidates (`candidate_miss_rate.csv`)                                                                                                                                       |
| 2026-11-18 | Trips with median point→label distance > 50 m are **unreliable** → excluded from training, kept in reporting                                           | rule flags exactly 00000033, 00000054 (next-worst median is 11 m); their labels are 120–140 m off the GPS                                                                                                                                                                                    |
| 2026-11-18 | Folium base map = OpenStreetMap tiles                                                                                                                  | CartoDB tiles now require an API key                                                                                                                                                                                                                                                         |
| 2026-11-18 | Problem framed as **binary candidate classification** (is this candidate the reference arc?), Top-1 = argmax score per point                           | fixed-size label space doesn't exist (arc ids are per-trip); gives P(arc) per candidate = HMM emission                                                                                                                                                                                       |
| 2026-11-18 | Features exclude lon/lat, timestamps, trip ids                                                                                                         | prevent memorising locations; same code runs in API at inference                                                                                                                                                                                                                             |
| 2026-11-18 | CV = 5-fold GroupKFold by trip (shuffle, seed 42) on 78 reliable trips; same folds reused in Phases 5–6 (`experiment.cv_splits`)                       | no trip leaks between fit and validation; identical folds make tuned/untuned comparable                                                                                                                                                                                                      |
| 2026-11-18 | **Final model family: XGBoost**                                                                                                                        | best CV macro-F1 (0.9369), CV Top-1 (0.8863), test Top-1 (0.8970) and lowest log loss (CV 0.082, test 0.073) — calibrated probabilities matter because they become HMM emissions; fast inference (3.9 ms / 1k points). LightGBM within noise but its validation loss rises after ~150 rounds |
| 2026-11-18 | Bayesian HPO: Optuna TPE (seed 42, 10 random start-up trials, median pruner), objective = mean CV Top-1 on the Phase 4 folds, trial 0 = untuned config | objective matches the primary metric; same folds keep tuned vs untuned comparable; student_test untouched during search                                                                                                                                                                      |
| 2026-11-18 | HMM: states = candidate arcs; emission = alpha·log P(classifier); transition = −                                                                       | route dist − GPS dist                                                                                                                                                                                                                                                                        | /beta (Newson–Krumm), directed bounded Dijkstra ≤ 1 km; unreachable → −30; gap > gap_s → uniform transition (chain split); points without candidates skipped | uses the classifier's calibrated probabilities and the road graph's direction/connectivity; never dead-ends |
| 2026-11-18 | Decoder params tuned on train out-of-fold predictions only: alpha 1.0, beta 2.0 m, gap_s 120 s                                                         | grid of 12 (alpha × beta) + 3 gap values; student_test only scored afterwards                                                                                                                                                                                                                |
| 2026-11-18 | **Final pipeline = untuned XGBoost + HMM**                                                                                                             | after decoding, untuned beats tuned on train OOF Top-1 (0.9323 vs 0.9312) — the selection set — and student_test agrees (0.9552 vs 0.9528); also 2× faster inference and better log loss                                                                                                     |
| 2026-11-18 | Model saved as XGBoost native JSON (not pickle) + a JSON pipeline config                                                                               | portable across Python/xgboost minor versions on the demo laptop; config holds every inference setting in one place                                                                                                                                                                          |
| 2026-11-18 | API never returns the reference route — only metrics; prediction loads trips with `with_route=False`, scoring reads `.route` afterwards                | brief: ground truth must stay server-side and never reach the model or UI                                                                                                                                                                                                                    |
| 2026-11-18 | UI shows GPS + predicted route only, **no reference layer** (PLAN listed one)                                                                          | brief: ground truth must never reach the UI; the API returns metrics only                                                                                                                                                                                                                    |
| 2026-11-18 | UI is a pure HTTP client (`requests` + Streamlit), no `pathsnap` import, no trip files                                                                 | Laptop 2 must not be able to bypass the API                                                                                                                                                                                                                                                  |
| 2026-11-18 | Two-laptop link over **Tailscale** (personal tailnet, machine shared to Laptop 2)                                                                      | fixed address independent of venue Wi-Fi; client isolation on campus/hotspot networks can block LAN traffic                                                                                                                                                                                  |
| 2026-11-18 | Report built by a script from result CSVs (HTML → Chrome headless PDF)                                                                                 | every number in the PDF traceable to `experiments/results/`                                                                                                                                                                                                                                  |
| 2026-11-18 | Missing heading (stationary, <2 m move) → `heading_cos=0`, `heading_diff=90`, `heading_valid=0`; edge neighbours clipped                               | neutral fill works for every model family (LogReg/KNN/MLP can't take NaN)                                                                                                                                                                                                                    |

## Data-quality findings

Source: `experiments/results/eda_inventory.csv`, `eda_summary.json` (train + student_test only).

| Finding          | Evidence                                                                                                                                                                       | Implication                                                                                                      |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| Partitions       | train 80 trips / 208,417 pts / 2,978 km; student_test 10 trips / 19,160 pts / 297 km                                                                                           | ~227k labelled observations                                                                                      |
| Clean values     | 0 NaNs, 0 invalid coords, 0 non-monotonic timestamps, 0 invalid route arc IDs                                                                                                  | no imputation needed                                                                                             |
| Sampling         | 96.5% (train) / 98.6% (test) of steps exactly 1 s; some fractional dt                                                                                                          | treat dt as a feature, not a constant                                                                            |
| Gaps             | 28/80 train trips have gaps > 30 s; trip 00000090 has a ~42 h gap, 00000078 ~15 h                                                                                              | decoder must break/relax transitions across gaps                                                                 |
| Teleport jumps   | 10 train trips have steps > 70 m/s (00000013: 95 such steps)                                                                                                                   | outlier-robust emissions / speed feature                                                                         |
| GPS noise        | GPS→reference-route distance median ≈ 5 m, test p95 ≈ 17 m, p99 ≈ 28 m                                                                                                         | candidate radius ~30–50 m should cover most points                                                               |
| **Offset trips** | 00000054 median GPS→route **139 m** (GPS 84 m from _any_ arc, ≈132 m southward shift); 00000033 median **122 m** but only 8 m from some arc (route ≈ parallel road 122 m away) | reference labels unreliable here → candidate generation will miss; flag/exclude from training, discuss in report |
| Networks         | 3.5k–165k arcs per trip; median arc 25 m; one-way share 0–65%; 89/90 trips have >1 weakly connected component (largest ≈ 97% of nodes)                                         | spatial index needed; shortest paths must allow "unreachable"                                                    |
| Route encoding   | routes are ordered arc IDs, ≈ 99.9% contiguous (v of arc i = u of arc i+1); 11 trips have 1–3 breaks; 00000056 repeats 255 arcs (loops)                                        | labels need monotone track↔route alignment that allows revisits                                                  |

Figures: `reports/figures/eda_*.png`. Maps: `reports/maps/eda_trip_{00000095,00000002,00000054,00000090,00000000}.html`.

## Results snapshot

Source of truth: `experiments/results/`.

**Phase 2 — candidate miss rate (%), reliable trips** (`candidate_miss_rate.csv`)

| Split        | R (m)  | K=4  | K=8  | K=12     | K=16 | all  | mean cands @K=12 |
| ------------ | ------ | ---- | ---- | -------- | ---- | ---- | ---------------- |
| train        | 30     | 4.58 | 2.80 | 2.57     | 2.50 | 2.41 | 5.1              |
| train        | **50** | 3.75 | 1.84 | **1.57** | 1.49 | 1.39 | **7.0**          |
| train        | 75     | 3.33 | 1.39 | 1.10     | 1.01 | 0.87 | 8.4              |
| student_test | 30     | 3.43 | 0.98 | 0.81     | 0.79 | 0.77 | 5.5              |
| student_test | **50** | 2.87 | 0.40 | **0.21** | 0.17 | 0.14 | **7.7**          |
| student_test | 75     | 2.85 | 0.37 | 0.18     | 0.14 | 0.11 | 9.0              |

**Phase 3 — feature table** (`feature_table_stats.json`, `feature_univariate.csv`, `heuristic_baselines.csv`)

- Rows: train 1,342,714 (reliable trips: 1,276,667), student_test 146,896; ~7 candidates/point; positive rate ≈ 13–14%.
- Strongest single features (train, |2·AUC−1|): `dist_minus_min` 0.82, `rank` 0.81, `dist_ratio` 0.80, `dist_win5_mean` 0.74, `dist` 0.72, `heading_cos_wide` 0.69, `at_endpoint` 0.66.
- Point-level context features (`speed`, `dt_*`, `min_dist`, `second_gap`) are ~0.5 alone — they only help through interactions.
- Rule-based Top-1 (no learning, all GPS points): nearest arc **49.5% train / 53.7% test** (two-way twins tie); nearest + heading **72.1% / 82.6%**. ML models must beat these.

**Phase 4 — model comparison** (`experiments/results/phase4/comparison.md`, figures `reports/figures/models_*.png`)

| Model   | Family         | CV macro-F1         | CV Top-1   | Test Top-1 | Test macro-F1 | Test log loss | Continuity | Fit s/fold | Infer ms/1k pts |
| ------- | -------------- | ------------------- | ---------- | ---------- | ------------- | ------------- | ---------- | ---------- | --------------- |
| logreg  | Linear         | 0.9292 ± 0.0111     | 0.8795     | 0.8764     | 0.9235        | 0.0985        | 0.9339     | 0.7        | 0.1             |
| gnb     | Probabilistic  | 0.8708 ± 0.0155     | 0.8574     | 0.8618     | 0.8673        | 0.4773        | 0.9239     | 0.1        | 0.3             |
| knn     | Instance-based | 0.9245 ± 0.0117     | 0.8717     | 0.8745     | 0.9199        | 0.1295        | 0.9296     | ~0 (lazy)  | 214.9           |
| dtree   | Tree           | 0.9313 ± 0.0121     | 0.8792     | 0.8837     | 0.9305        | 0.0864        | 0.9346     | 11.3       | 0.2             |
| rf      | Bagging        | 0.9364 ± 0.0113     | 0.8841     | 0.8859     | 0.9328        | 0.0769        | 0.9364     | 24.9       | 6.8             |
| lgbm    | Boosting       | 0.9359 ± 0.0127     | 0.8839     | 0.8945     | 0.9360        | 0.0765        | 0.9378     | 8.7        | 17.3            |
| **xgb** | Boosting       | **0.9369 ± 0.0127** | **0.8863** | **0.8970** | **0.9375**    | **0.0731**    | 0.9414     | 6.9        | 3.9             |
| mlp     | Neural net     | 0.9328 ± 0.0117     | 0.8830     | 0.8890     | 0.9309        | 0.0801        | 0.9423     | 12.1       | 0.7             |

Observations: every model beats the 0.826 rule baseline; fold-to-fold Top-1 spread (≈0.84–0.92) is larger than the gap between the top 4 models; GNB is badly calibrated (log loss 0.48, correlated distance features violate independence); KNN trained on a 60k subsample and is the slowest at inference; loss curves show LightGBM validation loss rising after ~150 rounds and the MLP's after ~4 epochs (mild overfitting), XGBoost's flat.

**Phase 5 — Bayesian optimisation of XGBoost** (`experiments/results/phase5/`, figures `reports/figures/hpo_*.png`)

40 trials (0 pruned), 51 min of trial time. Best = trial 39: n_estimators 1000, learning_rate 0.094, max_depth 6, min_child_weight 2.77, subsample 0.83, colsample_bytree 0.45, gamma 0.18, reg_lambda 1.08, reg_alpha 4.7e-4. Most influential (fANOVA): learning_rate 0.25, subsample 0.18, min_child_weight 0.16, n_estimators 0.16; max_depth 0.01.

| Model   | CV Top-1       | CV macro-F1 | CV log loss | Test Top-1 | Test macro-F1 | Test log loss | Continuity | Infer ms/1k pts |
| ------- | -------------- | ----------- | ----------- | ---------- | ------------- | ------------- | ---------- | --------------- |
| Untuned | 0.8863 ± 0.034 | 0.9369      | 0.0824      | 0.8970     | 0.9375        | 0.0731        | 0.9414     | 6.2             |
| Tuned   | 0.8871 ± 0.035 | 0.9360      | 0.0850      | 0.8968     | 0.9365        | 0.0739        | 0.9418     | 13.1            |

Tuning did **not** meaningfully outperform: +0.08 pp CV Top-1 (fold std ±3.4 pp), test level, log loss slightly worse, inference 2× slower. The search surface is flat — the classifier is limited by features/ambiguity, not hyperparameters. Both are carried into Phase 6 to compare after decoding.

**Phase 6 — HMM + Viterbi** (`experiments/results/phase6/<model>/`, figures `reports/figures/hmm_*.png`, maps `reports/maps/hmm_trip_*.html`)

| Classifier      | Decoding        | Train OOF Top-1 | Test Top-1 | Test continuity | Test route F1 (len-weighted) | Unbridged route gaps (test) |
| --------------- | --------------- | --------------- | ---------- | --------------- | ---------------------------- | --------------------------- |
| XGB untuned     | classifier only | 0.8850          | 0.8970     | 0.9414          | 0.8893                       | 175                         |
| **XGB untuned** | **+ HMM**       | **0.9323**      | **0.9552** | **0.9840**      | **0.9905**                   | **6**                       |
| XGB tuned       | classifier only | 0.8858          | 0.8968     | 0.9418          | 0.8920                       | 166                         |
| XGB tuned       | + HMM           | 0.9312          | 0.9528     | 0.9838          | 0.9889                       | 6                           |

- HMM improved Top-1 on all 10 student_test trips (largest: 00000077 0.775 → 0.902; two trips reach 100%). Route precision jumps 0.81 → 0.98: classifier-only routes contain many stray arcs.
- HMM hurt on 4 of 78 train trips, by ≤ 1.1 pp (worst 00000075 0.872 → 0.861): places where wrong-but-connected roads beat the classifier's correct pick.
- Smaller beta (trust GPS displacement more) was consistently better; alpha 1 (plain log-probabilities) best; gap threshold barely matters (only a few trips have gaps).
- Remaining errors: trips 00000065 / 00000009 (~0.89–0.92) — dense junctions and parallel roads.

Note: `hmm_*.png` = tuned run, `hmm_xgb_untuned_*.png` = final model; maps are from the final model.

K=2/4 miss a lot because every two-way street contributes two arcs (one per direction) at identical distance — direction must be decided by features (heading), not distance.

## Log

- **2026-11-18** — Repo scaffolded, plan written, environment set up.
- **2026-11-18** — Phase 1 done: loaders (`io.py`), per-trip projection (`geo.py`), network stats (`network.py`), Folium builder (`viz.py`), EDA script + outputs. Next: Phase 2 labels + candidates.
- **2026-11-18** — Phase 2 done: `labels.py` (monotone DP alignment, ~0.3 s/trip), `candidates.py` (KD-tree over ≤20 m arc pieces, verified identical to brute force), miss-rate grid + figures, label-assignment map (`reports/maps/labels_trip_00000095.html`). Next: Phase 3 features.
- **2026-11-18** — Phase 3 done: `features.py` (26 inference-safe features), feature tables cached as parquet (~15 s for all 90 trips), univariate AUC / class-distribution / correlation figures, rule-based baselines. Next: Phase 4 model experiments.
- **2026-11-18** — Phase 4 done: `models.py` (8-model zoo, loss histories for LGBM/XGB/MLP), `metrics.py`, `experiment.py`, `04_model_experiments.py` + `04b_model_plots.py`; XGBoost selected. Phase 5 search launched.
- **2026-11-18** — Phase 5 done: `05_bayesian_hpo.py` (Optuna study in `optuna_xgb.db`, trials.csv, importance, tuned-vs-untuned). Tuning gain within noise. Phase 6 running for both models.
- **2026-11-18** — Phase 6 done: `hmm.py` (bounded Dijkstra route graph, lattice, Viterbi, route reconstruction, route overlap), `06_hmm_viterbi.py` (OOF predictions, decoder grid, classifier-vs-HMM, maps). Final pipeline = untuned XGBoost + HMM. Next: Phase 7 artifacts + Phase 8 API.
- **2026-11-18** — Phases 7–8 done: `pipeline.py` (raw files → features → XGBoost → HMM → route; separate server-side scoring), `07_export_artifacts.py` (end-to-end check reproduces student_test Top-1 0.9552, 33–512 ms/trip), `api/main.py` (FastAPI with request logging). Next: Phase 9 UI.
- **2026-11-18** — Phases 9–11: `ui/app.py`, `08_integration_test.py` (10/10 trips over HTTP, 404 for unknown id, pooled Top-1 0.9552), report builder + PDF, UI/API screenshots in `reports/screenshots/`. Fixed decoder-grid figure (was plotting only the final gap_s slice). Next: two-laptop LAN dry run, fill team details in report.
- **2026-11-18** — Two-laptop run done over Tailscale: Laptop 2 UI → Laptop 1 API, 10/10 trips, pooled Top-1 0.9552; screenshots + Laptop 1 log in report §7. UI legend given explicit dark text (was faint in dark theme). Final-review slides exported to `reports/PathSnap_Final_Review_Slides.pdf`.
