# PathSnap — Project Plan

Map-match noisy GPS tracks to a road network: generate candidate road segments per GPS point, score them with a trained classifier, then decode a connected route with HMM + Viterbi. Serve it via FastAPI (Laptop 1) and an evaluator UI (Laptop 2).

## Data facts that shape the design

- Each trip has its **own local road network** (`.nodes`, `.arcs`); node/arc IDs are only meaningful within a trip.
- Trips are spread **worldwide** → project each trip into a local metric projection instead of using degrees.
- Arcs are **directed** (one-way streets have no reverse arc) → heading vs arc bearing is informative; transitions must follow arc direction.
- `.route` is an ordered list of arc IDs, **not one label per GPS point** → per-point labels must be derived.
- Tracks are ~1 Hz but some trips have gaps up to minutes → decoder must handle gaps.

## Pipeline

```
.track ─► project ─► candidate arcs ─► features ─► classifier P(true arc)
.nodes/.arcs ─► directed graph + spatial index ────────────────────────┤
                                                                        ▼
                                                    HMM + Viterbi decoder
                                                                        ▼
                                                    predicted connected route
```

## Phases

| # | Phase | Output | Done when |
|---|---|---|---|
| 0 | Setup | repository, environment, structure, plan, tracker | repository pushed |
| 1 | Data loading + EDA | loaders, geo utilities, inventory, plots | data quality reviewed |
| 2 | Labels + candidates | per-point reference labels and candidate generator | candidate miss rate measured |
| 3 | Feature engineering | candidate-level feature table | features cached |
| 4 | Model experiments | classifier comparisons and CV metrics | comparison completed |
| 5 | Bayesian HPO | Optuna tuning and trial logs | tuned model evaluated |
| 6 | HMM + Viterbi | emissions, directed transitions, gap handling | continuity evaluated |
| 7 | Artifacts | saved model and feature configuration | inference reproducible |
| 8 | FastAPI backend | `/health`, `/trips`, `/predict` | every evaluation trip served |
| 9 | UI | Streamlit + Folium map and metrics | end-to-end over LAN |
| 10 | Integration test | API/UI dry run on student test trips | all trips succeed |
| 11 | Report | final PDF and traceable results | all figures reproducible |

## Evaluation protocol

- Train data: fitting, grouped cross-validation and hyperparameter optimization.
- Student test: model comparison and decoder tuning.
- Evaluator test: never used during development; keep local on the API host only.
- Primary selection metric: observation-level Top-1 accuracy; report route-level metrics after HMM.

## Integrity rules

- No evaluation ground truth in the prediction path or UI; no hardcoded outputs.
- Every reported number/figure must be produced by committed code; retain logs and seeds.
- AI assistance must be disclosed in the report; every member must be able to explain their work.
