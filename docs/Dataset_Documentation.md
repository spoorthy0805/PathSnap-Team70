# PathSnap — Dataset Documentation

*Owner: Sreekar · Dataset Analysis & Quality Assessment*

This document explains how the PathSnap dataset is organised, what each raw file
represents, and how the different components relate to one another. Field-level
details are in [`Data_Dictionary.md`](Data_Dictionary.md); measured findings are
in [`Dataset_Analysis.md`](Dataset_Analysis.md); the automated validation results
are in [`Data_Quality_Report.md`](Data_Quality_Report.md).

## 1. What the dataset is

The task is **map matching / route reconstruction**: given a noisy GPS
trajectory, recover the sequence of road segments the vehicle actually
travelled. The dataset therefore pairs, for each trip, a **GPS track** with the
**road network** of that trip's area and (for development partitions) a
**reference route**.

A key structural point: **the road network is supplied per trip**, not as one
global map. Every trip directory carries its own `.nodes` and `.arcs`, and the
trips are drawn from different geographic areas (trip centres range across
roughly longitude 8–37°E, latitude 48–56°N). All spatial work is done within a
single trip.

## 2. Partitions

| Partition | Trips | GPS points | Purpose |
|---|---:|---:|---|
| `train` | 80 | 208,417 | Training, cross-validation, model selection |
| `student_test` | 10 | 19,160 | Development-stage evaluation and comparison |
| `evaluator_test` | 10 | — | Final evaluation only (held out, not in this repo) |

Leakage rules the pipeline respects: no trip spans two partitions; evaluator
labels are never used for training, feature selection or model/threshold choice;
learned preprocessing is fit on training data only; reference labels are kept
out of the features (see §5).

## 3. Raw files (one set per trip)

Each trip lives in `data/<partition>/<trip_id>/` with the 8-digit trip id as the
filename stem. All files are whitespace-separated, no header. **Coordinate order
in the raw files is `longitude, latitude`.**

| File | Rows are | Columns |
|---|---|---|
| `<id>.track` | GPS observations, time-ordered | `lon  lat  t(seconds)` — nominally 1 Hz |
| `<id>.nodes` | road-network vertices; **row index = `node_id`** | `lon  lat` |
| `<id>.arcs` | directed road segments; **row index = `arc_id`** | `from_node_id  to_node_id` |
| `<id>.route` | reference route (where supplied) | ordered `arc_id` values traversed |

A row in a GPS file does **not** correspond to a row in a network file — the
links between them are derived (see §4).

## 4. How the components relate

```
.track  (GPS points, lon/lat/t)
   │  project to a local metric frame (pyproj), then for each point:
   ▼
candidate arcs   ── nearest road segments within 50 m, k≤12      (src/pathsnap/candidates.py)
   │  a point-level label is derived by aligning the track to the route:
   ▼
.route (arc_id sequence) ──monotone DP alignment──▶ one reference arc per point  (src/pathsnap/labels.py)
   │
   ▼
feature table   ── one row per (point, candidate arc), 26 features (src/pathsnap/features.py)
```

- **Candidate generation.** Arcs are split into ≤20 m pieces whose midpoints are
  indexed in a KD-tree; for each GPS point the arcs within a 50 m radius are
  gathered, the exact point-to-arc distance is computed, and the nearest `k=12`
  are kept.
- **Labelling.** `.route` is an ordered arc list, **not** one label per point, so
  a label is produced by a monotone dynamic-programming alignment of the track to
  route *positions* (the vehicle never moves backwards along its own route),
  minimising total point-to-arc distance. This yields one reference arc per GPS
  observation plus its distance.
- **Reliability rule.** A trip whose median point→label distance exceeds 50 m is
  flagged *unreliable* and excluded from training (2 of the 80 training trips).

## 5. What stays out of the model

Features are computed from the **GPS track and road network only** — never from
`.route`. Absolute coordinates and timestamps are deliberately excluded so models
cannot memorise locations, and the identical feature code runs at inference time
in the API. Reference routes are used only to build labels for development and to
evaluate predictions server-side.

## 6. Where things live in the repo

| Area | Path |
|---|---|
| Raw data | `data/train/`, `data/student_test/` |
| Pipeline code | `src/pathsnap/` (`io`, `network`, `geo`, `candidates`, `labels`, `features`, …) |
| EDA / result tables | `experiments/results/` (`eda_inventory.csv`, `eda_summary.json`, `labels_trip_summary.csv`, `feature_table_stats.json`, `feature_univariate.csv`, `candidate_miss_rate.csv`) |
| Dataset validation | `scripts/validate_dataset.py` → `experiments/results/data_quality_report.json` |
| Figures | `reports/figures/` |

## 7. How to reproduce the validation

From the repository root:

```bash
pip install -r requirements.txt
python scripts/validate_dataset.py
```

This re-reads every raw trip, writes `experiments/results/data_quality_report.json`,
and regenerates the three dataset-quality figures in `reports/figures/`.
