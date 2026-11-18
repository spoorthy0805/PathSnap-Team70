# PathSnap — Data Dictionary

*Owner: Sreekar · Dataset Analysis & Quality Assessment*

Field-level reference for every raw file, the derived feature table, and the EDA
result tables. Conceptual overview is in
[`Dataset_Documentation.md`](Dataset_Documentation.md).

Units: coordinates are WGS-84 degrees (`lon`, `lat`); distances are metres after
projection to a local metric frame; time is seconds; headings are degrees.

---

## 1. Raw files

All files are whitespace-separated, no header. The 0-based **row index is the
id** for `.nodes` (node_id) and `.arcs` (arc_id).

### `<id>.track` — GPS observations
| Col | Name | Type | Notes |
|---:|---|---|---|
| 0 | `lon` | float | longitude, degrees |
| 1 | `lat` | float | latitude, degrees |
| 2 | `t` | float | timestamp, seconds; strictly increasing; nominally 1 Hz |

### `<id>.nodes` — road-network vertices
| Col | Name | Type | Notes |
|---:|---|---|---|
| 0 | `lon` | float | longitude, degrees |
| 1 | `lat` | float | latitude, degrees |

Row index = `node_id`, in `[0, n_nodes)`.

### `<id>.arcs` — directed road segments
| Col | Name | Type | Notes |
|---:|---|---|---|
| 0 | `from_node_id` | int | start node, in `[0, n_nodes)` |
| 1 | `to_node_id` | int | end node, in `[0, n_nodes)` |

Row index = `arc_id`, in `[0, n_arcs)`. Arcs are **directed**; a two-way street
appears as two arcs. An arc is a straight segment between its two nodes.

### `<id>.route` — reference route (where supplied)
| Col | Name | Type | Notes |
|---:|---|---|---|
| 0 | `arc_id` | int | ordered arcs traversed, in `[0, n_arcs)` |

This is a **sequence of arcs**, not one entry per GPS point. Consecutive arcs
normally share a node (contiguous chain).

---

## 2. Candidate table (derived) — `src/pathsnap/candidates.py`

One row per **(GPS point, candidate arc)**, nearest first, within 50 m, capped at
`k = 12`.

| Field | Type | Meaning |
|---|---|---|
| `point` | int | GPS observation index within the trip |
| `arc` | int | candidate `arc_id` |
| `dist` | float | exact point-to-arc distance, metres |
| `frac` | float | projection position along the arc, `[0, 1]` |
| `rank` | int | 0 = nearest arc for that point |

---

## 3. Feature table (derived) — `src/pathsnap/features.py`

One row per (point, candidate arc), **26 features**, computed from the GPS track
and road network only (no route information; no absolute coordinates/timestamps).
`mean_pos` / `mean_neg` / `auc_strength` below are from
`experiments/results/feature_univariate.csv` (strength = `|AUC − 0.5| × 2`,
higher = more discriminative on its own).

| Feature | Group | Meaning | auc_strength |
|---|---|---|---:|
| `dist` | geometry | point-to-arc distance (m) | 0.718 |
| `rank` | geometry | distance rank among the point's candidates (0 = nearest) | 0.810 |
| `dist_minus_min` | geometry | `dist` minus the point's nearest-candidate distance | 0.815 |
| `dist_ratio` | geometry | `(dist+1)/(min_dist+1)` | 0.800 |
| `frac` | geometry | projection fraction along the arc | 0.001 |
| `at_endpoint` | geometry | 1 if projection is at an arc endpoint | 0.662 |
| `n_cands` | cand-set | number of candidates kept for the point (≤12) | 0.369 |
| `n_within_radius` | cand-set | candidates within the 50 m radius before the `k` cap | 0.370 |
| `min_dist` | cand-set | nearest-candidate distance for the point | 0.027 |
| `second_gap` | cand-set | gap between 2nd-nearest and nearest candidate | 0.014 |
| `arc_len_log` | arc | `log1p(arc length in m)` | 0.439 |
| `two_way` | arc | 1 if the reverse arc also exists | 0.065 |
| `out_deg_end` | arc | out-degree of the arc's end node | 0.054 |
| `in_deg_start` | arc | in-degree of the arc's start node | 0.055 |
| `heading_cos` | direction | cosine between GPS heading (±1 step) and arc bearing | 0.605 |
| `heading_diff` | direction | angular difference (deg) for the ±1-step heading | 0.605 |
| `heading_cos_wide` | direction | cosine using a wider ±3-step heading | 0.689 |
| `heading_valid` | direction | 1 if the GPS displacement was ≥ 2 m (heading reliable) | 0.002 |
| `speed` | motion | local speed (m/s) from the ±1-step displacement and Δt | 0.061 |
| `dt_prev` | motion | seconds since the previous GPS point | 0.002 |
| `dt_next` | motion | seconds to the next GPS point | 0.002 |
| `gap_flag` | motion | 1 if an adjacent Δt exceeds 5 s | 0.002 |
| `dist_prev1` | temporal | distance from the previous point to this arc | 0.659 |
| `dist_next1` | temporal | distance from the next point to this arc | 0.661 |
| `dist_win5_mean` | temporal | mean distance to this arc over a ±2-point window | 0.739 |
| `dist_win5_max` | temporal | max distance to this arc over a ±2-point window | 0.672 |

**Label** (training/development only, from `src/pathsnap/labels.py`): a binary
target per row — 1 if the candidate arc equals the point's aligned reference arc.
Pooled positive rate ≈ 0.137 (train), 0.130 (student_test).

---

## 4. EDA / result tables — `experiments/results/`

### `eda_inventory.csv` — one row per trip (48 columns)
Per-trip inventory and quality metrics. Key columns: `trip_id`, `split`,
`n_points`, `nan_values`, `invalid_coords`, `duration_s`, `nonmonotonic_t`,
`duplicate_t`, `dt_median`, `dt_frac_1s`, `dt_max`, `gaps_gt_5s`, `gaps_gt_30s`,
`track_km`, `speed_median_ms`, `speed_p95_ms`, `jumps_gt_70ms`, `center_lon`,
`center_lat`, `extent_x_km`, `extent_y_km`, `n_nodes`, `n_arcs`,
`isolated_nodes`, `weak_components`, `largest_weak_frac`, `oneway_arc_frac`,
`self_loops`, `zero_length_arcs`, `arc_len_median_m`, `route_len`,
`route_contiguity`, `route_breaks`, `route_km`, `gps_to_route_median_m`,
`gps_to_route_p95_m`, `gps_to_route_max_m`, `points_per_route_arc`.

### `eda_summary.json` — per-partition rollup
Totals and pooled statistics for `train` and `student_test` (trips, points,
track km, `dt_frac_1s`, gap/jump counts, nan/invalid/non-monotonic counts,
isolated nodes, route contiguity, GPS→route distance percentiles, arc length).

### `labels_trip_summary.csv` — one row per trip
`trip_id`, `split`, `n_points`, `label_dist_median_m`, `label_dist_p95_m`,
`route_positions_used` (fraction of route positions that received ≥1 point),
`unreliable` (median label distance > 50 m).

### `feature_table_stats.json`
Per split (`train`, `student_test`, `train_reliable`): `rows`, `trips`,
`points_total`, `points_with_candidates`, `points_label_in_candidates`,
`positive_rate`, `mean_candidates_per_point`.

### `feature_univariate.csv`
Per feature: `mean_pos`, `mean_neg`, `auc`, `auc_strength`.

### `candidate_miss_rate.csv`
Candidate coverage sweep over `radius_m` × `k` for each `split`/`subset`:
`miss_rate` (label arc absent from the candidate set), `no_candidate_rate`
(point has no candidate at all), `mean_candidates`, `p95_candidates`.

### `data_quality_report.json` — produced by `scripts/validate_dataset.py`
See [`Data_Quality_Report.md`](Data_Quality_Report.md).
