# PathSnap — Dataset Analysis

*Owner: Sreekar · Dataset Analysis & Quality Assessment*

Measured findings from the exploratory analysis. Numbers come from
`experiments/results/eda_summary.json`, `eda_inventory.csv`,
`labels_trip_summary.csv`, `feature_table_stats.json`,
`feature_univariate.csv` and `candidate_miss_rate.csv`. Figures referenced here
are in `reports/figures/`.

## 1. Size and shape

| | train | student_test |
|---|---:|---:|
| Trips | 80 | 10 |
| GPS observations | 208,417 | 19,160 |
| Points per trip (median) | 1,529 | 1,420 |
| Points per trip (min–max) | 361 – 19,313 | 262 – 3,958 |
| Total track length | 2,978 km | 297 km |

Trips vary widely in length — from a few hundred points to ~19 k — so **trip is
the correct cross-validation group**: splitting by point would leak neighbouring
observations across folds. See `reports/figures/trip_sampling_distribution.png`.

## 2. Each trip is its own map

The road network is supplied per trip and the trips sit in different areas (trip
centres span roughly 8–37°E, 48–56°N). Networks range from ~8 k to ~90 k nodes
per trip. All candidate generation, labelling and features are therefore computed
**within a single trip**; there is no global graph.

## 3. Sampling and timing

- Sampling is close to 1 Hz: the pooled fraction of exactly-1 s steps is
  **96.5 % (train)** and **98.6 % (student_test)**.
- Gaps exist: **28 of 80** train trips and **2 of 10** student_test trips contain
  a sampling gap > 30 s. These are handled as sequence breaks rather than
  interpolated across.
- Timestamps are clean: **0** non-monotonic and **0** duplicate timestamps across
  all 90 trips.
- A few trips contain large position jumps (speed > 70 m/s between fixes): 10
  train trips, 1 student_test trip — consistent with GPS outliers.

## 4. GPS noise vs the road

The reason the task is non-trivial: GPS points do not sit on the road.

| GPS→nearest-route-arc distance | train | student_test |
|---|---:|---:|
| median | 5.6 m | 5.2 m |
| p95 | 140 m | 16.8 m |
| p99 | 204 m | 28.0 m |

The median offset is small (~5 m), but the train tail is heavy — a handful of
noisy trips drive p95 to 140 m (visible as the two outliers in
`reports/figures/dataset_quality_summary.png`, top-right). The student_test
partition is noticeably cleaner. Median arc length is ~25 m, so a 5 m offset is a
meaningful fraction of a block.

## 5. Candidate coverage (why radius = 50 m, k = 12)

From `candidate_miss_rate.csv`, at the chosen operating point **radius 50 m,
k = 12**:

| | train | student_test |
|---|---:|---:|
| Points with ≥1 candidate | 91.3 % | 100 % |
| True arc present in candidate set | 88.3 % | 99.8 % |
| Candidate **miss rate** | 11.7 % | 0.21 % |
| Mean candidates / point | 6.4 | 7.7 |

Coverage improves with radius and then plateaus (bottom-right panel of the
quality-summary figure). 50 m captures essentially all student_test labels; the
residual train miss is concentrated in the noisy trips above. This miss rate is
the ceiling on candidate-classifier accuracy and is reported as **Candidate Miss
Rate** in the modelling work.

## 6. Reference-label quality

Labels are derived by monotone track-to-route alignment (`labels.py`). From
`labels_trip_summary.csv`:

- Median point→label distance is small for most trips, but **2 training trips
  (`00000033`, `00000054`)** exceed the 50 m reliability threshold and are
  excluded from training (the `train_reliable` split: 78 trips).
- Route contiguity is near-perfect (mean ≈ 0.999); only a few trips have any
  route break, reflecting occasional gaps in the supplied route.
- See `reports/figures/label_distribution_analysis.png`.

## 7. Class balance and feature signal

- The learning problem is **imbalanced**: pooled positive rate ≈ **13.7 %**
  (train), **13.0 %** (student_test) — on average ~7 candidates per point, one of
  which is correct. Macro-F1 (not raw accuracy) is therefore the right primary
  metric.
- The most discriminative single features (univariate `auc_strength`) are
  **relative-distance** signals — `dist_minus_min` (0.82), `rank` (0.81),
  `dist_ratio` (0.80) — and **temporal-consistency** signals —
  `dist_win5_mean` (0.74), `dist_next1`/`dist_prev1` (~0.66): the correct arc is
  the one that stays near the trajectory over a window, not just at one instant.
  **Heading agreement** (`heading_cos_wide`, 0.69) separates parallel/opposing
  streets. Raw `speed`, `frac` and the Δt features are weak on their own.

## 8. Implications for modelling

1. Group by trip for cross-validation; fit learned preprocessing on training
   folds only.
2. Exclude the 2 unreliable trips from training.
3. Treat large gaps as sequence breaks (relevant to the HMM stage).
4. Optimise for Macro-F1 given the ~13 % positive rate.
5. Candidate miss rate (≈0.2 % student_test) bounds achievable accuracy — the
   classifier can only pick among the supplied candidates.
