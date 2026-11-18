# PathSnap — Data Quality Report

*Owner: Sreekar · Dataset Analysis & Quality Assessment*

This report is produced by `scripts/validate_dataset.py`, which re-reads every
raw trip independently of the main pipeline and checks the structural and
value-level properties the project relies on. The machine-readable output is
`experiments/results/data_quality_report.json`; this document summarises it.

Regenerate with:

```bash
python scripts/validate_dataset.py
```

## Overall result

**Status: `pass_with_warnings`** across **90 trips / 227,577 GPS observations**.
No hard integrity violations were found; the warnings are inherent properties of
the data (sampling gaps and a few route breaks), not errors.

## Integrity checks (hard failures) — all zero

| Check | train | student_test | total |
|---|---:|---:|---:|
| NaN / non-finite values | 0 | 0 | 0 |
| Out-of-range coordinates | 0 | 0 | 0 |
| Non-monotonic timestamps | 0 | 0 | 0 |
| Arc node-index out of range | 0 | 0 | 0 |
| Route arc-index out of range | 0 | 0 | 0 |
| Isolated nodes | 0 | 0 | 0 |

Every `.track`, `.nodes`, `.arcs` and `.route` file is present and parses with the
expected column counts; all node and arc indices are in range; all timestamps are
strictly increasing. Totals validated: 2,024,174 nodes and 4,061,945 arcs across
the two partitions.

## Per-trip grading

A trip is graded **pass** (clean), **warn** (sampling gap > 30 s or a route
break, but otherwise valid) or **fail** (any hard violation).

| Partition | Trips | Pass | Warn | Fail |
|---|---:|---:|---:|---:|
| train | 80 | 49 | 31 | 0 |
| student_test | 10 | 8 | 2 | 0 |

**No trip failed.**

## Warnings (expected data properties)

- **Sampling gaps > 30 s** — 28 train trips, 2 student_test trips. Handled as
  sequence breaks downstream, not interpolated.
- **Route breaks** (consecutive reference arcs that do not share a node) — 9
  train trips, 1 student_test trip. A limitation of the supplied reference
  routes, not of our processing.
- **Unreliable for training** — trips whose median point→label distance exceeds
  50 m: **`00000033`, `00000054`** (cross-referenced from
  `labels_trip_summary.csv`). Excluded from the training split.

The full lists of flagged trip ids are in the JSON report under `flags`.

## Figures

- `reports/figures/dataset_quality_summary.png` — integrity violations,
  GPS→route offset vs trip length, 1 Hz sampling fraction, candidate miss rate
  vs radius.
- `reports/figures/trip_sampling_distribution.png` — points per trip, largest
  gap per trip, median speed per trip.
- `reports/figures/label_distribution_analysis.png` — median point→label distance
  (with the 50 m reliability line), route-position coverage, reliable vs
  unreliable counts.

## Known limitations

1. **Reference routes are not perfect ground truth.** A few trips have route
   breaks and two have large point→label offsets; routes are treated as reference
   information, not verified physical truth.
2. **GPS noise is heavy-tailed.** Median offset ≈ 5 m, but a handful of train
   trips reach 120–140 m median offset and drive the partition's p95 to 140 m.
3. **Candidate miss rate is non-zero.** At radius 50 m / k = 12, ~11.7 % of train
   points (0.21 % student_test) do not have the true arc among their candidates —
   an upper bound on achievable classifier accuracy.
4. **Validation scope.** This validator checks structure, value ranges, timestamp
   ordering, index validity, connectivity (isolated nodes) and route contiguity.
   It does not re-derive the full graph-component analysis in `scripts/01_eda.py`;
   the two are consistent where they overlap.
