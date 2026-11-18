"""Project paths. Override the data root with the PATHSNAP_DATA env var."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("PATHSNAP_DATA", REPO_ROOT / "data"))
SPLITS = ("train", "student_test")  # eval_test is deliberately excluded from development code

RESULTS_DIR = REPO_ROOT / "experiments" / "results"
FIGURES_DIR = REPO_ROOT / "reports" / "figures"
MAPS_DIR = REPO_ROOT / "reports" / "maps"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
CACHE_DIR = REPO_ROOT / "cache"

SEED = 42

# Candidate generation (chosen in Phase 2 from experiments/results/candidate_miss_rate.csv)
CAND_RADIUS_M = 50.0
CAND_K = 12
UNRELIABLE_TRIPS_RULE_M = 50.0  # trips whose median point→label distance exceeds this are excluded from training
