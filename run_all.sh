#!/usr/bin/env bash
# run_all.sh — Run the full PathSnap ML pipeline (Phases 1 to 4)
set -euo pipefail

# Ensure we run from the repository root directory
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Resolve python interpreter (prioritize activated venv/python3)
PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    PYTHON="python"
fi

echo "============================================================"
echo "PathSnap Pipeline: Running scripts in order"
echo "Repository root: $REPO_ROOT"
echo "Python interpreter: $($PYTHON --version 2>&1) ($PYTHON)"
echo "============================================================"

echo ""
echo ">>> [1/5] Step 1: Exploratory Data Analysis"
$PYTHON scripts/01_eda.py

echo ""
echo ">>> [2/5] Step 2: Labels and Candidates Generation"
$PYTHON scripts/02_labels_candidates.py

echo ""
echo ">>> [3/5] Step 3: Feature Engineering"
$PYTHON scripts/03_features.py

echo ""
echo ">>> [4/5] Step 4: Model Experiments"
# Accepts optional flags, e.g. --models xgb,lgbm
$PYTHON scripts/04_model_experiments.py "$@"

echo ""
echo ">>> [5/5] Step 5: Model Plots & Comparison Summary"
$PYTHON scripts/04b_model_plots.py

echo ""
echo "============================================================"
echo "PathSnap Pipeline finished successfully!"
echo "Outputs are available under experiments/results/ and reports/figures/"
echo "============================================================"
