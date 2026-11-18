# PathSnap

Map-matching noisy GPS trajectories to road networks: candidate road segments per GPS point → ML classifier scores → HMM/Viterbi route reconstruction, served by a FastAPI backend (Laptop 1) to an evaluator UI (Laptop 2).

- **Plan:** [PLAN.md](PLAN.md)
- **Progress, findings, decisions:** [TRACKER.md](TRACKER.md)
- **Team workflow:** [CONTRIBUTING.md](CONTRIBUTING.md)

## Setup

```bash
git clone https://github.com/spoorthy0805/PathSnap-Team70.git
cd pathsnap
python3.11 -m venv .venv && source .venv/bin/activate   # or: uv venv --python 3.11
pip install -r requirements.txt && pip install -e .
```

Place the dataset under `data/` as described in [data/README.md](data/README.md) (symlinks are fine), or set `PATHSNAP_DATA=/path/to/data`.

## Dataset

Each trip contains four header-less, whitespace-separated files:

| File     | Contents                                                                                                           |
| -------- | ------------------------------------------------------------------------------------------------------------------ |
| `.track` | GPS observations containing longitude (`lon`), latitude (`lat`), and timestamp (`t`) in seconds.                   |
| `.nodes` | Road network node coordinates containing longitude (`lon`) and latitude (`lat`). Row index represents the node ID. |
| `.arcs`  | Directed road segments containing source node (`u`) and destination node (`v`). Row index represents the arc ID.   |
| `.route` | Ordered sequence of arc IDs representing the reference route taken by the trip.                                    |

## Run

To run the complete pipeline end-to-end:

```bash
bash run_all.sh                             # runs all steps (Phases 1-4)
bash run_all.sh --models xgb,lgbm           # or pass flags to run a model subset
```

Or run individual steps:

| Step                   | Command                                                                                                               | Output                                                                                                                           |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| 1. EDA                 | `python scripts/01_eda.py`                                                                                            | `experiments/results/eda_*`, `reports/figures/eda_*.png`, `reports/maps/eda_trip_*.html`                                         |
| 2. Labels + candidates | `python scripts/02_labels_candidates.py`                                                                              | `cache/labels/*.npz`, `experiments/results/{labels_trip_summary,candidate_miss_rate}.csv`, `reports/figures/{cand,labels}_*.png` |
| 3. Features            | `python scripts/03_features.py`                                                                                       | `cache/features/*.parquet`, `experiments/results/{feature_*,heuristic_baselines}.*`, `reports/figures/feat_*.png`                |
| 4. Model comparison    | `python scripts/04_model_experiments.py` (`--models xgb,lgbm` for a subset), then `python scripts/04b_model_plots.py` | `experiments/results/phase4/`, `reports/figures/models_*.png`, `cache/{preds,models}/`                                           |
| 5. Bayesian HPO        | `python scripts/05_bayesian_hpo.py --trials 40` (resumable)                                                           | `experiments/results/phase5/`, `reports/figures/hpo_*.png`, `cache/{preds,models}/xgb_{tuned,untuned}*`                          |
| 6. HMM + Viterbi       | `python scripts/06_hmm_viterbi.py --model xgb_untuned --maps` (also `--model xgb_tuned`)                              | `experiments/results/phase6/<model>/`, `reports/figures/hmm_*.png`, `reports/maps/hmm_trip_*.html`                               |
| 7. Export artifacts    | `python scripts/07_export_artifacts.py`                                                                               | `artifacts/xgb_final.json`, `artifacts/pipeline_config.json`, `experiments/results/phase7/verification.csv`                      |
| 8. Integration test    | `python scripts/08_integration_test.py --api http://<laptop-1-ip>:8000` (API running)                                 | `experiments/results/phase10/integration.csv`                                                                                    |
| 9. Report              | `python scripts/09_build_report.py` (needs Google Chrome for the PDF)                                                 | `reports/report.html`, `reports/PathSnap_Report.pdf`                                                                             |

## API (Laptop 1)

The API needs only `artifacts/` and the trip folders; no training data or cache.

```bash
PATHSNAP_EVAL_DIR=data/eval_test uvicorn api.main:app --host 0.0.0.0 --port 8000
```

- `GET /health` — model and data status
- `GET /trips` — trip ids in `PATHSNAP_EVAL_DIR`
- `POST /predict` with `{"trip_id": "..."}` — runs the full pipeline and returns GPS points, the predicted route as `[lat, lon]` polylines, server-side metrics (Top-1, route precision/recall/F1, continuity) and timings. The reference route is never returned.

Every request is logged to the console (time, endpoint, client IP, status, latency, per-trip metrics). For development, point it at `data/student_test`. Find Laptop 1's LAN IP with `ipconfig getifaddr en0` (macOS) and allow incoming connections on port 8000 if the firewall asks.

## UI (Laptop 2)

The UI only talks to the API over HTTP; it needs `streamlit`, `streamlit-folium`, `folium`, `requests`, `pandas` and no data.

```bash
PATHSNAP_API=http://<laptop-1-ip>:8000 streamlit run ui/app.py --server.address 0.0.0.0
```

- Sidebar: API URL and connection status (`/health`).
- **Single trip:** choose one of the trips listed by `/trips`, press _Match trip_; shows Top-1, route F1/precision/recall, continuity, timings and a Folium map (GPS points, predicted route, layer control, legend).
- **All trips:** runs every trip through `/predict` and shows the per-trip table and pooled Top-1.

## Demo day checklist

1. Laptop 1: copy the 10 evaluation trip folders to `data/eval_test/`, start the API (above), note its IP.
2. Laptop 2: start the UI with `PATHSNAP_API` pointing at Laptop 1; check the sidebar says _Connected · 10 trips_.
3. Keep the Laptop 1 terminal visible: every UI click shows up there as `POST /predict … -> 200` with the trip's metrics.

## Repository layout

```
src/pathsnap/      library code (io, geo, network, viz, …)
scripts/           numbered pipeline scripts (01_eda.py, …)
experiments/       configs, logs and result CSV/JSON
artifacts/         exported model + preprocessing + HMM config
api/               FastAPI backend (Laptop 1)
ui/                evaluator UI (Laptop 2)
reports/           figures, Folium maps, final report
data/              local dataset (not committed)
```

## AI assistance

Parts of this repository were written with AI assistance (Claude). All experiments are run by the team and every reported number is produced by committed code.
