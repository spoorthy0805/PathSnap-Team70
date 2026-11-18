# Team workflow

- **Branch per task:** `phase<N>-<short-name>` (e.g. `phase4-lightgbm`). Open a PR into `main`; one teammate reviews.
- **Commit your own work from your own account.** Evaluators look at repository history and may ask any member about any part.
- **Update `TRACKER.md`** (status, owner, notes, results) in the same PR as the work.
- **Results:** every experiment writes its config, seed, metrics and timings to `experiments/results/`. Never paste numbers into the report that a script did not produce.
- **Never commit** raw data, evaluator data, secrets, or files > 50 MB.
- **Eval data** (`data/eval_test/`) is used only by the API on Laptop 1 for final scoring — never for training, tuning or feature choices.

## Phase 10: API and UI integration

Run the API from the repository root (Laptop 1):

```bash
PATHSNAP_EVAL_DIR=data/student_test uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Use `data/eval_test` for evaluator-provided data on the evaluation machine. Keep this directory local; raw/evaluator data must not be committed. Confirm `GET /health` and `GET /trips` before starting the UI.

Run the UI (Laptop 2 or the same machine):

```bash
PATHSNAP_API=http://<laptop-1-ip>:8000 streamlit run ui/app.py --server.address 0.0.0.0
```

Run the live integration test:

```bash
python scripts/08_integration_test.py --api http://<laptop-1-ip>:8000
```

The integration test requires at least one available trip and exits non-zero if API preflight fails, a prediction fails, the response contract is invalid, or an unknown trip does not return HTTP 404.
