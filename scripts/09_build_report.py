"""Phase 11 — build the technical report (HTML -> PDF) from the committed experiment outputs.

Every table below is read from experiments/results/; every figure is a file produced by
scripts 01-08 (reports/figures, reports/screenshots). Prose is written by the team.

  python scripts/09_build_report.py            # writes reports/report.html and reports/PathSnap_Report.pdf
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "experiments" / "results"
REP = REPO / "reports"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

TEAM = "Team 70"
MEMBERS = [
    "Spoorthy M - PES1UG24AM280",
    "Praneetha Sreedharala - PES1UG24AM281",
    "Sreekar Bandi - PES1UG24AM282",
    "Sri Ram V - PES1UG24AM283"
]
REPO_URL = "https://github.com/spoorthy0805/PathSnap-Team70.git"

# ------------------------------------------------------------------ data
cmp = pd.read_csv(RES / "phase4" / "comparison.csv").set_index("model")
folds = pd.read_csv(RES / "phase4" / "cv_folds.csv")
hpo = pd.read_csv(RES / "phase5" / "tuned_vs_untuned.csv").set_index("model")
space = json.loads((RES / "phase5" / "search_space.json").read_text())
best = json.loads((RES / "phase5" / "best_params.json").read_text())
trials = pd.read_csv(RES / "phase5" / "trials.csv")
hmm = pd.read_csv(RES / "phase6" / "xgb_untuned" / "summary.csv")
per_trip = pd.read_csv(RES / "phase6" / "xgb_untuned" / "per_trip.csv", dtype={"trip_id": str})
dec = json.loads((RES / "phase6" / "xgb_untuned" / "decoder_params.json").read_text())
eda = json.loads((RES / "eda_summary.json").read_text())
feat = json.loads((RES / "feature_table_stats.json").read_text())
base = pd.read_csv(RES / "heuristic_baselines.csv")
integ = pd.read_csv(RES / "phase10" / "integration.csv", dtype={"trip_id": str})
labels = pd.read_csv(RES / "labels_trip_summary.csv", dtype={"trip_id": str})


def f4(x):
    return f"{x:.4f}"


def pc(x, d=1):
    return f"{100 * x:.{d}f}%"


def table(df: pd.DataFrame, cls: str = "") -> str:
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = "".join("<tr>" + "".join(f"<td>{v}</td>" for v in r) + "</tr>" for r in df.astype(str).values)
    return f'<table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>'


def fig(path: str, caption: str, width: int = 100) -> str:
    return (f'<figure><img src="{path}" style="width:{width}%"><figcaption>{caption} '
            f'<span class="src">[{path}]</span></figcaption></figure>')


NAMES = {"logreg": "Logistic Regression", "gnb": "Gaussian Naive Bayes", "knn": "k-Nearest Neighbours",
         "dtree": "Decision Tree", "rf": "Random Forest", "lgbm": "LightGBM", "xgb": "XGBoost",
         "mlp": "Multi-layer Perceptron"}
ORDER = ["logreg", "gnb", "knn", "dtree", "rf", "lgbm", "xgb", "mlp"]


def m(model, col):
    return cmp.loc[model, col]


# ------------------------------------------------------------------ tables
overall = pd.DataFrame([{
    "Model": NAMES[k], "Family": m(k, "family"),
    "Key configuration": ", ".join(f"{a}={b}" for a, b in json.loads(m(k, "config")).items()),
    "CV macro-F1": f"{m(k, 'cv_f1_macro_mean'):.4f} ± {m(k, 'cv_f1_macro_std'):.4f}",
    "Top-1 (test)": f4(m(k, "test_top1")),
    "Route continuity (test)": f4(m(k, "test_continuity")),
    "Macro-F1 (test)": f4(m(k, "test_f1_macro")),
    "Log loss (test)": f4(m(k, "test_log_loss")),
    "Inference ms / 1k pts": f"{m(k, 'test_predict_ms_per_1k_points'):.1f}",
} for k in ORDER])

fit_gap = pd.DataFrame([{
    "Model": NAMES[k],
    "CV train Top-1": f4(m(k, "cv_train_top1_mean")), "CV val Top-1": f4(m(k, "cv_top1_mean")),
    "Gap (pp)": f"{100 * (m(k, 'cv_train_top1_mean') - m(k, 'cv_top1_mean')):.1f}",
    "CV train log loss": f4(m(k, "cv_train_log_loss_mean")), "CV val log loss": f4(m(k, "cv_log_loss_mean")),
    "Test Top-1": f4(m(k, "test_top1")),
} for k in ORDER])

u, t = hpo.loc["xgb_untuned"], hpo.loc["xgb_tuned"]
hpo_tab = pd.DataFrame([
    {"Metric": "CV macro-F1", "Untuned": f4(u.cv_f1_macro_mean), "Tuned": f4(t.cv_f1_macro_mean)},
    {"Metric": "CV Top-1 (search objective)", "Untuned": f4(u.cv_top1_mean), "Tuned": f4(t.cv_top1_mean)},
    {"Metric": "Student-test Top-1", "Untuned": f4(u.test_top1), "Tuned": f4(t.test_top1)},
    {"Metric": "Student-test macro-F1", "Untuned": f4(u.test_f1_macro), "Tuned": f4(t.test_f1_macro)},
    {"Metric": "Student-test log loss", "Untuned": f4(u.test_log_loss), "Tuned": f4(t.test_log_loss)},
    {"Metric": "Inference ms / 1k points", "Untuned": f"{u.test_predict_ms_per_1k_points:.1f}",
     "Tuned": f"{t.test_predict_ms_per_1k_points:.1f}"},
    {"Metric": "Final fit time (s)", "Untuned": f"{u.final_fit_s:.1f}", "Tuned": f"{t.final_fit_s:.1f}"},
])
space_tab = pd.DataFrame([{"Hyperparameter": k, "Distribution": v[0].replace("float_log", "log-uniform float")
                           .replace("float", "uniform float").replace("int", "integer"),
                           "Range": f"[{v[1]}, {v[2]}]" + (f", step {v[3]}" if len(v) > 3 else ""),
                           "Best value": (f"{best['params'][k]:.4g}" if isinstance(best["params"][k], float)
                                          else best["params"][k])}
                          for k, v in space["space"].items()])

hmm_tab = hmm.assign(split=hmm.split.str.replace("_", "-"),
                     method=hmm.method.map({"classifier_only": "Classifier only",
                                            "classifier+hmm": "Classifier + HMM/Viterbi"}))
hmm_tab = pd.DataFrame({"Split": hmm_tab.split, "Method": hmm_tab.method, "Top-1": hmm_tab.top1.map(f4),
                        "Continuity": hmm_tab.continuity.map(f4), "Route precision": hmm_tab.route_precision.map(f4),
                        "Route recall": hmm_tab.route_recall.map(f4), "Route F1": hmm_tab.route_f1.map(f4),
                        "Unbridged gaps": hmm_tab.route_gaps_total})
st = per_trip[per_trip.split == "student_test"]
st_tab = pd.DataFrame({"Trip": st.trip_id, "Points": st.n_points,
                       "Top-1 clf": st.top1_classifier.map(f4), "Top-1 +HMM": st.top1_hmm.map(f4),
                       "Route F1 clf": st.route_f1_classifier.map(f4), "Route F1 +HMM": st.route_f1_hmm.map(f4),
                       "Gaps clf → HMM": st.route_gaps_classifier.astype(str) + " → " + st.route_gaps_hmm.astype(str)})
tr = per_trip[per_trip.split == "train"]
hurt = tr[tr.top1_gain < 0]
best_gain = tr.sort_values("top1_gain", ascending=False).head(3)

integ_tab = pd.DataFrame({"Trip": integ.trip_id, "HTTP": integ.status, "Points": integ.n_points,
                          "Top-1": integ.top1_accuracy.map(f4), "Route F1": integ.route_f1.map(f4),
                          "Continuity": integ.route_continuity.map(f4), "Server ms": integ.server_ms,
                          "Round trip ms": integ.round_trip_ms})
pooled_integ = (integ.top1_accuracy * integ.n_points).sum() / integ.n_points.sum()

tr_e, st_e = eda["train"], eda["student_test"]
eda_tab = pd.DataFrame([
    {"Quantity": "Trips", "Train": tr_e["trips"], "Student test": st_e["trips"]},
    {"Quantity": "GPS points", "Train": f"{tr_e['points_total']:,}", "Student test": f"{st_e['points_total']:,}"},
    {"Quantity": "Track length (km)", "Train": f"{tr_e['track_km_total']:,.0f}", "Student test": f"{st_e['track_km_total']:,.0f}"},
    {"Quantity": "Share of 1 s sampling steps", "Train": pc(tr_e["dt_frac_1s_pooled"]), "Student test": pc(st_e["dt_frac_1s_pooled"])},
    {"Quantity": "Trips with gaps > 30 s", "Train": tr_e["trips_with_gaps_gt_30s"], "Student test": st_e["trips_with_gaps_gt_30s"]},
    {"Quantity": "Trips with jumps > 70 m/s", "Train": tr_e["trips_with_jumps_gt_70ms"], "Student test": st_e["trips_with_jumps_gt_70ms"]},
    {"Quantity": "NaN values / invalid coordinates", "Train": f"{tr_e['nan_values']} / {tr_e['invalid_coords']}", "Student test": f"{st_e['nan_values']} / {st_e['invalid_coords']}"},
    {"Quantity": "Non-monotonic timestamps", "Train": tr_e["nonmonotonic_t"], "Student test": st_e["nonmonotonic_t"]},
    {"Quantity": "GPS → reference route distance, median / p95 (m)",
     "Train": f"{tr_e['gps_to_route_median_m_pooled']:.1f} / {tr_e['gps_to_route_p95_m_pooled']:.1f}",
     "Student test": f"{st_e['gps_to_route_median_m_pooled']:.1f} / {st_e['gps_to_route_p95_m_pooled']:.1f}"},
    {"Quantity": "Median arc length (m)", "Train": f"{tr_e['arc_len_median_m_pooled']:.1f}", "Student test": f"{st_e['arc_len_median_m_pooled']:.1f}"},
    {"Quantity": "Candidate rows (R = 50 m, K = 12)", "Train": f"{feat['train']['rows']:,}", "Student test": f"{feat['student_test']['rows']:,}"},
    {"Quantity": "Positive (reference-arc) rate", "Train": pc(feat["train"]["positive_rate"]), "Student test": pc(feat["student_test"]["positive_rate"])},
])
base_tab = base.assign(top1=base.top1.map(f4)).rename(columns={"split": "Split", "rule": "Rule", "top1": "Top-1"})
unrel = labels[labels.unreliable]

# ------------------------------------------------------------------ per-model prose
PROSE = {
"logreg": dict(
 why="A linear model over standardised features is the simplest probabilistic scorer: if the reference arc can be separated from the other candidates by a weighted sum of distance, rank and heading agreement, it gives calibrated probabilities that the HMM can use directly.",
 sl="Fast to train (≈0.7 s per fold) and to predict (0.1 ms per 1k points), and its probabilities are well calibrated. It cannot model interactions. For example, a large distance matters much less when the point's heading is invalid (the vehicle is stopped), and a single global weight for <code>dist</code> cannot express that.",
 hp="C = 1.0 (inverse L2 strength), max_iter = 1000, lbfgs solver; StandardScaler in front.",
 loss="L2-regularised binary cross-entropy (log loss) over candidate rows.",
 fit="It underfits: the train/validation Top-1 gap is only {gap} pp, and both sit about 1 pp below the tree ensembles. The error is bias, not variance, so more data or regularisation would not help; non-linear models do."),
"gnb": dict(
 why="Naive Bayes is the cheapest generative baseline. We included it to measure how much of the problem is solved by treating the features as independent evidence.",
 sl="Trains in 0.1 s. Its independence assumption is badly violated here: <code>dist</code>, <code>dist_minus_min</code>, <code>dist_ratio</code>, <code>rank</code> and the window distances are strongly correlated (see the feature correlation figure), so the same evidence is counted several times. That makes the probabilities over-confident.",
 hp="var_smoothing = 1e-9 (default); no other tunable parameters.",
 loss="No iterative loss; class-conditional Gaussian means and variances are estimated by maximum likelihood.",
 fit="It underfits and is badly calibrated. Train and validation Top-1 are almost equal ({tr} vs {va}), but log loss is {ll}, about 6× the other models, because the double-counted evidence pushes probabilities towards 0 and 1. This alone rules it out as an HMM emission model."),
"knn": dict(
 why="Map matching is local: a candidate whose distance, heading and neighbour-distance profile looks like previously seen correct candidates is probably correct. KNN tests that directly, without assuming a functional form.",
 sl="It needs no training, but prediction is brute-force over the stored rows. Even with only 60k training rows (a random subsample, needed because of memory and time) it is the slowest model at {ms} ms per 1k points, about 50× XGBoost. It is also sensitive to feature scaling and irrelevant dimensions.",
 hp="n_neighbors = 25, weights = distance, Euclidean metric on standardised features; trained on 60,000 sampled rows.",
 loss="None (lazy learner); the probability is the distance-weighted vote of the 25 neighbours.",
 fit="Mild variance: train Top-1 {tr} vs validation {va}, and validation log loss ({ll}) is noticeably worse than train ({trll}). With distance weighting each training row is its own nearest neighbour, which inflates the training scores. On held-out data it is below logistic regression on both Top-1 and log loss."),
"dtree": dict(
 why="A single tree learns threshold rules similar to those a human would write (\"nearest arc unless heading disagrees\"), so it is interpretable and captures the interactions the linear model misses.",
 sl="Interpretable and fast at inference (0.2 ms per 1k points). A single tree is high-variance and its probabilities are piecewise constant (leaf frequencies), which gives coarse HMM emissions.",
 hp="max_depth = 12, min_samples_leaf = 50, Gini criterion.",
 loss="Greedy minimisation of Gini impurity at each split.",
 fit="Mild overfitting: train Top-1 {tr} vs validation {va} (gap {gap} pp), and train log loss {trll} vs validation {ll}. The depth and leaf-size limits keep it from memorising, but it still trails the ensembles."),
"rf": dict(
 why="Averaging many decorrelated trees keeps the decision-tree interactions while removing most of their variance; it also gives smoother probabilities than one tree.",
 sl="Robust and needs little tuning. Each tree sees only half the rows (max_samples = 0.5) and √26 ≈ 5 features per split. Fit time is the largest of the tree models (≈25 s per fold), and the model file is big.",
 hp="n_estimators = 100, min_samples_leaf = 5, max_features = sqrt, max_samples = 0.5.",
 loss="Each tree minimises Gini impurity; the forest averages the leaf probabilities.",
 fit="It overfits the training trips (train Top-1 {tr} vs validation {va}, gap {gap} pp; train log loss {trll} vs {ll}), but validation performance is still among the best. The gap is the usual ensemble memorisation of training rows rather than a loss of generalisation."),
"lgbm": dict(
 why="Gradient-boosted trees fit the residual errors of the previous trees, which suits the many weak but complementary signals here (distance, heading, neighbour consistency, arc degree).",
 sl="Strong accuracy and good calibration. The leaf-wise growth (num_leaves = 63) overfits easily. Its averaged validation log loss is lowest at round 153 (0.082) and then rises to 0.090 by round 400 (see the loss curves), so it would need early stopping.",
 hp="n_estimators = 400, learning_rate = 0.05, num_leaves = 63, subsample = 0.8 (every iteration), colsample_bytree = 0.8.",
 loss="Binary log loss, minimised by gradient boosting with second-order (Newton) leaf values.",
 fit="It overfits slightly: train Top-1 {tr} vs validation {va} (gap {gap} pp), and the per-round validation log loss bottoms out and then creeps up while the training loss keeps falling. Test Top-1 ({te}) is still second best."),
"xgb": dict(
 why="Same reasoning as LightGBM, but depth-wise trees with a fixed maximum depth and explicit L2 regularisation on the leaf weights give smoother, better-calibrated probabilities. That matters because these probabilities become the HMM emission scores.",
 sl="Best Top-1, macro-F1 and log loss of all eight models, with fast histogram-based training and only {ms} ms per 1k points at inference. Like every candidate-level classifier, it scores each GPS point independently and cannot enforce route connectivity, which is why the HMM is still needed.",
 hp="n_estimators = 400, learning_rate = 0.05, max_depth = 7, subsample = 0.8, colsample_bytree = 0.8, tree_method = hist, λ = 1 (default).",
 loss="Binary log loss (<code>binary:logistic</code>), minimised by second-order gradient boosting with L2-regularised leaf weights.",
 fit="Moderate train/validation gap (train Top-1 {tr} vs validation {va}, {gap} pp). The averaged validation log loss reaches its minimum (0.0815) at round 171 and is essentially flat afterwards (0.0824 at round 400) while the training loss keeps falling to 0.052, so the extra rounds add a little overfitting but do not hurt held-out accuracy. Student-test Top-1 ({te}) is higher than its CV estimate, which shows it generalises to unseen trips."),
"mlp": dict(
 why="A small neural network can learn smooth non-linear decision boundaries over all features jointly, and its training loss can be monitored epoch by epoch.",
 sl="Smooth probabilities and the highest classifier-only route continuity ({cont}). It needs feature scaling and careful learning-rate and epoch choices, and on tabular data of this size it does not beat boosted trees.",
 hp="Hidden layers (64, 32), ReLU, Adam, learning rate 1e-3, L2 α = 1e-4, batch 1024, 20 epochs (partial_fit loop recording train and validation loss each epoch).",
 loss="Binary cross-entropy with an L2 penalty, minimised by Adam.",
 fit="Mild overfitting late in training: train Top-1 {tr} vs validation {va} (gap {gap} pp). The averaged validation loss is lowest at epoch 4 (0.0876) and then drifts up to 0.0898 at epoch 20 while the training loss keeps falling, so early stopping around epoch 4–5 would have been slightly better."),
}


def model_section(k: str, i: int) -> str:
    p = PROSE[k]
    ctx = dict(gap=f"{100 * (m(k, 'cv_train_top1_mean') - m(k, 'cv_top1_mean')):.1f}",
               tr=f4(m(k, "cv_train_top1_mean")), va=f4(m(k, "cv_top1_mean")), ll=f4(m(k, "cv_log_loss_mean")),
               trll=f4(m(k, "cv_train_log_loss_mean")), te=f4(m(k, "test_top1")),
               ms=f"{m(k, 'test_predict_ms_per_1k_points'):.1f}", cont=f4(m(k, "test_continuity")))
    res = (f"On the 10 student-test trips (147k candidate rows, 19,160 GPS points): observation Top-1 "
           f"<b>{f4(m(k, 'test_top1'))}</b>, candidate macro-F1 <b>{f4(m(k, 'test_f1_macro'))}</b> "
           f"(precision {f4(m(k, 'test_precision_macro'))}, recall {f4(m(k, 'test_recall_macro'))}), log loss "
           f"<b>{f4(m(k, 'test_log_loss'))}</b>, route continuity {f4(m(k, 'test_continuity'))}. In 5-fold CV: "
           f"Top-1 {f4(m(k, 'cv_top1_mean'))} ± {f4(m(k, 'cv_top1_std'))}, macro-F1 {f4(m(k, 'cv_f1_macro_mean'))}.")
    return f"""
<h3>3.2.{i} {NAMES[k]} ({m(k, 'family')})</h3>
<p class="q">A1. Why suitable?</p><p>{p['why']}</p>
<p class="q">A2. Strengths and limitations</p><p>{p['sl'].format(**ctx)}</p>
<p class="q">B5. Important hyperparameters (initial values)</p><p>{p['hp']}</p>
<p class="q">B6. Loss / objective</p><p>{p['loss']}</p>
<p class="q">C7–C8. Student-test performance</p><p>{res}</p>
<p class="q">C9. Over- or underfitting?</p><p>{p['fit'].format(**ctx)}</p>"""


# ------------------------------------------------------------------ html
CSS = """
@page {
    size: A4;
    margin: 18mm 17mm 18mm 17mm;
}

body {
    font-family: Arial, Helvetica, sans-serif;
    font-size: 10pt;
    line-height: 1.6;
    color: #263238;
    background: #ffffff;
}

h1 {
    font-size: 28pt;
    font-weight: 700;
    color: #123c40;
    margin-bottom: 14px;
    letter-spacing: 0.5px;
}

h2 {
    font-size: 15pt;
    color: #126b69;
    border-bottom: 1.5px solid #37a6a0;
    padding-bottom: 7px;
    margin-top: 28px;
    margin-bottom: 13px;
    page-break-after: avoid;
}

h3 {
    font-size: 12pt;
    color: #254e58;
    margin-top: 21px;
    margin-bottom: 8px;
    page-break-after: avoid;
}

h4 {
    font-size: 10.5pt;
    color: #34656d;
    margin-top: 15px;
    margin-bottom: 7px;
    page-break-after: avoid;
}

p {
    margin: 7px 0 11px;
    text-align: justify;
}

ul, ol {
    padding-left: 22px;
    margin: 9px 0 14px;
}

li {
    margin-bottom: 6px;
}

table {
    border-collapse: collapse;
    width: 100%;
    margin: 15px 0 20px;
    font-size: 8.2pt;
    page-break-inside: avoid;
}

th {
    background: #176b68;
    color: #ffffff;
    font-weight: 600;
    text-align: left;
    padding: 9px 8px;
    border: 1px solid #176b68;
}

td {
    padding: 8px;
    border: 1px solid #d5e3e2;
    text-align: left;
    vertical-align: top;
}

tbody tr:nth-child(even) {
    background: #f0f7f6;
}

tbody tr:nth-child(odd) {
    background: #ffffff;
}

table.wide {
    font-size: 7.3pt;
}

table.wide th,
table.wide td {
    padding: 5px;
}

table.wide td:nth-child(3) {
    font-size: 7pt;
    max-width: 155px;
}

figure {
    margin: 20px 0 23px;
    text-align: center;
    page-break-inside: avoid;
}

figure img {
    max-width: 100%;
    height: auto;
    border: 1px solid #dce8e6;
    border-radius: 5px;
    padding: 4px;
}

figcaption {
    font-size: 8.5pt;
    color: #52666a;
    margin-top: 10px;
    line-height: 1.4;
    text-align: center;
}

.src {
    color: #839596;
    font-family: Consolas, monospace;
    font-size: 7.2pt;
}

p.q {
    font-size: 10pt;
    font-weight: 700;
    color: #126b69;
    margin-top: 14px;
    margin-bottom: 4px;
    text-align: left;
}

code {
    font-family: Consolas, monospace;
    font-size: 8.5pt;
    color: #22565a;
    background: #edf5f4;
    padding: 2px 4px;
    border-radius: 3px;
    overflow-wrap: anywhere;
}

.cover {
    text-align: center;
    padding-top: 105px;
    page-break-after: always;
}

.cover h1 {
    font-size: 38pt;
    color: #115e59;
    letter-spacing: 2px;
    margin-bottom: 20px;
}

.cover p {
    font-size: 12pt;
    line-height: 1.9;
    text-align: center;
    color: #425b60;
}

.cover .box {
    margin-top: 65px;
    text-align: left;
}

.box {
    background: #eef7f5;
    border-left: 4px solid #16847d;
    padding: 13px 17px;
    margin: 16px 0;
    border-radius: 3px;
    color: #263f43;
}

.pb {
    page-break-before: always;
}

.two {
    display: flex;
    gap: 18px;
    align-items: flex-start;
}

.two figure {
    flex: 1;
    min-width: 0;
}

a {
    color: #087e79;
    text-decoration: none;
}

strong, b {
    color: #183f43;
}

@media print {
    body {
        background: white;
    }

    h2, h3, h4 {
        break-after: avoid;
    }

    figure {
        break-inside: avoid;
    }

    tr {
        break-inside: avoid;
    }
}
"""

hurt_list = ", ".join(f"{r.trip_id} ({100 * r.top1_gain:+.1f} pp)" for r in hurt.itertuples())
gain_list = ", ".join(f"{r.trip_id} ({100 * r.top1_gain:+.1f} pp)" for r in best_gain.itertuples())
test_clf, test_hmm = hmm.set_index(["split", "method"]).loc["student_test"].loc["classifier_only"], \
    hmm.set_index(["split", "method"]).loc["student_test"].loc["classifier+hmm"]

html = f"""<!doctype html><html><head><meta charset="utf-8"><title>PathSnap Technical Report</title>
<style>{CSS}</style></head><body>

<div class="cover">
<h1>PathSnap</h1>
<p style="font-size:15pt">Distributed map-matching and route reconstruction</p>
<p>UE24CS352A Machine Learning · Hackathon technical report</p>
<p style="margin-top:40px"><b>{TEAM}</b><br>{'<br>'.join(MEMBERS)}</p>
<p style="margin-top:30px">Repository: <a href="{REPO_URL}">{REPO_URL}</a></p>
<div class="box" style="text-align:left;margin:50px auto 0;max-width:520px">
<b>Final system:</b> candidate generation (R = 50 m, K = 12) → XGBoost candidate scorer (26 features) →
HMM/Viterbi decoding over the directed road graph.<br>
<b>Student-test results:</b> observation Top-1 <b>{f4(test_hmm.top1)}</b>, route F1 <b>{f4(test_hmm.route_f1)}</b>,
route continuity {f4(test_hmm.continuity)}; 33–513 ms per trip on a laptop CPU.
</div></div>

<h2>1. Problem and approach</h2>
<p>Each trip consists of a ~1 Hz GPS track (<code>.track</code>), the local road network (<code>.nodes</code>,
directed <code>.arcs</code>) and, for training, the reference route as an ordered list of arc ids (<code>.route</code>).
The task is to assign each GPS point to the road arc the vehicle was actually on, and to output a connected route.</p>
<p>Arc ids are local to each trip, so there is no fixed label space for a multi-class classifier. We therefore treat
map matching as <b>binary candidate classification</b>. For each GPS point, the arcs within 50 m (at most the 12
nearest) are candidates. A classifier scores P(candidate is the true arc) from 26 features, and the observation
prediction is the highest-scoring candidate (Top-1). These probabilities then become the emission scores of an HMM
whose transitions come from directed shortest-path distances in the road graph. Viterbi decoding returns the most
likely connected arc sequence.</p>
<p><b>Data partitions used throughout:</b> <i>train</i> (80 trips) for fitting, 5-fold GroupKFold cross-validation
grouped by trip, Bayesian optimisation and HMM-parameter selection; <i>student test</i> (10 trips) only for
reporting and model comparison; the 10 evaluator trips were never available to us and are only read by the API on
Laptop 1 on evaluation day.</p>

<h2>2. Dataset understanding and exploratory data analysis</h2>
<h3>2.1 Preprocessing</h3>
<p><b>Loading and validation</b> (<code>src/pathsnap/io.py</code>, <code>scripts/01_eda.py</code>). Every file was parsed and
checked for NaNs, out-of-range coordinates, non-monotonic timestamps, dangling arc endpoints and isolated nodes.
None were found (table 2.1), so no rows were imputed or dropped at load time.</p>
<p><b>Projection.</b> Trips span several countries, so each trip is projected to a local azimuthal-equidistant
plane centred on the trip (<code>pyproj</code>, <code>geo.py</code>). This lets all distances and bearings be computed in
metres with plain Euclidean geometry.</p>
<p><b>Per-point labels.</b> The reference route is a sequence of arcs, not one label per GPS point. We derive
per-point labels with a <b>monotone dynamic-programming alignment</b> of the track to the route positions
(<code>labels.py</code>): each point gets one route arc, positions may not go backwards, and the total point-to-arc
distance is minimised.</p>
<p><b>Unreliable trips.</b> Two training trips have a median point-to-label distance above 50 m:
{', '.join(f"{r.trip_id} ({r.label_dist_median_m:.0f} m)" for r in unrel.itertuples())}. In 00000054 the whole GPS
track is shifted about 130 m away from every road; in 00000033 the reference route runs along a parallel road about
120 m away. Their labels cannot be trusted, so these trips are <b>excluded from training</b> (78 reliable trips
remain). They stay in the EDA tables.</p>
<p><b>Candidate generation</b> (<code>candidates.py</code>). Arcs are cut into ≤ 20 m pieces and indexed in a KD-tree. For
each point we take all arcs within R metres, keeping the K nearest. A grid search over R and K
(<code>candidate_miss_rate.csv</code>) chose <b>R = 50 m, K = 12</b>. With these, the true arc is missing from the
candidate set for 1.57% of reliable training points and 0.21% of student-test points, with about 7 candidates per
point. Raising R to 75 m would save only about 0.5 pp of misses for 20% more candidates.</p>

<h3>2.2 Missing values and data quality</h3>
{table(eda_tab)}
<p class="src">Table 2.1 — experiments/results/eda_summary.json, feature_table_stats.json</p>
{fig("figures/eda_missing_values.png", "Figure 2.1 — Missing-value and validity analysis per partition (train + student test).", 80)}

<h3>2.3 Patterns observed</h3>
<ul>
<li><b>Sampling:</b> {pc(tr_e['dt_frac_1s_pooled'])} of training steps are exactly 1 s apart, but 28 training trips have
gaps over 30 s (tunnels, dropouts). This motivated the <code>dt_prev</code>/<code>dt_next</code>/<code>gap_flag</code>
features and the gap rule in the HMM.</li>
<li><b>GPS noise is small but not negligible:</b> the median distance to the reference route is
{tr_e['gps_to_route_median_m_pooled']:.1f} m, but the 95th percentile is {tr_e['gps_to_route_p95_m_pooled']:.1f} m. That
is wider than the spacing of parallel roads and short arcs (median arc {tr_e['arc_len_median_m_pooled']:.0f} m), so
"nearest arc" is often wrong.</li>
<li><b>Teleports:</b> 10 training trips contain jumps faster than 70 m/s (e.g. 00000013 has 95 such steps). These are
GPS outliers that a point-wise matcher follows and a route-level decoder can ignore.</li>
<li><b>Heading matters:</b> the nearest-arc rule gets only {f4(base.query("split=='student_test'").top1.iloc[0])} Top-1 on
student test, while adding a heading-agreement check raises it to {f4(base.query("split=='student_test'").top1.iloc[1])}
(table 2.2). Two-way streets are stored as two opposite arcs that are equally close, and only the direction of motion
separates them.</li>
<li><b>Networks are fragmented:</b> almost every trip's network has several weakly connected components (it is clipped
to a box around the trip). The HMM therefore has to cope with unreachable transitions.</li>
</ul>
{table(base_tab)}
<p class="src">Table 2.2 — rule-based baselines, experiments/results/heuristic_baselines.csv</p>
<div class="two">
{fig("figures/eda_sampling_speed_noise.png", "Figure 2.2 — Sampling interval, speed and GPS-to-route distance distributions (train vs student test).")}
{fig("figures/eda_trip_sizes.png", "Figure 2.3 — Trip sizes (points, km) per partition.")}
</div>
<div class="two">
{fig("figures/eda_trip_locations.png", "Figure 2.4 — GPS trajectory start locations of all 90 trips (train and student test).")}
{fig("figures/eda_network.png", "Figure 2.5 — Road-network visualisation and network statistics (arc length, node degree, components).")}
</div>
{fig("screenshots/map_labels_trip_00000095.jpg", "Figure 2.6 — Folium map (train trip 00000095): GPS observations, road network near the track, and the reference route with the per-point label alignment. Interactive version: reports/maps/labels_trip_00000095.html; EDA maps for 5 trips in reports/maps/eda_trip_*.html (scripts/01_eda.py, src/pathsnap/viz.py).", 92)}
{fig("figures/labels_dist_and_rank.png", "Figure 2.7 — Distance from each GPS point to its labelled arc, and the rank of the labelled arc among the candidates (train). The true arc is the nearest candidate only about half the time.", 85)}
{fig("figures/cand_miss_rate_heatmap.png", "Figure 2.8 — Candidate miss rate over radius R and K (reliable train trips), used to choose R = 50 m, K = 12.", 75)}

<h3>2.4 Features extracted and why</h3>
<p>Each candidate row describes one (GPS point, candidate arc) pair. Raw longitude, latitude, timestamps and ids are
deliberately <b>not</b> used, so the model cannot memorise locations of specific trips. All 26 features are computed
from the track and the network only (<code>features.py</code>):</p>
<table><tr><th>Group</th><th>Features</th><th>Rationale (observed in EDA)</th></tr>
<tr><td>Geometry</td><td>dist, rank, dist_minus_min, dist_ratio, frac, at_endpoint</td><td>Closeness is the main signal, but relative to the other candidates (dist_minus_min has the strongest univariate separation, AUC strength 0.82).</td></tr>
<tr><td>Candidate context</td><td>n_cands, n_within_radius, min_dist, second_gap</td><td>The same distance means different things in a dense junction and on an isolated highway.</td></tr>
<tr><td>Arc properties</td><td>arc_len_log, two_way, out_deg_end, in_deg_start</td><td>Short connector arcs and two-way pairs cause most confusions.</td></tr>
<tr><td>Direction</td><td>heading_cos, heading_diff, heading_cos_wide, heading_valid</td><td>Separates the two directions of a two-way street (baseline jump 0.54 → 0.83).</td></tr>
<tr><td>Motion</td><td>speed, dt_prev, dt_next, gap_flag</td><td>Heading is meaningless when stopped; gaps make neighbours unreliable.</td></tr>
<tr><td>Temporal consistency</td><td>dist_prev1, dist_next1, dist_win5_mean, dist_win5_max</td><td>The true arc is also close to neighbouring points; a crossing street is not.</td></tr></table>
<div class="two">
{fig("figures/feat_univariate_auc.png", "Figure 2.9 — Univariate ROC-AUC strength of each feature (train).")}
{fig("figures/feat_correlation.png", "Figure 2.10 — Feature correlation matrix (train). Strong correlation within the distance group.")}
</div>
{fig("figures/feat_class_distributions.png", "Figure 2.11 — Feature distributions for true vs false candidates (train).", 90)}

<h2 class="pb">3. Model experimentation and comparison</h2>
<p><b>Protocol.</b> Eight models from seven algorithm families were trained on the same 26-feature candidate table
(<code>scripts/04_model_experiments.py</code>, <code>src/pathsnap/models.py</code>). The setup was identical for all:</p>
<ul>
<li>5-fold <b>GroupKFold by trip</b> (shuffled, seed 42) over the 78 reliable training trips, so no trip appears in
both training and validation.</li>
<li>After CV, a final fit on all 78 trips, evaluated once on the 10 student-test trips.</li>
</ul>
<p><b>Metrics.</b></p>
<ul>
<li><b>Observation Top-1:</b> the share of <i>all</i> GPS points whose highest-scoring candidate is the labelled arc.
A point with no candidate counts as wrong.</li>
<li><b>Macro-F1:</b> the candidate-level binary classification at threshold 0.5, averaged over both classes.</li>
<li><b>Log loss:</b> of the candidate probabilities.</li>
<li><b>Route continuity:</b> the share of consecutive predicted arcs that are equal or connected in the graph.</li>
<li><b>Inference time:</b> in ms per 1,000 GPS points.</li>
</ul>
<p>The selection rule was fixed in advance: Top-1 first (it is what the HMM consumes), then macro-F1 and log loss as
tie-breakers.</p>

<h3>3.1 Overall model comparison</h3>
{table(overall, "wide")}
<p class="src">Table 3.1 — experiments/results/phase4/comparison.csv. "Top-1", "Macro-F1", "Log loss", "Route continuity" are student-test values (classifier only, no HMM); CV values are mean ± std over 5 folds. All metrics apply to all models (each outputs probabilities), so there are no N/A cells.</p>
<div class="box"><b>Selection.</b> XGBoost is best or tied-best on every quality metric: CV macro-F1 {f4(m('xgb','cv_f1_macro_mean'))},
CV Top-1 {f4(m('xgb','cv_top1_mean'))}, test Top-1 {f4(m('xgb','test_top1'))}, and the lowest log loss (CV
{f4(m('xgb','cv_log_loss_mean'))}, test {f4(m('xgb','test_log_loss'))}). LightGBM and Random Forest are within one standard
deviation on CV Top-1, so the decision was driven by:
<ol>
<li>the lowest log loss, because better-calibrated probabilities make better HMM emissions;</li>
<li>a stabler validation loss: XGBoost's averaged validation log loss bottoms at round 171 and rises only 0.0009 by round 400, while LightGBM's rises from 0.082 (round 153) to 0.090;</li>
<li>4.4× faster inference than LightGBM ({m('xgb','test_predict_ms_per_1k_points'):.1f} vs {m('lgbm','test_predict_ms_per_1k_points'):.1f} ms per 1k points).</li>
</ol>
Every learned model beats the best rule-based baseline (0.8261).</div>

<h4>Train vs validation (evidence for over/underfitting)</h4>
{table(fit_gap)}
<p class="src">Table 3.2 — experiments/results/phase4/comparison.csv (cv_train_* computed on each fold's training trips).</p>

<h3>3.2 Per-model questions</h3>
{''.join(model_section(k, i + 1) for i, k in enumerate(ORDER))}

<h3 class="pb">3.3 Model visualisations</h3>
{fig("figures/models_cv_f1.png", "Figure 3.1 — Cross-validation macro-F1 per model (mean ± std over 5 trip-grouped folds).", 85)}
{fig("figures/models_test_top1_f1.png", "Figure 3.2 — Student-test observation Top-1 and macro-F1 per model.", 85)}
{fig("figures/models_logloss.png", "Figure 3.3 — Log loss (CV and student test) per model.", 80)}
{fig("figures/models_train_vs_val.png", "Figure 3.4 — Train vs validation Top-1 per model (fit gap).", 80)}
{fig("figures/models_fold_stability.png", "Figure 3.5 — Per-fold validation Top-1: model ranking is consistent across folds.", 80)}
{fig("figures/models_loss_curves.png", "Figure 3.6 — Training and validation loss curves, shown only for models that produce epoch/round-wise histories (XGBoost, LightGBM per boosting round; MLP per epoch). No curves are drawn for the other models.", 95)}
{fig("figures/models_confusion.png", "Figure 3.7 — Candidate-level confusion matrices on student test (threshold 0.5).", 95)}
{fig("figures/models_timing.png", "Figure 3.8 — Fit time per fold and inference time per 1k GPS points.", 80)}

<h2 class="pb">4. Bayesian hyperparameter optimisation (XGBoost)</h2>
<p><b>Setup</b> (<code>scripts/05_bayesian_hpo.py</code>):</p>
<ul>
<li>Optuna with a TPE sampler (seed 42, 10 random start-up trials) and a median pruner.</li>
<li><b>{int(u.n_trials_total)} trials</b>, about {u.search_wallclock_s_this_run / 60:.0f} minutes of wall time.</li>
<li>Trial 0 was the untuned Phase 4 configuration, enqueued so the search starts from the baseline.</li>
</ul>
<p><b>Objective:</b> {space['objective']}. Student test was <b>not</b> used during the search; it was evaluated
once afterwards.</p>
{table(space_tab)}
<p class="src">Table 4.1 — search space (experiments/results/phase5/search_space.json) and best trial #{best['trial']} (best_params.json). Full trial log: trials.csv, optuna_xgb.db.</p>
{table(hpo_tab)}
<p class="src">Table 4.2 — untuned vs tuned XGBoost on identical folds and partitions (experiments/results/phase5/tuned_vs_untuned.csv).</p>
<p><b>Did tuning help? Not meaningfully.</b> The best trial raised CV Top-1 by
{100 * (t.cv_top1_mean - u.cv_top1_mean):.2f} pp, which is far below the fold-to-fold standard deviation
({100 * u.cv_top1_std:.1f} pp). On student test, the tuned model is equal or very slightly worse on Top-1, macro-F1 and
log loss. It also needs 1000 trees instead of 400, roughly doubling fit and inference time.</p>
<p>The parameter-importance analysis puts learning rate, subsample and min_child_weight first. The history plot shows that the
baseline (trial 0) stayed the best trial for the first 12 trials, and the running best then crept up by less than 0.1 pp in total. Our reading is that the classifier is limited by the
information in the features (genuinely ambiguous parallel or overlapping arcs), not by its capacity. We kept the
<b>untuned</b> configuration, and Phase 6 confirmed this: after HMM decoding, the untuned model is also slightly better
(student-test Top-1 0.9552 vs 0.9528).</p>
<div class="two">
{fig("figures/hpo_history.png", "Figure 4.1 — Optimisation history (CV Top-1 per trial and running best).")}
{fig("figures/hpo_param_importance.png", "Figure 4.2 — Hyperparameter importance (fANOVA).")}
</div>
{fig("figures/hpo_tuned_vs_untuned.png", "Figure 4.3 — Untuned vs tuned XGBoost on CV and student test.", 80)}

<h2 class="pb">5. HMM and Viterbi route reconstruction</h2>
<p>Implemented in <code>src/pathsnap/hmm.py</code> and evaluated by <code>scripts/06_hmm_viterbi.py</code>.</p>
<p class="q">1. Hidden states</p><p>The candidate arcs of each GPS point, the same up-to-12 arcs within 50 m that the
classifier scored. Point t has its own state set S<sub>t</sub>.</p>
<p class="q">2. Emission scores</p><p>log e<sub>t</sub>(s) = α · log P<sub>XGB</sub>(s | features), with P clipped at
1e-6. The classifier's probability already combines distance, heading and neighbour consistency, so we do not add a
separate Gaussian distance term. α = {dec['alpha']} balances emission against transition.</p>
<p class="q">3. Transition scores</p><p>We use the Newson–Krumm idea: the road distance between consecutive matched
positions should be close to the straight-line GPS distance. For candidates s at t and s' at t+1:</p>
<ul>
<li>d<sub>route</sub> is the <i>directed</i> shortest-path distance from the projected position on s to that on s',
found with Dijkstra in the arc graph, bounded at {dec['route_limit_m']:.0f} m and cached per trip.</li>
<li>d<sub>gps</sub> is the straight-line distance between the two GPS points in the trip's local metric projection.</li>
<li>log a(s → s') = −|d<sub>route</sub> − d<sub>gps</sub>| / β with β = {dec['beta']}.</li>
</ul>
<p>Staying on the same arc moving forwards costs nothing extra. Driving the wrong way along a one-way arc would need a
long detour, so it is penalised.</p>
<p class="q">4. Missing candidates and large gaps</p><p>Points with no candidate within 50 m are skipped by the decoder
and left unmatched. They count as errors in Top-1, which keeps the metric honest. If consecutive points are more than
gap_s = {dec['gap_s']:.0f} s apart, the transition is replaced by a uniform one. This splits the chain, because after a
long dropout the vehicle may have driven anywhere.</p>
<p class="q">5. Disconnected consecutive segments</p><p>If no candidate pair is reachable within the bound (for example
the network is clipped, or a GPS jump), every transition gets a fixed log penalty of −{dec['unreachable_penalty']:.0f}
instead of −∞. Viterbi therefore never dead-ends and simply chooses the best emission. When the route is rebuilt, each
pair of consecutive decoded arcs is joined by the shortest directed path between them. Pairs that still cannot be
joined are counted as unbridged gaps.</p>
<p class="q">Parameter selection</p><p>The decoder settings were chosen on <b>out-of-fold training predictions</b> (the 78 trips, each scored by the CV
model that had not seen it), maximising Top-1, in two stages. First α ∈ {{0.5, 1, 2}} and β ∈ {{2, 5, 10, 25}} m
were searched at gap_s = 30 s; then gap_s ∈ {{10, 30, 120}} s was searched at the best (α, β). Student test was not
used. The best β (2 m) is at the edge of the grid, so a smaller β might be slightly better; we did not search
further.</p>
<p class="q">6. Did the HMM improve the route?</p><p><b>Yes, substantially</b>:</p>
{table(hmm_tab)}
<p class="src">Table 5.1 — experiments/results/phase6/xgb_untuned/summary.csv</p>
<ul>
<li><b>Student test:</b> Top-1 {f4(test_clf.top1)} → <b>{f4(test_hmm.top1)}</b> (+{100 * (test_hmm.top1 - test_clf.top1):.1f} pp);
route F1 {f4(test_clf.route_f1)} → <b>{f4(test_hmm.route_f1)}</b>; unbridged gaps {int(test_clf.route_gaps_total)} → {int(test_hmm.route_gaps_total)}.</li>
<li><b>Training trips (out-of-fold):</b> the gain is similar.</li>
<li><b>Where the gain comes from:</b> mostly <i>route precision</i> (0.81 → 0.98). Classifier-only routes contain short
spurious detours onto side streets and opposite-direction arcs. The classifier picks those for one or two noisy points
and then has to bridge back, and the HMM removes them.</li>
</ul>
<p><b>Where the HMM did not help.</b> On 4 of 78 training trips Top-1 dropped slightly: {hurt_list}. The losses are at most 1.1 pp. A likely
explanation, which we have not verified trip by trip, is that the transition model prefers the
shorter consistent path, so real but unusual manoeuvres, such as a quick U-turn or a loop over short arcs, can be
smoothed away. The largest gains were {gain_list}. In these trips the classifier alternates between nearby candidate
arcs, and the HMM forces one consistent choice.</p>
{table(st_tab)}
<p class="src">Table 5.2 — per-trip student-test results, experiments/results/phase6/xgb_untuned/per_trip.csv</p>
{fig("figures/hmm_xgb_untuned_vs_classifier.png", "Figure 5.1 — Classifier-only vs classifier + HMM on train (OOF) and student test.", 85)}
{fig("figures/hmm_xgb_untuned_per_trip.png", "Figure 5.2 — Per-trip Top-1 change from HMM decoding (points below zero are the trips where the HMM hurt).", 85)}
{fig("figures/hmm_xgb_untuned_decoder_grid.png", "Figure 5.3 — Two-stage decoder grid search on train OOF predictions: α × β, then gap_s.", 80)}

<h2 class="pb">6. Geographical visualisation (Folium)</h2>
<p>All maps are produced by <code>src/pathsnap/viz.py::trip_map</code> (called from <code>scripts/01_eda.py</code>,
<code>02_labels_candidates.py</code> and <code>06_hmm_viterbi.py</code>), and live in <code>reports/maps/*.html</code>.
Each map has:</p>
<ul>
<li>separate toggleable layers in a layer control: road network near the track (grey), reference route (green),
classifier-only route (orange), HMM route (red), GPS observations (blue dots);</li>
<li>a legend and start/end markers;</li>
<li>an extent fitted to the track.</li>
</ul>
<p>The evaluator UI draws the same kind of map from the API response, but without a reference layer, because the
reference never leaves the server.</p>
{fig("screenshots/map_hmm_trip_00000077.jpg", "Figure 6.1 — Student-test trip 00000077 (Syktyvkar): GPS observations, road network, reference route and HMM route. This is the trip with the largest HMM gain on student test (Top-1 0.775 → 0.902). Interactive: reports/maps/hmm_trip_00000077.html.", 95)}
<p>Figure 7.1 shows the same kind of Folium map as the evaluator UI draws it from an API response.</p>

<h2 class="pb">7. API, UI and deployment</h2>
<p><b>Laptop 1, API</b> (<code>api/main.py</code>, FastAPI + uvicorn). At start-up it loads <code>artifacts/xgb_final.json</code>
and <code>artifacts/pipeline_config.json</code>, which are exported by <code>scripts/07_export_artifacts.py</code> and hold the
features, candidate settings and decoder parameters. Its endpoints:</p>
<ul>
<li><code>GET /health</code>: model status and decoder settings.</li>
<li><code>GET /trips</code>: the trip ids in the evaluation folder (ids only).</li>
<li><code>POST /predict {{"trip_id"}}</code>: runs the full pipeline, which reads only <code>.track/.nodes/.arcs</code>. After
prediction it reads the <code>.route</code> server-side to compute metrics, and returns the matched route polylines,
downsampled GPS points, metrics and timings. The reference route is never returned.</li>
</ul>
<p>Every request is logged to the console with client IP, status, Top-1, route F1 and latency.</p>
<p><b>Laptop 2, UI</b> (<code>ui/app.py</code>, Streamlit + streamlit-folium). It talks to the API only over HTTP and
reads no trip files:</p>
<ul>
<li>It lists the trips returned by <code>/trips</code>.</li>
<li>It sends the selected id to <code>/predict</code> and shows the metrics and a Folium map.</li>
<li>An "All trips" tab runs all ten trips in turn.</li>
</ul>
<p><b>Deployment</b> (see README):</p>
<ul>
<li>Laptop 1: <code>PATHSNAP_EVAL_DIR=data/eval_test uvicorn api.main:app --host 0.0.0.0 --port 8000</code></li>
<li>Laptop 2: <code>PATHSNAP_API=http://&lt;laptop-1-ip&gt;:8000 streamlit run ui/app.py</code></li>
</ul>
<p><b>Integration test</b> (<code>scripts/08_integration_test.py</code>). The student-test trips stood in for the evaluation
trips. All <b>{int((integ.status == 200).sum())}/{len(integ)}</b> trips succeeded over HTTP, with pooled Top-1
{f4(pooled_integ)}, identical to the offline result. An unknown trip id returns HTTP 404, and no ground truth appears
in any response.</p>
{table(integ_tab)}
<p class="src">Table 7.1 — experiments/results/phase10/integration.csv</p>
<p><b>Two-laptop run.</b> We then ran the real setup on two separate laptops connected over Tailscale (a private VPN
that gives each laptop a fixed address, so the demo does not depend on the venue Wi-Fi):</p>
<ul>
<li>Laptop 1 (macOS) ran the API at <code>http://100.87.111.15:8000</code>.</li>
<li>Laptop 2 ran the UI from a fresh clone of the repository, pointed at that address.</li>
</ul>
<p>From Laptop 2, single trips and the "Evaluate all trips" run all succeeded, 10/10 trips, with pooled Top-1 95.52%
and mean route F1 0.9905. These are the same per-trip numbers as Table 7.1. Laptop 1's console logged every request
with Laptop 2's address (100.124.152.77), as shown in Figure 7.4.</p>
{fig("screenshots/laptop2_ui_trip_00000000.jpg", "Figure 7.1 — Evaluator UI running on Laptop 2, connected to the API on Laptop 1: single-trip view of trip 00000000 with server-side metrics, timings and the predicted route.", 95)}
{fig("screenshots/laptop2_ui_trip_00000052.jpg", "Figure 7.2 — Laptop 2, trip 00000052: Folium map with GPS observations, predicted route, layer control and legend.", 95)}
{fig("screenshots/laptop2_ui_all_trips.jpg", "Figure 7.3 — Laptop 2, all-trips view: 10/10 trips succeeded, pooled Top-1 95.52%, mean route F1 99.05%; the decoder settings received from /health are shown in the sidebar.", 95)}
{fig("screenshots/laptop1_api_log_two_laptops.jpg", "Figure 7.4 — Laptop 1 API console during the all-trips run from Laptop 2: every /predict request from 100.124.152.77 with the trip's metrics and latency (experiments/logs/phase10_api.log).", 90)}

<h2>8. Conclusions</h2>
<ul>
<li>Framing map matching as candidate classification lets standard tabular models handle trip-specific arc ids.
XGBoost was the best of eight models (student-test Top-1 {f4(m('xgb','test_top1'))}), well above the best hand-written
rule (0.8261).</li>
<li>Bayesian optimisation confirmed the default configuration was already near the optimum. The remaining error comes
from ambiguity in the data, not from model capacity.</li>
<li>Route-level decoding is where the biggest gain comes from: the HMM raises Top-1 to {f4(test_hmm.top1)} and route F1
to {f4(test_hmm.route_f1)}, and produces almost gap-free routes.</li>
<li><b>Limitations:</b> points with no candidate within 50 m stay unmatched; manoeuvres such as U-turns on short arcs can
be smoothed away; and the label alignment we built ourselves, not given ground truth, defines the per-point targets.</li>
</ul>

<h2>9. Reproducibility and AI-assistance disclosure</h2>
<p><b>Reproducing the results.</b> Every number and figure in this report comes from scripts in the repository, which
run in order: <code>scripts/01_eda.py</code> to <code>09_build_report.py</code> (or <code>run_all.sh</code>). Seeds are fixed
(42), and logs are in <code>experiments/logs/</code>.</p>
<p><b>AI assistance.</b> The team used an AI coding assistant (Claude) to help write code, scaffolding and report
text. All experiments were run by the team on the provided data, and all reported numbers are read from the committed
result files. Each member reviewed the code and the results, and can explain each part.</p>
</body></html>"""

out_html = REP / "report.html"
out_html.write_text(html, encoding="utf-8")
print(f"wrote {out_html}")

pdf = REP / "PathSnap_Report.pdf"
chrome = CHROME if Path(CHROME).exists() else shutil.which("chromium") or shutil.which("google-chrome")
if chrome:
    subprocess.run([chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", out_html.as_uri()], check=True, capture_output=True)
    print(f"wrote {pdf} ({pdf.stat().st_size / 1e6:.1f} MB)")
else:
    print("Chrome not found: open reports/report.html in a browser and print to PDF")
