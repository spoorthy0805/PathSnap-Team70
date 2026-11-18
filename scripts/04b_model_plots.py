"""Phase 4 figures and report table, generated only from experiments/results/phase4/*.

Outputs
  reports/figures/models_cv_f1.png               CV macro-F1 (mean ± std over folds)
  reports/figures/models_test_top1_f1.png        student_test Top-1 and macro-F1 (+ rule baselines)
  reports/figures/models_logloss.png             CV and student_test log loss
  reports/figures/models_fold_stability.png      per-fold validation Top-1
  reports/figures/models_train_vs_val.png        training-fold vs validation Top-1 (fit diagnosis)
  reports/figures/models_confusion.png           student_test confusion matrices (row-normalised)
  reports/figures/models_loss_curves.png         train/valid log loss per iteration (iterative models only)
  reports/figures/models_timing.png              fit time vs inference time
  experiments/results/phase4/comparison.md       consolidated comparison table
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathsnap.config import FIGURES_DIR, RESULTS_DIR  # noqa: E402

P4 = RESULTS_DIR / "phase4"
BLUE, RED, GREY = "#1f6feb", "#d1242f", "#9aa0a6"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, dpi=130)
    plt.close(fig)


def main():
    comp = pd.read_csv(P4 / "comparison.csv")
    folds = pd.read_csv(P4 / "cv_folds.csv")
    base = pd.read_csv(RESULTS_DIR / "heuristic_baselines.csv")
    models = comp.model.tolist()
    x = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x, comp.cv_f1_macro_mean, yerr=comp.cv_f1_macro_std, capsize=4, color=BLUE)
    for i, v in enumerate(comp.cv_f1_macro_mean):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x, models); ax.set_ylim(comp.cv_f1_macro_mean.min() - 0.05, 1.0)
    ax.set(title="5-fold GroupKFold CV macro-F1 (78 train trips, mean ± std)", ylabel="macro-F1")
    ax.grid(alpha=0.3, axis="y"); save(fig, "models_cv_f1.png")

    fig, ax = plt.subplots(figsize=(10, 4.8))
    w = 0.38
    ax.bar(x - w / 2, comp.test_top1, w, label="Top-1 accuracy", color=BLUE)
    ax.bar(x + w / 2, comp.test_f1_macro, w, label="macro-F1", color=RED)
    for i in range(len(models)):
        ax.text(i - w / 2, comp.test_top1[i] + 0.004, f"{comp.test_top1[i]:.3f}", ha="center", fontsize=8)
        ax.text(i + w / 2, comp.test_f1_macro[i] + 0.004, f"{comp.test_f1_macro[i]:.3f}", ha="center", fontsize=8)
    b = base[(base.split == "student_test") & base.rule.str.contains("heading agreement")].top1.iloc[0]
    ax.axhline(b, ls="--", color=GREY, label=f"rule: nearest + heading ({b:.3f})")
    ax.set_xticks(x, models); ax.set_ylim(min(comp.test_top1.min(), b) - 0.05, 1.0)
    ax.set(title="Student-test (10 trips): observation Top-1 and candidate macro-F1")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3, axis="y"); save(fig, "models_test_top1_f1.png")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w / 2, comp.cv_log_loss_mean, w, yerr=comp.cv_log_loss_std, capsize=3, label="CV", color=BLUE)
    ax.bar(x + w / 2, comp.test_log_loss, w, label="student_test", color=RED)
    ax.set_xticks(x, models); ax.set_yscale("log")
    ax.set(title="Log loss (lower is better; log scale)", ylabel="log loss"); ax.legend(); ax.grid(alpha=0.3, axis="y")
    save(fig, "models_logloss.png")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, m in enumerate(models):
        v = folds[folds.model == m].val_top1
        ax.scatter(np.full(len(v), i), v, color=BLUE, alpha=0.7)
        ax.plot([i - 0.25, i + 0.25], [v.mean()] * 2, color=RED)
    ax.set_xticks(x, models); ax.set(title="Validation Top-1 per CV fold (red = mean)", ylabel="Top-1")
    ax.grid(alpha=0.3, axis="y"); save(fig, "models_fold_stability.png")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w / 2, comp.cv_train_top1_mean, w, label="training-fold trips", color=GREY)
    ax.bar(x + w / 2, comp.cv_top1_mean, w, label="validation-fold trips", color=BLUE)
    ax.set_xticks(x, models); ax.set_ylim(0.8, 1.0)
    ax.set(title="Top-1 on training vs validation trips (CV mean)", ylabel="Top-1"); ax.legend()
    ax.grid(alpha=0.3, axis="y"); save(fig, "models_train_vs_val.png")

    n = len(models)
    cols = 4
    fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(3.6 * cols, 3.3 * np.ceil(n / cols)))
    for ax, m in zip(np.ravel(axes), models):
        cm = np.array(json.loads((P4 / "confusion" / f"{m}.json").read_text())["matrix"], float)
        cmn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cmn[i, j]:.2f}\n({int(cm[i, j]):,})", ha="center", va="center", fontsize=8,
                        color="white" if cmn[i, j] > 0.5 else "black")
        ax.set_xticks([0, 1], ["pred 0", "pred 1"]); ax.set_yticks([0, 1], ["true 0", "true 1"]); ax.set_title(m)
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    fig.suptitle("Student-test candidate-level confusion matrices (threshold 0.5, row-normalised)")
    save(fig, "models_confusion.png")

    hist_models = [m for m in models if (P4 / "histories" / f"{m}.json").exists()]
    if hist_models:
        fig, axes = plt.subplots(1, len(hist_models), figsize=(5 * len(hist_models), 4))
        for ax, m in zip(np.atleast_1d(axes), hist_models):
            h = json.loads((P4 / "histories" / f"{m}.json").read_text())
            tr = np.array([v["train_logloss"] for v in h.values()])
            va = np.array([v["valid_logloss"] for v in h.values()])
            it = np.arange(1, tr.shape[1] + 1)
            for arr, color, lab in [(tr, GREY, "train"), (va, BLUE, "validation")]:
                ax.plot(it, arr.mean(0), color=color, label=f"{lab} (fold mean)")
                ax.fill_between(it, arr.min(0), arr.max(0), color=color, alpha=0.2)
            ax.set(title=f"{m}: log loss per {'epoch' if m == 'mlp' else 'boosting round'}",
                   xlabel="epoch" if m == "mlp" else "iteration", ylabel="log loss")
            ax.legend(); ax.grid(alpha=0.3)
        save(fig, "models_loss_curves.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(comp.cv_fit_s_mean, comp.test_predict_ms_per_1k_points, color=BLUE)
    for _, r in comp.iterrows():
        ax.annotate(r.model, (r.cv_fit_s_mean, r.test_predict_ms_per_1k_points), textcoords="offset points",
                    xytext=(5, 4))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set(title="Cost: CV fit time vs inference time", xlabel="mean fit time per fold (s)",
           ylabel="inference ms per 1,000 GPS points"); ax.grid(alpha=0.3)
    save(fig, "models_timing.png")

    t = comp.assign(**{
        "CV macro-F1": comp.cv_f1_macro_mean.map("{:.4f}".format) + " ± " + comp.cv_f1_macro_std.map("{:.4f}".format),
        "CV Top-1": comp.cv_top1_mean.map("{:.4f}".format),
        "Test Top-1": comp.test_top1.map("{:.4f}".format),
        "Test macro-F1": comp.test_f1_macro.map("{:.4f}".format),
        "Test log loss": comp.test_log_loss.map("{:.4f}".format),
        "Route continuity": comp.test_continuity.map("{:.4f}".format),
        "Fit s/fold": comp.cv_fit_s_mean.map("{:.1f}".format),
        "Inference ms/1k pts": comp.test_predict_ms_per_1k_points.map("{:.1f}".format),
    })[["model", "family", "config", "CV macro-F1", "CV Top-1", "Test Top-1", "Test macro-F1", "Test log loss",
        "Route continuity", "Fit s/fold", "Inference ms/1k pts"]]
    t.config = t.config.str.replace("|", "/", regex=False)
    lines = ["| " + " | ".join(t.columns) + " |", "|" + "---|" * len(t.columns)]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in t.itertuples(index=False)]
    (P4 / "comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
