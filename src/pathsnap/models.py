"""Model zoo for candidate scoring, with the initial (untuned) configurations.

Every model outputs P(candidate is the reference arc). Models that train
iteratively (LightGBM, XGBoost, MLP) also return per-iteration train/validation
log-loss histories; the others have no epoch-wise history.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .config import SEED


class MLPWithHistory(BaseEstimator, ClassifierMixin):
    """Standardise + sklearn MLP trained epoch by epoch so train/val loss can be recorded."""

    def __init__(self, hidden=(64, 32), alpha=1e-4, lr=1e-3, batch_size=1024, epochs=20, seed=SEED):
        self.hidden, self.alpha, self.lr = hidden, alpha, lr
        self.batch_size, self.epochs, self.seed = batch_size, epochs, seed

    def fit(self, X, y, X_val=None, y_val=None):
        rng = np.random.default_rng(self.seed)
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        Xv = self.scaler_.transform(X_val) if X_val is not None else None
        self.mlp_ = MLPClassifier(hidden_layer_sizes=self.hidden, alpha=self.alpha, learning_rate_init=self.lr,
                                  batch_size=self.batch_size, random_state=self.seed)
        mon = rng.choice(len(Xs), size=min(len(Xs), 100_000), replace=False)
        self.history_ = {"train_logloss": [], "valid_logloss": []}
        self.classes_ = np.array([0, 1])
        for _ in range(self.epochs):
            perm = rng.permutation(len(Xs))
            self.mlp_.partial_fit(Xs[perm], y[perm], classes=self.classes_)
            self.history_["train_logloss"].append(log_loss(y[mon], self.mlp_.predict_proba(Xs[mon])[:, 1], labels=[0, 1]))
            if Xv is not None:
                self.history_["valid_logloss"].append(log_loss(y_val, self.mlp_.predict_proba(Xv)[:, 1], labels=[0, 1]))
        return self

    def predict_proba(self, X):
        return self.mlp_.predict_proba(self.scaler_.transform(X))


@dataclass
class ZooModel:
    name: str
    family: str
    config: dict
    make: Callable[[dict], object]
    max_train_rows: int | None = None   # subsample training rows (instance-based models)
    history: str | None = None          # "lgbm" | "xgb" | "mlp"
    notes: str = ""
    extra: dict = field(default_factory=dict)


def _logreg(c):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c["C"], max_iter=c["max_iter"]))


def _gnb(c):
    return GaussianNB(var_smoothing=c["var_smoothing"])


def _knn(c):
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=c["n_neighbors"], weights=c["weights"],
                                                                algorithm="brute", n_jobs=-1))


def _dtree(c):
    return DecisionTreeClassifier(max_depth=c["max_depth"], min_samples_leaf=c["min_samples_leaf"], random_state=SEED)


def _rf(c):
    return RandomForestClassifier(n_estimators=c["n_estimators"], min_samples_leaf=c["min_samples_leaf"],
                                  max_features=c["max_features"], max_samples=c["max_samples"], n_jobs=-1,
                                  random_state=SEED)


def _lgbm(c):
    return LGBMClassifier(**c, random_state=SEED, n_jobs=-1, verbose=-1)


def _xgb(c):
    return XGBClassifier(**c, tree_method="hist", eval_metric="logloss", random_state=SEED, n_jobs=-1)


def _mlp(c):
    return MLPWithHistory(**c)


ZOO: dict[str, ZooModel] = {m.name: m for m in [
    ZooModel("logreg", "Linear", {"C": 1.0, "max_iter": 1000}, _logreg),
    ZooModel("gnb", "Probabilistic", {"var_smoothing": 1e-9}, _gnb),
    ZooModel("knn", "Instance-based", {"n_neighbors": 25, "weights": "distance"}, _knn, max_train_rows=60_000,
             notes="trained on a 60k-row random subsample (brute-force search cost)"),
    ZooModel("dtree", "Tree", {"max_depth": 12, "min_samples_leaf": 50}, _dtree),
    ZooModel("rf", "Ensemble (bagging)", {"n_estimators": 100, "min_samples_leaf": 5, "max_features": "sqrt",
                                          "max_samples": 0.5}, _rf),
    ZooModel("lgbm", "Boosting", {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 63, "subsample": 0.8,
                                  "subsample_freq": 1, "colsample_bytree": 0.8}, _lgbm, history="lgbm"),
    ZooModel("xgb", "Boosting", {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 7, "subsample": 0.8,
                                 "colsample_bytree": 0.8}, _xgb, history="xgb"),
    ZooModel("mlp", "Neural network", {"hidden": (64, 32), "alpha": 1e-4, "lr": 1e-3, "batch_size": 1024,
                                       "epochs": 20}, _mlp, history="mlp"),
]}


def fit_model(zm: ZooModel, X, y, X_val=None, y_val=None):
    """Fit and return (estimator, history dict or None)."""
    est = zm.make(zm.config)
    if zm.history == "lgbm":
        sets = [(X, y)] + ([(X_val, y_val)] if X_val is not None else [])
        est.fit(X, y, eval_set=sets, eval_metric="binary_logloss")
        curves = [v["binary_logloss"] for v in est.evals_result_.values()]  # insertion order = eval_set order
        h = {"train_logloss": curves[0]}
        if X_val is not None:
            h["valid_logloss"] = curves[1]
        return est, h
    if zm.history == "xgb":
        sets = [(X, y)] + ([(X_val, y_val)] if X_val is not None else [])
        est.fit(X, y, eval_set=sets, verbose=False)
        r = est.evals_result()
        h = {"train_logloss": r["validation_0"]["logloss"]}
        if X_val is not None:
            h["valid_logloss"] = r["validation_1"]["logloss"]
        return est, h
    if zm.history == "mlp":
        est.fit(X, y, X_val, y_val)
        return est, est.history_
    est.fit(X, y)
    return est, None
