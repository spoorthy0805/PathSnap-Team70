| model | family | config | CV macro-F1 | CV Top-1 | Test Top-1 | Test macro-F1 | Test log loss | Route continuity | Fit s/fold | Inference ms/1k pts |
|---|---|---|---|---|---|---|---|---|---|---|
| logreg | Linear | {"C": 1.0, "max_iter": 1000} | 0.9292 ± 0.0111 | 0.8795 | 0.8764 | 0.9235 | 0.0985 | 0.9339 | 0.7 | 0.1 |
| gnb | Probabilistic | {"var_smoothing": 1e-09} | 0.8708 ± 0.0155 | 0.8574 | 0.8618 | 0.8673 | 0.4773 | 0.9239 | 0.1 | 0.3 |
| knn | Instance-based | {"n_neighbors": 25, "weights": "distance"} | 0.9245 ± 0.0117 | 0.8717 | 0.8745 | 0.9199 | 0.1295 | 0.9296 | 0.0 | 214.9 |
| dtree | Tree | {"max_depth": 12, "min_samples_leaf": 50} | 0.9313 ± 0.0121 | 0.8792 | 0.8837 | 0.9305 | 0.0864 | 0.9346 | 11.3 | 0.2 |
| rf | Ensemble (bagging) | {"n_estimators": 100, "min_samples_leaf": 5, "max_features": "sqrt", "max_samples": 0.5} | 0.9364 ± 0.0113 | 0.8841 | 0.8859 | 0.9328 | 0.0769 | 0.9364 | 24.9 | 6.8 |
| lgbm | Boosting | {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 63, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8} | 0.9359 ± 0.0127 | 0.8839 | 0.8945 | 0.9360 | 0.0765 | 0.9378 | 8.7 | 17.3 |
| xgb | Boosting | {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 7, "subsample": 0.8, "colsample_bytree": 0.8} | 0.9369 ± 0.0127 | 0.8863 | 0.8970 | 0.9375 | 0.0731 | 0.9414 | 6.9 | 3.9 |
| mlp | Neural network | {"hidden": [64, 32], "alpha": 0.0001, "lr": 0.001, "batch_size": 1024, "epochs": 20} | 0.9328 ± 0.0117 | 0.8830 | 0.8890 | 0.9309 | 0.0801 | 0.9423 | 12.1 | 0.7 |
