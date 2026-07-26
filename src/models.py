"""The model zoo and their hyperparameter search spaces.

Each classifier is wrapped in a Pipeline with median imputation (some
channels are missing for some samples) and, where the algorithm needs it,
standard scaling.  Tree ensembles skip scaling since they don't need it.

``class_weight='balanced'`` / ``scale_pos_weight`` handle the imbalance so
we don't have to resample.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:  # pragma: no cover
    _HAS_XGB = False


def _scaled(estimator) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", estimator),
    ])


def _unscaled(estimator) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", estimator),
    ])


def build_models(pos_weight: float = 1.0) -> Dict[str, Tuple[Pipeline, dict]]:
    """Return {name: (pipeline, param_grid)}.

    ``pos_weight`` = (n_negative / n_positive) from the training set, used
    by XGBoost's ``scale_pos_weight``.  Param-grid keys are namespaced with
    the ``clf__`` pipeline-step prefix so they plug straight into
    GridSearchCV.
    """
    rs = 42
    models: Dict[str, Tuple[Pipeline, dict]] = {}

    # Trivial reference point -- always predicts the majority class.
    models["Dummy"] = (
        _unscaled(DummyClassifier(strategy="most_frequent")),
        {},
    )

    models["LogReg"] = (
        _scaled(LogisticRegression(max_iter=2000, class_weight="balanced")),
        {"clf__C": [0.01, 0.1, 1.0, 10.0]},
    )

    models["KNN"] = (
        _scaled(KNeighborsClassifier()),
        {"clf__n_neighbors": [5, 11, 21, 41],
         "clf__weights": ["uniform", "distance"]},
    )

    models["SVM"] = (
        _scaled(SVC(kernel="rbf", probability=True,
                    class_weight="balanced", random_state=rs)),
        {"clf__C": [0.5, 1.0, 5.0], "clf__gamma": ["scale", 0.01, 0.1]},
    )

    models["RandomForest"] = (
        _unscaled(RandomForestClassifier(
            n_estimators=400, class_weight="balanced_subsample",
            n_jobs=-1, random_state=rs)),
        {"clf__n_estimators": [300, 600],
         "clf__max_depth": [None, 8, 16],
         "clf__min_samples_leaf": [1, 3, 5]},
    )

    models["GradBoost"] = (
        _unscaled(GradientBoostingClassifier(random_state=rs)),
        {"clf__n_estimators": [200, 400],
         "clf__learning_rate": [0.03, 0.1],
         "clf__max_depth": [2, 3]},
    )

    if _HAS_XGB:
        models["XGBoost"] = (
            _unscaled(XGBClassifier(
                n_estimators=400, learning_rate=0.05, max_depth=4,
                subsample=0.9, colsample_bytree=0.9,
                eval_metric="logloss", tree_method="hist",
                scale_pos_weight=pos_weight, random_state=rs, n_jobs=-1)),
            {"clf__n_estimators": [300, 600],
             "clf__max_depth": [3, 4, 6],
             "clf__learning_rate": [0.03, 0.1]},
        )

    return models


def has_xgboost() -> bool:
    return _HAS_XGB
