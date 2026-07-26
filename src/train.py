"""Train, tune, evaluate and interpret the tabular models.

Pipeline:
  1. Load artifacts/features.csv.
  2. Group-split by active region (no AR appears in both train and test).
  3. For every model: evaluate default vs. GridSearch-tuned (GroupKFold CV).
  4. Pick the best tuned model by test F1.
  5. Produce every figure the report needs (confusion, ROC, PR, feature
     importance, learning curve, model comparison, feature-set ablation)
     and a short regression comparison.
  6. Dump artifacts/metrics.json + artifacts/model_comparison.csv and the
     best pipeline to artifacts/best_model.joblib.

Run:  python -m src.train
"""
from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import (
    GroupShuffleSplit, GroupKFold, GridSearchCV, learning_curve,
)
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    matthews_corrcoef, confusion_matrix,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from . import config, plots
from .models import build_models, has_xgboost

warnings.filterwarnings("ignore")

META_COLS = ["id", "active_region", "peak_flux", "log10_flux", "label"]
CYCLE_COLS = ["cycle_sin", "cycle_cos", "year"]
CV_SPLITS = 4
TUNE_SCORING = "f1"


# ----------------------------------------------------------------------
# Data loading / splitting
# ----------------------------------------------------------------------
def load_data():
    df = pd.read_csv(config.FEATURES_CSV)
    feat_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feat_cols].astype(float)
    y = df["label"].astype(int).values
    groups = df["active_region"].astype(str).values
    return df, X, y, groups, feat_cols


def group_holdout(X, y, groups, test_size, seed):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(X, y, groups))
    return tr, te


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def score_all(y_true, y_pred, y_score) -> dict:
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else 0.0,
    }
    try:
        out["roc_auc"] = roc_auc_score(y_true, y_score)
        out["pr_auc"] = average_precision_score(y_true, y_score)
    except ValueError:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    out["specificity"] = tn / (tn + fp) if (tn + fp) else 0.0
    return out


def get_scores(pipe, X):
    """Positive-class probability (or decision score fallback)."""
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    if hasattr(pipe, "decision_function"):
        d = pipe.decision_function(X)
        return (d - d.min()) / (d.ptp() + 1e-9)
    return pipe.predict(X).astype(float)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    config.ensure_dirs()
    df, X, y, groups, feat_cols = load_data()
    print(f"Loaded {len(df)} samples, {len(feat_cols)} features, "
          f"{y.mean()*100:.1f}% positive, "
          f"{len(set(groups))} active regions.")

    plots.plot_class_balance(y)

    tr, te = group_holdout(X, y, groups, config.TEST_SIZE, config.RANDOM_STATE)
    X_tr, X_te = X.iloc[tr], X.iloc[te]
    y_tr, y_te = y[tr], y[te]
    g_tr = groups[tr]
    print(f"Train: {len(tr)}  Test: {len(te)}  "
          f"(train pos {y_tr.mean()*100:.1f}%, test pos {y_te.mean()*100:.1f}%)")

    pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    models = build_models(pos_weight=pos_weight)
    cv = GroupKFold(n_splits=CV_SPLITS)

    results = {}            # name -> {"default":..., "tuned":..., "params":...}
    roc_curves = {}         # name -> (y_true, y_score)  [tuned models]
    tuned_estimators = {}   # name -> fitted best pipeline

    for name, (pipe, grid) in models.items():
        print(f"\n=== {name} ===")
        # --- default (untuned) ---
        base = clone(pipe).fit(X_tr, y_tr)
        yp = base.predict(X_te)
        ys = get_scores(base, X_te)
        default_metrics = score_all(y_te, yp, ys)
        print(f"  default : F1={default_metrics['f1']:.3f} "
              f"ROC-AUC={default_metrics['roc_auc']:.3f} "
              f"recall={default_metrics['recall']:.3f}")

        # --- tuned ---
        if grid:
            search = GridSearchCV(clone(pipe), grid, scoring=TUNE_SCORING,
                                  cv=cv, n_jobs=-1, refit=True)
            search.fit(X_tr, y_tr, groups=g_tr)
            best = search.best_estimator_
            best_params = search.best_params_
        else:
            best, best_params = base, {}
        yp = best.predict(X_te)
        ys = get_scores(best, X_te)
        tuned_metrics = score_all(y_te, yp, ys)
        print(f"  tuned   : F1={tuned_metrics['f1']:.3f} "
              f"ROC-AUC={tuned_metrics['roc_auc']:.3f} "
              f"recall={tuned_metrics['recall']:.3f}  {best_params}")

        results[name] = {"default": default_metrics, "tuned": tuned_metrics,
                         "best_params": best_params}
        roc_curves[name] = (y_te, ys)
        tuned_estimators[name] = best

    # ------------------------------------------------------------------
    # Pick the best model (by tuned test F1, excluding the Dummy)
    # ------------------------------------------------------------------
    ranked = sorted(
        [(n, r["tuned"]["f1"]) for n, r in results.items() if n != "Dummy"],
        key=lambda t: t[1], reverse=True)
    best_name = ranked[0][0]
    best_model = tuned_estimators[best_name]
    print(f"\nBest model: {best_name} (test F1={ranked[0][1]:.3f})")

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    tuned_only = {n: results[n]["tuned"] for n in results}
    plots.plot_model_comparison(tuned_only)

    curve_subset = {n: roc_curves[n] for n in roc_curves if n != "Dummy"}
    plots.plot_roc(curve_subset)
    plots.plot_pr(curve_subset)

    yp_best = best_model.predict(X_te)
    plots.plot_confusion(y_te, yp_best,
                         f"Confusion Matrix — {best_name} (test)",
                         "confusion_best.png")

    def _top(names, scores, k=15):
        order = np.argsort(scores)[::-1][:k]
        return [{"feature": names[i], "importance": float(scores[i])}
                for i in order]

    # Feature importance: permutation on the best model (model-agnostic)
    perm = permutation_importance(best_model, X_te, y_te, scoring="f1",
                                  n_repeats=10, random_state=config.RANDOM_STATE,
                                  n_jobs=-1)
    plots.plot_feature_importance(
        feat_cols, perm.importances_mean,
        f"Permutation Importance — {best_name}", "feature_importance_perm.png")
    top_perm = _top(feat_cols, perm.importances_mean)

    # Native impurity importance from the tuned RandomForest (always present)
    top_rf = []
    if "RandomForest" in tuned_estimators:
        rf = tuned_estimators["RandomForest"].named_steps["clf"]
        plots.plot_feature_importance(
            feat_cols, rf.feature_importances_,
            "RandomForest Feature Importance (impurity)",
            "feature_importance_rf.png")
        top_rf = _top(feat_cols, rf.feature_importances_)

    # Learning curve for the best model
    try:
        ts, tr_sc, va_sc = learning_curve(
            clone(best_model), X_tr, y_tr, groups=g_tr, cv=cv,
            scoring="f1", train_sizes=np.linspace(0.2, 1.0, 5), n_jobs=-1)
        plots.plot_learning_curve(ts, tr_sc, va_sc,
                                  f"Learning Curve — {best_name}",
                                  "learning_curve.png")
    except Exception as e:
        print("Learning curve skipped:", e)

    # ------------------------------------------------------------------
    # Feature-set ablation (how does feature selection change results?)
    # ------------------------------------------------------------------
    img_cols = [c for c in feat_cols if c not in CYCLE_COLS]
    mag_cols = [c for c in feat_cols if c.startswith(("magnetogram", "magx"))]
    feature_sets = {
        "cycle_only": [c for c in CYCLE_COLS if c in feat_cols],
        "magnetogram_only": mag_cols,
        "all_images": img_cols,
        "images+cycle (full)": feat_cols,
    }
    ablation = {}
    for set_name, cols in feature_sets.items():
        if not cols:
            continue
        est = clone(best_model)
        est.fit(X_tr[cols], y_tr)
        yp = est.predict(X_te[cols])
        ys = get_scores(est, X_te[cols])
        ablation[set_name] = score_all(y_te, yp, ys)
        print(f"  ablation [{set_name:>22}] "
              f"F1={ablation[set_name]['f1']:.3f} "
              f"ROC-AUC={ablation[set_name]['roc_auc']:.3f} "
              f"({len(cols)} feats)")
    if ablation:
        plots.plot_ablation(ablation, metric="f1")

    # ------------------------------------------------------------------
    # Regression comparison (the original framing): predict log10 flux
    # ------------------------------------------------------------------
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    reg = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("rf", RandomForestRegressor(n_estimators=400, n_jobs=-1,
                                     random_state=config.RANDOM_STATE)),
    ])
    yreg_tr = df["log10_flux"].values[tr]
    yreg_te = df["log10_flux"].values[te]
    reg.fit(X_tr, yreg_tr)
    yreg_pred = reg.predict(X_te)
    regression = {
        "mae": float(mean_absolute_error(yreg_te, yreg_pred)),
        "rmse": float(np.sqrt(mean_squared_error(yreg_te, yreg_pred))),
        "r2": float(r2_score(yreg_te, yreg_pred)),
    }
    print(f"\nRegression (RF, log10 flux): "
          f"MAE={regression['mae']:.3f} R2={regression['r2']:.3f}")

    # ------------------------------------------------------------------
    # Persist everything
    # ------------------------------------------------------------------
    summary = {
        "n_samples": int(len(df)),
        "n_features": int(len(feat_cols)),
        "n_active_regions": int(len(set(groups))),
        "positive_rate": float(y.mean()),
        "flare_threshold": config.FLARE_THRESHOLD,
        "test_positive_rate": float(y_te.mean()),
        "best_model": best_name,
        "has_xgboost": has_xgboost(),
        "models": results,
        "ablation": ablation,
        "regression_rf": regression,
        "feature_names": feat_cols,
        "top_features_permutation": top_perm,
        "top_features_rf": top_rf,
    }
    with open(os.path.join(config.ARTIFACTS_DIR, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    # Flat comparison table (tuned metrics) for the report
    rows = []
    for n, r in results.items():
        m = r["tuned"]
        rows.append({"model": n, **{k: m[k] for k in
                    ["accuracy", "precision", "recall", "f1",
                     "roc_auc", "pr_auc", "balanced_accuracy",
                     "fn", "fp", "tp", "tn"]}})
    comp = pd.DataFrame(rows).sort_values("f1", ascending=False)
    comp.to_csv(os.path.join(config.ARTIFACTS_DIR, "model_comparison.csv"),
                index=False)

    try:
        import joblib
        joblib.dump(best_model,
                    os.path.join(config.ARTIFACTS_DIR, "best_model.joblib"))
    except Exception as e:
        print("Could not pickle best model:", e)

    print("\nWrote artifacts/metrics.json, model_comparison.csv, "
          "best_model.joblib and figures under report/figures/.")
    return summary


if __name__ == "__main__":
    main()
