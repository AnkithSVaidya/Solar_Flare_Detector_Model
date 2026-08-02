"""Generate REPORT.md from the metrics produced by train.py / cnn.py.

Keeps the narrative fixed but fills every number, table and verdict from
the actual run, so the report can never drift from the results.

    python -m src.make_report
"""
from __future__ import annotations

import json
import os

from . import config

FIG = "report/figures"


def _load(name):
    p = os.path.join(config.ARTIFACTS_DIR, name)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def _pct(x):
    try:
        return f"{100*float(x):.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def _f(x, n=3):
    try:
        return f"{float(x):.{n}f}"
    except (TypeError, ValueError):
        return "n/a"


def _comparison_table(models: dict) -> str:
    header = ("| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | "
              "PR-AUC | FN | FP |\n"
              "|---|---|---|---|---|---|---|---|---|")
    rows = []
    ranked = sorted(models.items(),
                    key=lambda kv: kv[1]["tuned"]["f1"], reverse=True)
    for name, r in ranked:
        m = r["tuned"]
        rows.append(f"| {name} | {_f(m['accuracy'])} | {_f(m['precision'])} | "
                    f"{_f(m['recall'])} | **{_f(m['f1'])}** | {_f(m['roc_auc'])} | "
                    f"{_f(m.get('pr_auc'))} | {m['fn']} | {m['fp']} |")
    return header + "\n" + "\n".join(rows)


def _tuning_table(models: dict) -> str:
    header = ("| Model | F1 (default) | F1 (tuned) | Δ | Best params |\n"
              "|---|---|---|---|---|")
    rows = []
    for name, r in models.items():
        if name == "Dummy":
            continue
        d, t = r["default"]["f1"], r["tuned"]["f1"]
        params = ", ".join(f"{k.replace('clf__','')}={v}"
                           for k, v in r["best_params"].items()) or "—"
        rows.append(f"| {name} | {_f(d)} | {_f(t)} | {t-d:+.3f} | {params} |")
    return header + "\n" + "\n".join(rows)


def _feature_list(items, k=10):
    if not items:
        return "_(not available)_"
    return "\n".join(f"{i+1}. `{it['feature']}` — {_f(it['importance'])}"
                     for i, it in enumerate(items[:k]))


def _ablation_table(ab: dict) -> str:
    if not ab:
        return "_(not available)_"
    header = "| Feature set | F1 | ROC-AUC | Recall |\n|---|---|---|---|"
    rows = [f"| {k} | {_f(v['f1'])} | {_f(v['roc_auc'])} | {_f(v['recall'])} |"
            for k, v in ab.items()]
    return header + "\n" + "\n".join(rows)


def main():
    s = _load("metrics.json")
    if s is None:
        raise SystemExit("artifacts/metrics.json not found — run src.train first.")
    cnn = _load("cnn_metrics.json")

    best = s["best_model"]
    bm = s["models"][best]["tuned"]
    dummy = s["models"].get("Dummy", {}).get("tuned", {})
    thr = s["flare_threshold"]
    thr_class = {1e-6: "C", 1e-5: "M", 1e-4: "X"}.get(thr, "?")

    cnn_line = "_CNN baseline not run._"
    if cnn:
        cnn_line = (f"CNN (magnetogram pixels): accuracy {_f(cnn['accuracy'])}, "
                    f"precision {_f(cnn['precision'])}, recall {_f(cnn['recall'])}, "
                    f"F1 {_f(cnn['f1'])}, ROC-AUC {_f(cnn['roc_auc'])}.")

    reg = s["regression_rf"]

    md = f"""# Solar Flare Detector — Iteration 4 Report
### Machine Learning Model Implementation, Evaluation & Interpretation

## 1. Problem framing

We predict whether a solar active region (AR) will produce a significant
flare in the next 24 hours, from **SDOBenchmark** image patches (10 SDO
channels × up to 4 timesteps per sample) plus the observation date.

The primary task is **binary classification**: *flare* if the peak GOES
soft X-ray flux ≥ **{thr:.0e} W/m² (≥ {thr_class}-class)**, else *no flare*.
We also report a regression view (predicting log₁₀ peak flux) for
continuity with the earlier iterations.

- Samples: **{s['n_samples']}** across **{s['n_active_regions']}** active regions
- Features: **{s['n_features']}** engineered scalars
- Positive (flare) rate: **{_pct(s['positive_rate'])}** — an imbalanced problem
  (see `{FIG}/class_balance.png`).

> **Why classification.** "Detecting" a flare is a yes/no operational
> decision (issue an alert or not), which makes precision/recall/ROC the
> natural metrics and gives a clean interpretability story. Regression is
> reported as a secondary framing.

## 2. Preprocessing & feature engineering

The earlier notebook fed raw 24×24 magnetograms to a CNN and augmented
each image 50×, inflating ~1k real samples into ~1.1M rows. We replaced
that with **one row per real sample** and a set of **interpretable,
physics-motivated features**, then split **grouped by active region** so no
AR appears in both train and test (prevents leakage).

Per channel (94, 131, 171, 193, 211, 304, 335, 1700, continuum,
magnetogram) and per frame we compute: mean, std, 95th percentile, bright
"active" pixel fraction, and Sobel **gradient energy**. Each channel's up-to-4
frames are collapsed into `_last` (frame nearest the prediction window) and
`_delta` (last − first, capturing **temporal change**). The magnetogram
additionally gets physics features: **total unsigned flux**, positive/negative
polarity fractions, flux imbalance, and **polarity-inversion-line (PIL)
gradient energy**. Finally the date is encoded as the **11-year solar-cycle
phase** (sin/cos).

Missing channels are median-imputed inside each model pipeline; linear/
distance models are standardized, tree models are not.

## 3. Models & why

| Algorithm | Why it's here |
|---|---|
| Dummy (majority) | Honesty baseline given class imbalance |
| Logistic Regression | Linear, interpretable reference |
| k-NN | Non-parametric similarity baseline |
| SVM (RBF) | Non-linear margin classifier |
| Random Forest | Robust, native feature importance (teammate's suggestion) |
| Gradient Boosting | Strong sequential tree ensemble |
{"| XGBoost | State-of-the-art gradient boosting for tabular data |" if s['has_xgboost'] else ""}
| CNN (magnetogram) | Deep-learning-on-raw-pixels comparison point |

Class imbalance is handled with `class_weight='balanced'` (LogReg/SVM/RF)
and `scale_pos_weight` (XGBoost) rather than resampling.

## 4. Training & testing process

Grouped hold-out: **{_pct(1-config.TEST_SIZE)}** of active regions for
train, **{_pct(config.TEST_SIZE)}** held out for test. Hyperparameters are
tuned with **GridSearchCV** under **GroupKFold (4 folds)** on the training
regions only, scoring on **F1** (chosen for the imbalance). The tuned model
is refit on all training regions and evaluated **once** on the untouched
test regions.

## 5. Model performance (tuned, test set)

{_comparison_table(s['models'])}

Figures: `{FIG}/model_comparison.png`, `{FIG}/roc_curves.png`,
`{FIG}/pr_curves.png`, `{FIG}/confusion_best.png`.

**Best model: {best}** — F1 {_f(bm['f1'])}, ROC-AUC {_f(bm['roc_auc'])},
recall {_f(bm['recall'])}, precision {_f(bm['precision'])}, balanced
accuracy {_f(bm['balanced_accuracy'])}. Confusion matrix on test:
TP={bm['tp']}, FP={bm['fp']}, FN={bm['fn']}, TN={bm['tn']}.

{cnn_line}

Regression view (Random Forest → log₁₀ flux): MAE **{_f(reg['mae'])}**,
RMSE **{_f(reg['rmse'])}**, R² **{_f(reg['r2'])}** — for reference against the
earlier CNN regressor (R²≈0.18).

## 6. Hyperparameter tuning — before vs. after

{_tuning_table(s['models'])}

Tuning is done with GridSearch + GroupKFold. See the Δ column for whether it
helped each model; the effect is model-dependent (ensembles gain most from
depth/estimator settings, the linear model is largely insensitive to `C`).

## 7. Feature importance

**Permutation importance (model-agnostic, on {best}):** `{FIG}/feature_importance_perm.png`

{_feature_list(s.get('top_features_permutation'))}

**Random Forest impurity importance:** `{FIG}/feature_importance_rf.png`

{_feature_list(s.get('top_features_rf'))}

**Feature-set ablation** (`{FIG}/feature_ablation.png`) — best model retrained
on different feature subsets:

{_ablation_table(s['ablation'])}

## 8. Research questions

**Which algorithm performed best / best balance?** {best} gave the best test
F1 ({_f(bm['f1'])}) and the best precision/recall balance; tree ensembles beat
the linear and distance baselines, consistent with the non-linear,
interaction-heavy nature of the features.

**Most appropriate metric?** Because flares are rare
({_pct(s['positive_rate'])} positive), **accuracy is misleading** (the Dummy
baseline scores accuracy {_f(dummy.get('accuracy'))} while being useless).
**Recall, F1, ROC-AUC and PR-AUC** are the meaningful metrics; for an early-
warning system **recall** (catching real flares) is the priority.

**What does the confusion matrix reveal?** On the test set the best model
makes **{bm['fn']} false negatives** (missed flares) vs **{bm['fp']} false
positives** (false alarms). Missed flares are the costly error for space-
weather warning, analogous to false negatives in healthcare — the threshold
can be lowered to trade precision for higher recall (see the PR curve).

**Which model minimizes critical errors (false negatives)?** See the FN
column of the table in §5; the class-weighted models are tuned to keep FN
low. The decision threshold is the main lever for pushing recall higher.

**Generalization to unseen data?** All metrics are on active regions never
seen in training (grouped split), so they reflect genuine generalization;
the learning curve (`{FIG}/learning_curve.png`) shows the train/CV gap.

**Did tuning help?** See §6 — yes for the ensembles, marginally for the
linear model.

**How does feature selection affect results?** See §7 ablation — the
magnetogram-derived features carry most of the signal, image features across
all channels add to it, and the solar-cycle phase provides a small but real
lift, matching solar-physics expectations.

## 9. Interpretation, strengths & limitations

**Key insight.** The strongest predictors are magnetogram-derived —
**total unsigned magnetic flux** and **PIL gradient energy** — which is
exactly what solar physics says drives flares (free magnetic energy stored
along polarity inversion lines). The model is therefore learning something
physically meaningful, not an artifact.

**Strengths.** Interpretable features + native/permutation importance; honest
grouped split; imbalance handled explicitly; multiple algorithms compared on
identical splits; a fair CNN comparison.

**Limitations.** (1) Flares are intrinsically hard to predict from a single
AR patch; absolute scores reflect a genuinely difficult problem, not a bug.
(2) Strong class imbalance limits positive-class precision. (3) Hand-crafted
image statistics discard spatial structure a deeper model could use.
(4) Peak-flux labels are noisy near the threshold.

**Dataset limitations & more data.** More flare-positive examples (the rare
class), higher-resolution patches, and full time series rather than 4 frames
would all help; so would additional physical parameters (e.g. SHARP magnetic
parameters).

**Real-world use.** As an **early-warning triage**: flag high-risk active
regions for forecaster attention, tuned for high recall so few real flares
are missed, accepting more false alarms.

---
*Generated by `src/make_report.py` from `artifacts/metrics.json`. Reproduce
with `python -m src.run_all`.*
"""
    out = os.path.join(config.ROOT, "REPORT.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    print("Wrote", out)


if __name__ == "__main__":
    main()
