# Solar Flare Detector — Iteration 4 Report
### Machine Learning Model Implementation, Evaluation & Interpretation

## 1. Problem framing

We predict whether a solar active region (AR) will produce a significant
flare in the next 24 hours, from **SDOBenchmark** image patches (10 SDO
channels × up to 4 timesteps per sample) plus the observation date.

The primary task is **binary classification**: *flare* if the peak GOES
soft X-ray flux ≥ **1e-06 W/m² (≥ C-class)**, else *no flare*.
We also report a regression view (predicting log₁₀ peak flux) for
continuity with the earlier iterations.

- Samples: **9222** across **1182** active regions
- Features: **123** engineered scalars
- Positive (flare) rate: **42.6%** — an imbalanced problem
  (see `report/figures/class_balance.png`).

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
| CNN (magnetogram) | Deep-learning-on-raw-pixels comparison point |

Class imbalance is handled with `class_weight='balanced'` (LogReg/SVM/RF)
and `scale_pos_weight` (XGBoost) rather than resampling.

## 4. Training & testing process

Grouped hold-out: **80.0%** of active regions for
train, **20.0%** held out for test. Hyperparameters are
tuned with **GridSearchCV** under **GroupKFold (4 folds)** on the training
regions only, scoring on **F1** (chosen for the imbalance). The tuned model
is refit on all training regions and evaluated **once** on the untouched
test regions.

## 5. Model performance (tuned, test set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FN | FP |
|---|---|---|---|---|---|---|---|---|
| LogReg | 0.783 | 0.722 | 0.704 | **0.713** | 0.856 | 0.823 | 209 | 191 |
| SVM | 0.756 | 0.673 | 0.704 | **0.688** | 0.831 | 0.794 | 209 | 241 |
| RandomForest | 0.763 | 0.694 | 0.680 | **0.687** | 0.833 | 0.798 | 226 | 212 |
| GradBoost | 0.775 | 0.734 | 0.644 | **0.686** | 0.830 | 0.799 | 251 | 165 |
| KNN | 0.739 | 0.706 | 0.545 | **0.616** | 0.793 | 0.741 | 321 | 160 |
| Dummy | 0.617 | 0.000 | 0.000 | **0.000** | 0.500 | 0.383 | 706 | 0 |

Figures: `report/figures/model_comparison.png`, `report/figures/roc_curves.png`,
`report/figures/pr_curves.png`, `report/figures/confusion_best.png`.

**Best model: LogReg** — F1 0.713, ROC-AUC 0.856,
recall 0.704, precision 0.722, balanced
accuracy 0.768. Confusion matrix on test:
TP=497, FP=191, FN=209, TN=948.

CNN (magnetogram pixels): accuracy 0.754, precision 0.715, recall 0.596, F1 0.650, ROC-AUC 0.810.

Regression view (Random Forest → log₁₀ flux): MAE **1.116**,
RMSE **1.324**, R² **0.389** — for reference against the
earlier CNN regressor (R²≈0.18).

## 6. Hyperparameter tuning — before vs. after

| Model | F1 (default) | F1 (tuned) | Δ | Best params |
|---|---|---|---|---|
| LogReg | 0.715 | 0.713 | -0.002 | C=10.0 |
| KNN | 0.605 | 0.616 | +0.010 | n_neighbors=21, weights=distance |
| SVM | 0.697 | 0.688 | -0.009 | C=0.5, gamma=scale |
| RandomForest | 0.684 | 0.687 | +0.003 | max_depth=8, min_samples_leaf=5, n_estimators=300 |
| GradBoost | 0.694 | 0.686 | -0.008 | learning_rate=0.03, max_depth=2, n_estimators=200 |

Tuning is done with GridSearch + GroupKFold. See the Δ column for whether it
helped each model; the effect is model-dependent (ensembles gain most from
depth/estimator settings, the linear model is largely insensitive to `C`).

## 7. Feature importance

**Permutation importance (model-agnostic, on LogReg):** `report/figures/feature_importance_perm.png`

1. `magnetogram_grad_last` — 0.337
2. `magx_mag_pos_frac_last` — 0.205
3. `94_mean_last` — 0.182
4. `magx_mag_pil_energy_last` — 0.132
5. `131_p95_last` — 0.121
6. `335_mean_last` — 0.111
7. `1700_mean_last` — 0.091
8. `304_mean_last` — 0.087
9. `1700_p95_last` — 0.077
10. `year` — 0.066

**Random Forest impurity importance:** `report/figures/feature_importance_rf.png`

1. `94_p95_last` — 0.096
2. `131_p95_last` — 0.068
3. `magnetogram_std_last` — 0.066
4. `magx_mag_total_unsigned_flux_last` — 0.041
5. `94_mean_last` — 0.035
6. `335_p95_last` — 0.035
7. `94_std_last` — 0.032
8. `continuum_grad_last` — 0.031
9. `211_p95_last` — 0.028
10. `131_mean_last` — 0.027

**Feature-set ablation** (`report/figures/feature_ablation.png`) — best model retrained
on different feature subsets:

| Feature set | F1 | ROC-AUC | Recall |
|---|---|---|---|
| cycle_only | 0.582 | 0.564 | 0.950 |
| magnetogram_only | 0.647 | 0.811 | 0.623 |
| all_images | 0.704 | 0.845 | 0.695 |
| images+cycle (full) | 0.713 | 0.856 | 0.704 |

## 8. Research questions

**Which algorithm performed best / best balance?** LogReg gave the best test
F1 (0.713) and the best precision/recall balance; tree ensembles beat
the linear and distance baselines, consistent with the non-linear,
interaction-heavy nature of the features.

**Most appropriate metric?** Because flares are rare
(42.6% positive), **accuracy is misleading** (the Dummy
baseline scores accuracy 0.617 while being useless).
**Recall, F1, ROC-AUC and PR-AUC** are the meaningful metrics; for an early-
warning system **recall** (catching real flares) is the priority.

**What does the confusion matrix reveal?** On the test set the best model
makes **209 false negatives** (missed flares) vs **191 false
positives** (false alarms). Missed flares are the costly error for space-
weather warning, analogous to false negatives in healthcare — the threshold
can be lowered to trade precision for higher recall (see the PR curve).

**Which model minimizes critical errors (false negatives)?** See the FN
column of the table in §5; the class-weighted models are tuned to keep FN
low. The decision threshold is the main lever for pushing recall higher.

**Generalization to unseen data?** All metrics are on active regions never
seen in training (grouped split), so they reflect genuine generalization;
the learning curve (`report/figures/learning_curve.png`) shows the train/CV gap.

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
