"""All figure generation for the report, kept out of the training logic.

Every function saves a PNG under report/figures/ and returns its path.
The palette is a small, colour-blind-safe categorical set used
consistently across figures.
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc,
    precision_recall_curve, average_precision_score,
)

from . import config

# Colour-blind-safe categorical palette (Okabe-Ito subset).
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00",
           "#CC79A7", "#56B4E9", "#F0E442", "#999999"]


def _save(fig, name: str) -> str:
    config.ensure_dirs()
    path = os.path.join(config.FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_confusion(y_true, y_pred, title: str, name: str) -> str:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ConfusionMatrixDisplay(cm, display_labels=["No Flare", "Flare"]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(title)
    return _save(fig, name)


def plot_roc(curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
             name: str = "roc_curves.png") -> str:
    """curves: {model_name: (y_true, y_score)}."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for i, (label, (yt, ys)) in enumerate(curves.items()):
        fpr, tpr, _ = roc_curve(yt, ys)
        ax.plot(fpr, tpr, color=PALETTE[i % len(PALETTE)], lw=2,
                label=f"{label} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right", fontsize=8)
    return _save(fig, name)


def plot_pr(curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
            name: str = "pr_curves.png") -> str:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for i, (label, (yt, ys)) in enumerate(curves.items()):
        prec, rec, _ = precision_recall_curve(yt, ys)
        ap = average_precision_score(yt, ys)
        ax.plot(rec, prec, color=PALETTE[i % len(PALETTE)], lw=2,
                label=f"{label} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(loc="upper right", fontsize=8)
    return _save(fig, name)


def plot_feature_importance(names: List[str], scores: np.ndarray,
                            title: str, name: str, top: int = 20) -> str:
    order = np.argsort(scores)[::-1][:top]
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(order) + 1))
    ax.barh(range(len(order)), scores[order][::-1], color=PALETTE[0])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[i] for i in order][::-1], fontsize=8)
    ax.set_xlabel("Importance")
    ax.set_title(title)
    return _save(fig, name)


def plot_least_important(names: List[str], scores: np.ndarray,
                         title: str, name: str, bottom: int = 15) -> str:
    """The features with the smallest (near-zero) importance."""
    order = np.argsort(scores)[:bottom]  # ascending -> least important first
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(order) + 1))
    ax.barh(range(len(order)), scores[order][::-1], color=PALETTE[7])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[i] for i in order][::-1], fontsize=8)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("Importance (drop in F1 when shuffled)")
    ax.set_title(title)
    return _save(fig, name)


def plot_signed_importance(names: List[str], coefs: np.ndarray,
                           title: str, name: str, top: int = 15) -> str:
    """Signed linear-model coefficients (direction + magnitude)."""
    order = np.argsort(np.abs(coefs))[::-1][:top][::-1]
    colors = [PALETTE[3] if coefs[i] > 0 else PALETTE[0] for i in order]
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(order) + 1))
    ax.barh(range(len(order)), coefs[order], color=colors)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[i] for i in order], fontsize=8)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("Standardized coefficient  (red = raises flare prob, blue = lowers)")
    ax.set_title(title)
    return _save(fig, name)


def plot_model_comparison(metrics: Dict[str, Dict[str, float]],
                          keys=("accuracy", "precision", "recall",
                                "f1", "roc_auc"),
                          name: str = "model_comparison.png") -> str:
    models = list(metrics.keys())
    x = np.arange(len(models))
    w = 0.8 / len(keys)
    fig, ax = plt.subplots(figsize=(1.4 * len(models) + 3, 4.5))
    for j, k in enumerate(keys):
        vals = [metrics[m].get(k, np.nan) for m in models]
        ax.bar(x + j * w, vals, w, label=k, color=PALETTE[j % len(PALETTE)])
    ax.set_xticks(x + w * (len(keys) - 1) / 2)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison (test set)", pad=28)
    ax.legend(ncol=len(keys), fontsize=8, loc="lower center",
              bbox_to_anchor=(0.5, 1.03), frameon=False)
    return _save(fig, name)


def plot_learning_curve(train_sizes, train_scores, val_scores,
                        title: str, name: str) -> str:
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    tr_m, tr_s = train_scores.mean(1), train_scores.std(1)
    va_m, va_s = val_scores.mean(1), val_scores.std(1)
    ax.plot(train_sizes, tr_m, "o-", color=PALETTE[0], label="Train")
    ax.fill_between(train_sizes, tr_m - tr_s, tr_m + tr_s, alpha=0.15,
                    color=PALETTE[0])
    ax.plot(train_sizes, va_m, "o-", color=PALETTE[1], label="Cross-val")
    ax.fill_between(train_sizes, va_m - va_s, va_m + va_s, alpha=0.15,
                    color=PALETTE[1])
    ax.set_xlabel("Training examples")
    ax.set_ylabel("F1 score")
    ax.set_title(title)
    ax.legend(loc="best")
    return _save(fig, name)


def plot_ablation(ablation: Dict[str, Dict[str, float]],
                  metric: str = "f1",
                  name: str = "feature_ablation.png") -> str:
    sets = list(ablation.keys())
    vals = [ablation[s][metric] for s in sets]
    fig, ax = plt.subplots(figsize=(1.5 * len(sets) + 2, 4.2))
    ax.bar(sets, vals, color=PALETTE[2])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Feature-set ablation ({metric.upper()}, best model)")
    ax.set_xticklabels(sets, rotation=15, ha="right")
    return _save(fig, name)


def plot_class_balance(labels, name: str = "class_balance.png") -> str:
    fig, ax = plt.subplots(figsize=(4, 4))
    vals = [int((labels == 0).sum()), int((labels == 1).sum())]
    ax.bar(["No Flare", "Flare"], vals, color=[PALETTE[7], PALETTE[3]])
    for i, v in enumerate(vals):
        ax.text(i, v, str(v), ha="center", va="bottom")
    ax.set_ylabel("Samples")
    ax.set_title("Class balance (all samples)")
    return _save(fig, name)
