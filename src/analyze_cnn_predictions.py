"""Real analysis of the EXISTING CNN regressor, from test_flux_predictions.csv.

This does not need the image dataset — it works from the predictions the
earlier CNN already wrote to disk, so it produces genuine (not synthetic)
Iteration-4 metrics and figures for the baseline model while the full
multi-model run waits on the image download.

    python -m src.analyze_cnn_predictions

Outputs real numbers to artifacts/cnn_regressor_analysis.json and figures
to report/figures/cnn_*.png.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve,
)

from . import config, plots

CSV = os.path.join(config.ROOT, "test_flux_predictions.csv")


def _class_metrics(y_true, y_pred, y_score):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_score)) if len(set(y_true)) > 1 else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def main():
    config.ensure_dirs()
    if not os.path.exists(CSV):
        raise SystemExit(f"{CSV} not found.")
    df = pd.read_csv(CSV)
    yt = df["true_log10_flux"].values
    yp = df["pred_log10_flux"].values

    # --- Regression metrics (the model's native task) ---
    regression = {
        "mae": float(mean_absolute_error(yt, yp)),
        "mse": float(mean_squared_error(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "r2": float(r2_score(yt, yp)),
        "n": int(len(df)),
    }
    print("Regression:", json.dumps(regression, indent=2))

    # --- Classification by thresholding the flux, at C- and M-class ---
    classification = {}
    for thr, cls in [(1e-6, "C"), (1e-5, "M")]:
        log_thr = np.log10(thr)
        y_true = (yt >= log_thr).astype(int)
        y_pred = (yp >= log_thr).astype(int)
        m = _class_metrics(y_true, y_pred, yp)  # continuous pred as score
        m["positive_rate"] = float(y_true.mean())
        classification[f"{cls}-class(>={thr:.0e})"] = m
        print(f"\n>= {cls}-class threshold: "
              f"acc={m['accuracy']:.3f} prec={m['precision']:.3f} "
              f"rec={m['recall']:.3f} f1={m['f1']:.3f} auc={m['roc_auc']:.3f}")
        # Figures for the C-class operating point (the primary framing)
        if cls == "C":
            plots.plot_confusion(y_true, y_pred,
                                 "CNN regressor @ C-class (test)",
                                 "cnn_reg_confusion.png")
            plots.plot_roc({"CNN regressor": (y_true, yp)},
                           "cnn_reg_roc.png")
            plots.plot_pr({"CNN regressor": (y_true, yp)},
                          "cnn_reg_pr.png")

    # --- Predicted vs true scatter (regression view) ---
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(yt, yp, s=6, alpha=0.15, color=plots.PALETTE[0])
    lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
    ax.plot(lims, lims, "k--", lw=1, label="ideal")
    ax.set_xlabel("True log10 flux"); ax.set_ylabel("Predicted log10 flux")
    ax.set_title("CNN regressor: predicted vs. true"); ax.legend()
    fig.savefig(os.path.join(config.FIGURES_DIR, "cnn_reg_scatter.png"),
                dpi=150, bbox_inches="tight"); plt.close(fig)

    # --- Threshold sweep (recall/precision trade-off) at C-class labels ---
    y_true_c = (yt >= np.log10(1e-6)).astype(int)
    prec, rec, thr_pr = precision_recall_curve(y_true_c, yp)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(thr_pr, prec[:-1], label="precision", color=plots.PALETTE[0])
    ax.plot(thr_pr, rec[:-1], label="recall", color=plots.PALETTE[1])
    f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    ax.plot(thr_pr, f1s, label="F1", color=plots.PALETTE[2])
    ax.axvline(np.log10(1e-6), color="r", ls=":", label="C-class threshold")
    ax.set_xlabel("Decision threshold (log10 flux)")
    ax.set_ylabel("Score"); ax.set_title("CNN regressor: threshold trade-off")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(config.FIGURES_DIR, "cnn_reg_threshold.png"),
                dpi=150, bbox_inches="tight"); plt.close(fig)

    out = {"regression": regression, "classification": classification,
           "note": ("Computed on the augmentation-expanded test set the CNN "
                    "wrote; rows are correlated so treat as baseline-model "
                    "behaviour, not an independent-sample estimate.")}
    with open(os.path.join(config.ARTIFACTS_DIR,
                           "cnn_regressor_analysis.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nWrote artifacts/cnn_regressor_analysis.json and "
          "report/figures/cnn_reg_*.png")
    return out


if __name__ == "__main__":
    main()
