"""A clean, honest CNN baseline: magnetogram image -> flare / no-flare.

This is the "deep learning on raw pixels" comparison point for the report.
It deliberately fixes two problems with the original notebook:

  * No 50x augmentation explosion.  One row per real sample (the last
    magnetogram frame, i.e. the one closest to the prediction window),
    with light on-the-fly augmentation applied to the *training* split
    only.  That keeps train/test honest and CPU-tractable.
  * Grouped split by active region, so no region leaks across splits.

Run:  python -m src.cnn
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import config, plots
from .data import discover_samples, load_gray

CNN_IMG = 32
AUG_PER_SAMPLE = 4
EPOCHS = 40
BATCH = 64
LR = 1e-3


def _load_last_magnetograms(size=CNN_IMG):
    """One magnetogram per sample (last timestep) + label + group."""
    samples = discover_samples()
    X, y, groups = [], [], []
    for s in samples:
        paths = s.images.get("magnetogram", [])
        if not paths:
            continue
        try:
            arr = load_gray(paths[-1], size=size)
        except Exception:
            continue
        X.append(arr)
        y.append(int(s.peak_flux >= config.FLARE_THRESHOLD))
        groups.append(s.active_region)
    return np.asarray(X, np.float32), np.asarray(y, np.int64), np.asarray(groups)


def _augment(a):
    from PIL import Image
    im = Image.fromarray((a * 255).astype(np.uint8))
    im = im.rotate(np.random.uniform(-25, 25), resample=Image.BILINEAR)
    dx, dy = np.random.randint(-3, 4), np.random.randint(-3, 4)
    im = im.transform(im.size, Image.AFFINE, (1, 0, dx, 0, 1, dy))
    out = np.asarray(im, np.float32) / 255.0
    out = np.clip(out + np.random.normal(0, 0.05, out.shape), 0, 1)
    return out.astype(np.float32)


def main():
    try:
        import torch
        import torch.nn as nn
    except Exception as e:  # pragma: no cover
        print("PyTorch unavailable, skipping CNN baseline:", e)
        return None

    from sklearn.model_selection import GroupShuffleSplit
    config.ensure_dirs()
    torch.manual_seed(config.RANDOM_STATE)
    np.random.seed(config.RANDOM_STATE)

    X, y, groups = _load_last_magnetograms()
    print(f"CNN dataset: {len(X)} magnetograms, {y.mean()*100:.1f}% positive")

    gss = GroupShuffleSplit(1, test_size=config.TEST_SIZE,
                            random_state=config.RANDOM_STATE)
    tr, te = next(gss.split(X, y, groups))

    # Light augmentation of the training split only.
    Xtr = [X[i] for i in tr]
    ytr = [y[i] for i in tr]
    aug_X, aug_y = list(Xtr), list(ytr)
    for i in tr:
        for _ in range(AUG_PER_SAMPLE):
            aug_X.append(_augment(X[i]))
            aug_y.append(int(y[i]))
    Xtr = np.asarray(aug_X, np.float32)[:, None, :, :]
    ytr = np.asarray(aug_y, np.float32)[:, None]
    Xte = X[te][:, None, :, :]
    yte = y[te]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            )
            self.h = nn.Sequential(nn.Linear(64, 64), nn.ReLU(),
                                   nn.Dropout(0.4), nn.Linear(64, 1))

        def forward(self, x):
            return self.h(self.f(x))

    net = Net().to(device)
    pos_w = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)],
                         dtype=torch.float32, device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-4)

    Xtr_t = torch.tensor(Xtr, device=device)
    ytr_t = torch.tensor(ytr, device=device)
    n = len(Xtr_t)
    for epoch in range(EPOCHS):
        net.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = crit(net(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1:2d}/{EPOCHS}  loss {tot/n:.4f}")

    net.eval()
    with torch.no_grad():
        logits = net(torch.tensor(Xte, device=device)).cpu().numpy().ravel()
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)

    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score, confusion_matrix)
    tn, fp, fn, tp = confusion_matrix(yte, preds, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": float(accuracy_score(yte, preds)),
        "precision": float(precision_score(yte, preds, zero_division=0)),
        "recall": float(recall_score(yte, preds, zero_division=0)),
        "f1": float(f1_score(yte, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(yte, probs)) if len(set(yte)) > 1 else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    print("CNN test metrics:", json.dumps(metrics, indent=2))

    plots.plot_confusion(yte, preds, "Confusion Matrix — CNN (test)",
                         "confusion_cnn.png")
    with open(os.path.join(config.ARTIFACTS_DIR, "cnn_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    try:
        import torch as _t
        _t.save(net.state_dict(),
                os.path.join(config.ARTIFACTS_DIR, "cnn_classifier.pt"))
    except Exception:
        pass
    return metrics


if __name__ == "__main__":
    main()
