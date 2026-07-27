"""Build the tabular feature table from the raw image dataset.

Run once after the data is in place:

    python -m src.build_dataset

Uses SDOBenchmark's **official** train/test split (the two folders have
zero active-region overlap, so it's a clean, leakage-free split) and tags
each row with a ``split`` column.  Falls back to a single pool if the
``test`` folder isn't present.  Writes ``artifacts/features.csv``.
"""
from __future__ import annotations

import os
import time

import pandas as pd

from . import config
from .data import discover_samples
from .features import build_feature_frame


def _discover_split(split):
    try:
        return discover_samples(split=split)
    except FileNotFoundError:
        return []


def main() -> None:
    config.ensure_dirs()
    t0 = time.time()
    print("Dataset root:", config.DATASET_DIR)

    train_samples = _discover_split("training")
    test_samples = _discover_split("test")

    if not train_samples and not test_samples:
        # No official split folders -> discover everything as one pool.
        train_samples = discover_samples()
        print("No training/test folders found; using a single pool "
              "(train.py will make a grouped split).")

    print(f"Train samples: {len(train_samples)}  Test samples: {len(test_samples)}")

    fluxes = pd.Series([s.peak_flux for s in train_samples + test_samples])
    for thr, name in [(1e-6, "C"), (1e-5, "M"), (1e-4, "X")]:
        print(f"  >= {name}-class ({thr:.0e}): "
              f"{(fluxes >= thr).mean()*100:5.1f}% positive")

    frames = []
    for split_name, samples in [("train", train_samples), ("test", test_samples)]:
        if not samples:
            continue
        print(f"\nExtracting features for '{split_name}' ...")
        X, meta = build_feature_frame(samples)
        part = pd.concat([meta, X], axis=1)
        part.insert(0, "split", split_name)
        frames.append(part)

    df = pd.concat(frames, axis=0, ignore_index=True)
    n_feat = df.shape[1] - 6  # split + 5 meta cols
    df.to_csv(config.FEATURES_CSV, index=False)

    print(f"\nWrote {config.FEATURES_CSV}  ({df.shape[0]} rows x {n_feat} features)")
    print(f"Positive rate at active threshold "
          f"({config.FLARE_THRESHOLD:.0e}): {df['label'].mean()*100:.1f}%")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
