"""Build the tabular feature table from the raw image dataset.

Run once after the data is in place:

    python -m src.build_dataset

Writes ``artifacts/features.csv`` (features + id/active_region/peak_flux/
log10_flux/label), which every model script reads instead of re-touching
the tens of thousands of raw images.
"""
from __future__ import annotations

import time

import pandas as pd

from . import config
from .data import discover_samples
from .features import build_feature_frame


def main() -> None:
    config.ensure_dirs()

    t0 = time.time()
    print("Discovering samples under", config.DATA_DIR, "...")
    samples = discover_samples()
    print(f"Found {len(samples)} samples "
          f"across {len({s.active_region for s in samples})} active regions.")

    # A quick look at how the flare threshold splits the classes, at a few
    # candidate cutoffs, so we can sanity-check the balance.
    fluxes = pd.Series([s.peak_flux for s in samples])
    for thr, name in [(1e-6, "C"), (1e-5, "M"), (1e-4, "X")]:
        pos = (fluxes >= thr).mean()
        print(f"  >= {name}-class ({thr:.0e}): {pos*100:5.1f}% positive")

    print("\nExtracting features ...")
    X, meta = build_feature_frame(samples)
    df = pd.concat([meta, X], axis=1)

    df.to_csv(config.FEATURES_CSV, index=False)
    print(f"\nWrote {config.FEATURES_CSV}  "
          f"({df.shape[0]} rows x {X.shape[1]} features)")
    print(f"Positive rate at active threshold "
          f"({config.FLARE_THRESHOLD:.0e}): {meta['label'].mean()*100:.1f}%")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
