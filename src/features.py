"""Feature engineering: turn a Sample's images into interpretable scalars.

For each wavelength we summarise every frame with a handful of physically
meaningful statistics, then collapse the (up to 4) timesteps into:

  * ``_last``  -- the value at the frame closest to the prediction time
  * ``_delta`` -- last minus first (captures how the region is *changing*,
                  which is what actually precedes a flare)

The magnetogram gets extra physics-inspired features (total unsigned flux
and polarity-inversion-line gradient energy) because the line-of-sight
magnetic field is the single most flare-relevant channel.  Finally we add
the 11-year solar-cycle phase.  The result is one flat dict per sample.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy import ndimage

from . import config
from .data import Sample, load_gray, year_to_cycle_features


# ----------------------------------------------------------------------
# Per-frame statistics
# ----------------------------------------------------------------------
def _gradient_energy(arr: np.ndarray) -> float:
    """Mean Sobel-gradient magnitude -- high where there are sharp
    intensity boundaries (e.g. polarity inversion lines / bright edges)."""
    gx = ndimage.sobel(arr, axis=0)
    gy = ndimage.sobel(arr, axis=1)
    return float(np.mean(np.hypot(gx, gy)))


def frame_stats(arr: np.ndarray) -> Dict[str, float]:
    """Summary statistics for one image frame (values in [0, 1])."""
    active_thr = arr.mean() + 2.0 * arr.std()
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p95": float(np.percentile(arr, 95)),
        "active_frac": float(np.mean(arr > active_thr)),
        "grad": _gradient_energy(arr),
    }


def magnetogram_extras(arr: np.ndarray) -> Dict[str, float]:
    """Extra features specific to the (signed) magnetogram.

    In SDOBenchmark magnetograms, mid-grey (~0.5) is zero field, bright is
    positive polarity and dark is negative.  So |x - 0.5| is proportional
    to field strength.
    """
    signed = arr - 0.5
    total_unsigned_flux = float(np.mean(np.abs(signed)))   # ~ total |B|
    pos_frac = float(np.mean(signed > 0.1))
    neg_frac = float(np.mean(signed < -0.1))
    # Strong-gradient pixels straddling opposite polarities approximate the
    # polarity inversion line, a known flare precursor.
    pil_energy = _gradient_energy(np.abs(signed))
    return {
        "mag_total_unsigned_flux": total_unsigned_flux,
        "mag_pos_frac": pos_frac,
        "mag_neg_frac": neg_frac,
        "mag_flux_imbalance": abs(pos_frac - neg_frac),
        "mag_pil_energy": pil_energy,
    }


# ----------------------------------------------------------------------
# Temporal collapse
# ----------------------------------------------------------------------
def _collapse_time(per_frame: List[Dict[str, float]],
                   prefix: str) -> Dict[str, float]:
    """Turn a chronological list of per-frame stat dicts into ``_last`` and
    ``_delta`` features."""
    out: Dict[str, float] = {}
    if not per_frame:
        return out
    keys = per_frame[0].keys()
    first, last = per_frame[0], per_frame[-1]
    for k in keys:
        out[f"{prefix}_{k}_last"] = last[k]
        out[f"{prefix}_{k}_delta"] = last[k] - first[k]
    return out


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def extract_features(sample: Sample) -> Dict[str, float]:
    """Full feature dict for one sample (NaN where a channel is missing)."""
    feats: Dict[str, float] = {}

    for wl in config.WAVELENGTHS:
        paths = sample.images.get(wl, [])
        per_frame: List[Dict[str, float]] = []
        mag_frames: List[Dict[str, float]] = []
        for p in paths:
            try:
                arr = load_gray(p)
            except Exception:
                continue
            per_frame.append(frame_stats(arr))
            if wl == "magnetogram":
                mag_frames.append(magnetogram_extras(arr))

        feats.update(_collapse_time(per_frame, prefix=wl))
        feats[f"{wl}_n_frames"] = float(len(per_frame))
        if wl == "magnetogram":
            feats.update(_collapse_time(mag_frames, prefix="magx"))

    # Solar-cycle phase + raw year
    feats.update(year_to_cycle_features(sample.year))
    return feats


def build_feature_frame(samples: List[Sample]):
    """Vectorise ``extract_features`` over all samples.

    Returns (X_df, meta_df) where meta_df carries id / active_region /
    peak_flux / label so the training script can group-split and label.
    """
    import pandas as pd

    rows, meta = [], []
    n = len(samples)
    for i, s in enumerate(samples):
        rows.append(extract_features(s))
        meta.append({
            "id": s.sample_id,
            "active_region": s.active_region,
            "peak_flux": s.peak_flux,
            "log10_flux": float(np.log10(max(s.peak_flux, config.FLUX_FLOOR))),
            "label": int(s.peak_flux >= config.FLARE_THRESHOLD),
        })
        if (i + 1) % 100 == 0 or (i + 1) == n:
            print(f"  extracted {i + 1}/{n} samples", flush=True)

    X = pd.DataFrame(rows)
    meta_df = pd.DataFrame(meta)
    return X, meta_df
