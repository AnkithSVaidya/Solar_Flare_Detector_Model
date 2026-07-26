"""Dataset discovery and image loading for SDOBenchmark.

The on-disk layout (created by ``load_data.py`` / kagglehub) is:

    data/<split>/<NOAA_AR>/<sample_datetime>/<timestamp>__<wavelength>.jpg
    data/<split>/meta_data.csv        # columns include id, peak_flux

where a meta_data ``id`` looks like ``11390_2012_01_05_17_06_01_0`` — the
NOAA active-region number, an underscore, then the sample-folder name.

This module turns that tree into a tidy list of ``Sample`` records, each
pointing at its images (grouped by wavelength, ordered in time) and
carrying its peak flux and date.  Everything downstream consumes
``discover_samples()`` and never touches the raw folder layout again.
"""
from __future__ import annotations

import os
import re
import glob
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image

from . import config


# ----------------------------------------------------------------------
# Record types
# ----------------------------------------------------------------------
@dataclass
class Sample:
    sample_id: str                       # full meta id, e.g. 11390_2012_..._0
    active_region: str                   # NOAA AR number (the group key)
    folder: str                          # absolute path to the sample folder
    peak_flux: float                     # regression target (W/m^2)
    year: float                          # decimal-ish year parsed from the id
    # wavelength -> list of image file paths, ordered by timestamp
    images: Dict[str, List[str]] = field(default_factory=dict)


# ----------------------------------------------------------------------
# Parsing helpers
# ----------------------------------------------------------------------
_TS_RE = re.compile(r"(\d{4})[-_](\d{2})[-_](\d{2})")


def parse_year_from_id(sample_id: str) -> float:
    """The year is the second underscore-separated field of the id
    (``11390_2012_...`` -> 2012).  Falls back to the cycle epoch."""
    parts = str(sample_id).split("_")
    if len(parts) >= 2:
        try:
            return float(int(parts[1]))
        except ValueError:
            pass
    return config.SOLAR_CYCLE_EPOCH


def wavelength_of(filename: str) -> Optional[str]:
    """Return which channel a file belongs to, or None.

    Files are named ``<timestamp>__<wavelength>.jpg``.  We match on the
    suffix so that '171' doesn't accidentally match inside '1710'-style
    names (it won't here, but be defensive)."""
    name = os.path.basename(filename).lower()
    if not name.endswith((".jpg", ".jpeg", ".png")):
        return None
    stem = os.path.splitext(name)[0]
    token = stem.split("__")[-1] if "__" in stem else stem
    for wl in config.WAVELENGTHS:
        if token == wl.lower() or token.endswith(wl.lower()):
            return wl
    return None


def _timestamp_key(path: str) -> str:
    """Sort key that orders a wavelength's frames chronologically."""
    return os.path.basename(path)


# ----------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------
def _find_meta_files(root: str) -> List[str]:
    return glob.glob(os.path.join(root, "**", "meta_data.csv"), recursive=True)


def discover_samples(data_dir: Optional[str] = None,
                     split: Optional[str] = None) -> List[Sample]:
    """Walk the dataset and return one ``Sample`` per active-region patch.

    ``split`` optionally restricts to a sub-tree ("training"/"test"); by
    default every meta_data.csv found under ``data_dir`` is used.
    """
    data_dir = data_dir or config.DATA_DIR
    if split:
        data_dir = os.path.join(data_dir, split)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            "Download the SDOBenchmark dataset and place it under data/ "
            "(see README)."
        )

    samples: List[Sample] = []
    meta_files = _find_meta_files(data_dir)
    if not meta_files:
        raise FileNotFoundError(
            f"No meta_data.csv found anywhere under {data_dir}. "
            "The dataset may not be extracted correctly."
        )

    for meta_path in meta_files:
        split_root = os.path.dirname(meta_path)  # e.g. data/training
        meta = pd.read_csv(meta_path)
        if "id" not in meta.columns or "peak_flux" not in meta.columns:
            continue

        for _, row in meta.iterrows():
            sid = str(row["id"])
            # id = "<AR>_<sample_datetime>"  ->  folder = <split>/<AR>/<rest>
            ar, _, rest = sid.partition("_")
            folder = os.path.join(split_root, ar, rest)
            if not os.path.isdir(folder):
                # Some releases nest the sample folder directly; try a
                # recursive fallback before giving up on this row.
                hits = glob.glob(os.path.join(split_root, ar, "**", rest),
                                 recursive=True)
                folder = hits[0] if hits else None
            if not folder or not os.path.isdir(folder):
                continue

            images: Dict[str, List[str]] = {}
            for f in os.listdir(folder):
                wl = wavelength_of(f)
                if wl is None:
                    continue
                images.setdefault(wl, []).append(os.path.join(folder, f))
            for wl in images:
                images[wl].sort(key=_timestamp_key)

            if not images:
                continue

            try:
                flux = float(row["peak_flux"])
            except (TypeError, ValueError):
                continue

            samples.append(Sample(
                sample_id=sid,
                active_region=ar,
                folder=folder,
                peak_flux=flux,
                year=parse_year_from_id(sid),
                images=images,
            ))

    return samples


# ----------------------------------------------------------------------
# Image loading
# ----------------------------------------------------------------------
def load_gray(path: str, size: int = config.IMG_SIZE) -> np.ndarray:
    """Load an image as a float32 array in [0, 1], resized to size x size."""
    img = Image.open(path).convert("L").resize((size, size))
    return np.asarray(img, dtype=np.float32) / 255.0


# ----------------------------------------------------------------------
# Solar-cycle features
# ----------------------------------------------------------------------
def year_to_cycle_features(year: float) -> Dict[str, float]:
    """Map a year to its (sin, cos) phase within the ~11-year cycle."""
    phase = ((year - config.SOLAR_CYCLE_EPOCH) % config.SOLAR_CYCLE_LENGTH) \
        / config.SOLAR_CYCLE_LENGTH
    angle = 2.0 * np.pi * phase
    return {
        "cycle_sin": float(np.sin(angle)),
        "cycle_cos": float(np.cos(angle)),
        "year": float(year),
    }
