"""Central configuration for the Solar Flare Detector pipeline.

Everything that another script might want to tweak (paths, the flare
threshold, image size, the solar-cycle anchor) lives here so the rest of
the code stays free of magic numbers.
"""
from __future__ import annotations

import os

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")
FIGURES_DIR = os.path.join(ROOT, "report", "figures")

FEATURES_CSV = os.path.join(ARTIFACTS_DIR, "features.csv")

# ----------------------------------------------------------------------
# The prediction target
# ----------------------------------------------------------------------
# GOES soft X-ray flux classes (peak flux, W/m^2):
#   A < 1e-7 , B 1e-7..1e-6 , C 1e-6..1e-5 , M 1e-5..1e-4 , X >= 1e-4
#
# We frame the primary task as binary classification: "will an active
# region produce at least a C-class flare in the next 24 h?".  The
# C-class line (1e-6) is the standard "flare vs. quiet" boundary and
# leaves a learnable (though still imbalanced) positive class.  Set to
# 1e-5 to reproduce the harder M-class task the original notebook used.
FLARE_THRESHOLD = 1e-6

# Floor applied before log10 so quiet regions (peak_flux ~ 0) don't blow
# up to -inf.  Matches the ~A-class floor used previously.
FLUX_FLOOR = 1e-9

# ----------------------------------------------------------------------
# Image handling
# ----------------------------------------------------------------------
# All 10 channels SDOBenchmark provides.  Order matters only for
# reproducibility of the feature columns.
WAVELENGTHS = [
    "94", "131", "171", "193", "211", "304", "335", "1700",
    "continuum", "magnetogram",
]

# Resolution we resize every patch to before computing statistics.  The
# original patches are small; 64 keeps texture/gradient information while
# staying cheap on CPU.
IMG_SIZE = 64

# ----------------------------------------------------------------------
# Solar cycle encoding
# ----------------------------------------------------------------------
# Solar activity follows an ~11-year cycle, so the raw calendar year is
# not directly meaningful.  We encode the date as a (sin, cos) phase
# within the cycle, anchored at a cycle minimum.  2008.9 ~= the minimum
# between Solar Cycles 23 and 24.
SOLAR_CYCLE_LENGTH = 11.0
SOLAR_CYCLE_EPOCH = 2008.9

# ----------------------------------------------------------------------
# Train / val / test split (grouped by active region to avoid leakage)
# ----------------------------------------------------------------------
TEST_SIZE = 0.20
VAL_SIZE = 0.20  # fraction of the *remaining* train pool
RANDOM_STATE = 42


def ensure_dirs() -> None:
    """Create output directories if they don't exist yet."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
