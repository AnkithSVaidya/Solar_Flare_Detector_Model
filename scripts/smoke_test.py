"""End-to-end smoke test on SYNTHETIC data.

Generates a tiny SDOBenchmark-shaped dataset (with a real, learnable signal
in the magnetogram) in a temp folder, points the config at it, and runs the
whole pipeline: feature build -> train/tune -> CNN -> report. This validates
every code path so the real run "just works" once the true dataset lands.

    python scripts/smoke_test.py

Nothing here touches the real data/, artifacts/ or report/ folders.
"""
import os
import sys
import shutil
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import config  # noqa: E402

SMOKE = os.path.join(ROOT, "._smoke")
WLS = config.WAVELENGTHS


def _make_image(path, strength, wl):
    """32x32 grayscale. Magnetogram amplitude scales with `strength` so the
    label (derived from strength) is genuinely predictable."""
    rng = np.random.default_rng(abs(hash(path)) % (2**32))
    base = rng.normal(0.5, 0.05, (32, 32))
    if wl == "magnetogram":
        amp = 0.15 + 0.35 * strength
        blob = np.zeros((32, 32))
        cx, cy = rng.integers(8, 24, size=2)
        yy, xx = np.mgrid[0:32, 0:32]
        blob += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 20)
        blob -= amp * np.exp(-((xx - cx - 6) ** 2 + (yy - cy) ** 2) / 20)
        img = np.clip(0.5 + blob + rng.normal(0, 0.03, (32, 32)), 0, 1)
    else:
        img = np.clip(base + 0.1 * strength, 0, 1)
    Image.fromarray((img * 255).astype(np.uint8)).save(path)


def generate(n_ar=60):
    train_root = os.path.join(SMOKE, "data", "training")
    os.makedirs(train_root, exist_ok=True)
    rng = np.random.default_rng(0)
    meta_rows = []
    for a in range(n_ar):
        ar = str(11000 + a)
        for k in range(rng.integers(1, 3)):
            strength = float(rng.uniform(0, 1))
            log_flux = -9 + 5 * strength + rng.normal(0, 0.2)
            peak_flux = 10 ** log_flux
            sample = f"2012_{a:02d}_{k:02d}_00_00_00_0"
            folder = os.path.join(train_root, ar, sample)
            os.makedirs(folder, exist_ok=True)
            for ts in range(4):
                stamp = f"2012-01-{ts+1:02d}T000000"
                for wl in WLS:
                    _make_image(os.path.join(folder, f"{stamp}__{wl}.jpg"),
                                strength, wl)
            meta_rows.append({"id": f"{ar}_{sample}", "peak_flux": peak_flux})
    import pandas as pd
    pd.DataFrame(meta_rows).to_csv(
        os.path.join(train_root, "meta_data.csv"), index=False)
    print(f"Generated {len(meta_rows)} synthetic samples across {n_ar} ARs.")


def main():
    if os.path.exists(SMOKE):
        shutil.rmtree(SMOKE)
    # Redirect every output path into the temp sandbox.
    config.DATA_DIR = os.path.join(SMOKE, "data")
    config.ARTIFACTS_DIR = os.path.join(SMOKE, "artifacts")
    config.FIGURES_DIR = os.path.join(SMOKE, "figures")
    config.FEATURES_CSV = os.path.join(config.ARTIFACTS_DIR, "features.csv")
    config.ROOT = SMOKE
    config.ensure_dirs()

    generate()

    from src import build_dataset, train, cnn, make_report
    print("\n--- build_dataset ---");  build_dataset.main()
    print("\n--- train ---");          train.main()
    print("\n--- cnn ---");            cnn.main()
    print("\n--- make_report ---");    make_report.main()

    figs = os.listdir(config.FIGURES_DIR)
    print(f"\nFigures produced ({len(figs)}):", sorted(figs))
    assert os.path.exists(os.path.join(SMOKE, "REPORT.md"))
    print("\nSMOKE TEST PASSED")

    shutil.rmtree(SMOKE)
    print("Cleaned up temp sandbox.")


if __name__ == "__main__":
    main()
