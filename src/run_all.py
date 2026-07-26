"""Run the whole Iteration 4 pipeline end to end.

    python -m src.run_all

Steps: build features (if needed) -> train/tune/evaluate tabular models ->
CNN baseline -> generate REPORT.md.  Safe to re-run; each step overwrites
its own artifacts.
"""
from __future__ import annotations

import os

from . import config


def main(rebuild: bool = False):
    config.ensure_dirs()

    if rebuild or not os.path.exists(config.FEATURES_CSV):
        from . import build_dataset
        build_dataset.main()
    else:
        print(f"Using existing {config.FEATURES_CSV} (pass rebuild=True to redo).")

    from . import train
    train.main()

    from . import cnn
    cnn.main()

    from . import make_report
    make_report.main()

    print("\nAll done. See REPORT.md and report/figures/.")


if __name__ == "__main__":
    import sys
    main(rebuild="--rebuild" in sys.argv)
