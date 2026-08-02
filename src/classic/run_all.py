"""Run the whole Iteration 4 pipeline end to end.

    python -m src.run_all

Steps: build features (if needed) -> train/tune/evaluate tabular models ->
CNN baseline.  Produces the metrics and figures in artifacts/ and
report/figures/.  Safe to re-run; each step overwrites its own artifacts.
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

    print("\nAll done. Results in artifacts/ and report/figures/; "
          "see REPORT.md for the write-up.")


if __name__ == "__main__":
    import sys
    main(rebuild="--rebuild" in sys.argv)
