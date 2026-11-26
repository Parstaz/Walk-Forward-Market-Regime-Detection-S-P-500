"""Append new trading days to the saved history.

Exit code 0: updated, or no new trading day (nothing written).
Exit code 1: a fetch or check failed; existing files are untouched.

Usage:
    python scripts/daily_update.py [--data-dir data] [--models-dir models]
"""
import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src  # noqa: E402,F401  (pins BLAS threads before numpy loads)
from src.pipeline import Paths, daily_update  # noqa: E402

log = logging.getLogger("daily_update")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        status = daily_update(Paths(args.data_dir, args.models_dir))
    except Exception:
        log.exception("Daily update failed; existing data left untouched")
        return 1
    log.info("Done: %s", status)
    return 0


if __name__ == "__main__":
    sys.exit(main())
