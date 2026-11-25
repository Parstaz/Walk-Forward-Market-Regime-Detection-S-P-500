"""Run the full walk-forward once and save data/ and models/ from scratch.

Usage:
    python scripts/build_history.py [--data-dir data] [--models-dir models] [--cost-bps 5]
"""
import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src  # noqa: E402,F401  (pins BLAS threads before numpy loads)
from src.backtest import DEFAULT_COST_BPS  # noqa: E402
from src.pipeline import Paths, build_history  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    build_history(Paths(args.data_dir, args.models_dir), cost_bps=args.cost_bps)


if __name__ == "__main__":
    main()
