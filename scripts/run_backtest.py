"""Run the full v2 walk-forward backtest and print the results table.

Usage:
    python scripts/run_backtest.py [--cost-bps 5] [--out-dir DIR] [--prices-csv FILE]

With --out-dir, writes prices.csv, regimes.csv, strategy.csv, fits.csv and
metrics.json there so two runs can be compared file by file.
"""
import argparse
import json
from dataclasses import asdict
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src  # noqa: E402,F401  (pins BLAS threads before numpy loads)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from threadpoolctl import threadpool_info  # noqa: E402

from src.backtest import DEFAULT_COST_BPS, apply_rule, walk_forward  # noqa: E402
from src.data import fetch_prices  # noqa: E402
from src.features import compute_features  # noqa: E402
from src.metrics import load_nber, nber_check, performance  # noqa: E402

log = logging.getLogger("run_backtest")


def run(cost_bps: float, prices_csv: Path | None = None):
    if prices_csv:
        prices = pd.read_csv(prices_csv, index_col="Date", parse_dates=["Date"])
    else:
        prices = fetch_prices()
    feat = compute_features(prices)
    regimes, fits = walk_forward(feat)
    nber = load_nber(ROOT / "data" / "nber_recessions.csv")

    strat = apply_rule(regimes, feat["Returns"], cost_bps)
    metrics = {"buy_and_hold": performance(strat["bh_ret"])}
    for kind in ("gross", "net"):
        metrics[f"strategy_{kind}"] = performance(strat[f"{kind}_ret"], strat["position"], strat["switch"])

    eval_labels = regimes["label"].reindex(strat.index)
    summary = {
        "data": {
            "first_price": str(prices.index[0].date()),
            "last_price": str(prices.index[-1].date()),
            "feature_rows": len(feat),
            "eval_start": str(strat.index[0].date()),
            "eval_end": str(strat.index[-1].date()),
            "eval_days": len(strat),
        },
        "cost_bps": cost_bps,
        "fits": {
            "refits": len(fits),
            "failed": sum(not f.ok for f in fits),
            "hit_max_iter": sum(f.hit_max_iter for f in fits),
            "with_nonmonotonic_warnings": sum(f.nonmonotonic_warnings > 0 for f in fits),
            "mean_iter": float(np.mean([f.n_iter for f in fits if f.ok])),
            "best_seed_not_42": sum(f.ok and f.best_seed != 42 for f in fits),
            "best_seed_counts": pd.Series([f.best_seed for f in fits if f.ok]).value_counts().sort_index().to_dict(),
            "failed_seed_fits": sum(f.failed_seeds for f in fits),
            "loglik_spread_median": float(np.nanmedian([f.loglik_spread for f in fits])),
            "loglik_spread_max": float(np.nanmax([f.loglik_spread for f in fits])),
        },
        "label_share": eval_labels.value_counts(normalize=True).to_dict(),
        "metrics": metrics,
        "nber": nber_check(eval_labels, nber),
    }
    return prices, regimes, strat, fits, summary


def print_table(summary):
    m = pd.DataFrame(summary["metrics"]).T
    fmt = {
        "sharpe": "{:.2f}", "ann_vol": "{:.2%}", "total_return": "{:.1%}", "cagr": "{:.2%}",
        "max_drawdown": "{:.2%}", "switches": "{:.0f}", "pct_invested": "{:.1%}",
    }
    for col, f in fmt.items():
        m[col] = m[col].map(f.format)
    print(m.to_string())
    print()
    print(json.dumps({k: v for k, v in summary.items() if k != "metrics"}, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--prices-csv", type=Path,
                        help="use prices saved by a previous --out-dir run instead of fetching")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    blas = {(p["internal_api"], p["num_threads"]) for p in threadpool_info()}
    log.info("Thread pools (api, threads): %s", sorted(blas))

    prices, regimes, strat, fits, summary = run(args.cost_bps, args.prices_csv)
    print_table(summary)

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        prices.to_csv(args.out_dir / "prices.csv", float_format="%.17g")
        regimes.to_csv(args.out_dir / "regimes.csv", float_format="%.17g")
        strat.to_csv(args.out_dir / "strategy.csv", float_format="%.17g")
        pd.DataFrame([asdict(f) for f in fits]).to_csv(args.out_dir / "fits.csv", index=False,
                                                       float_format="%.17g")
        (args.out_dir / "metrics.json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
