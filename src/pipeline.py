"""Saved state: full history build and the daily incremental update.

Daily log returns in data/returns.parquet are the source of truth.
yfinance's adjusted closes are rescaled over the whole history after every
dividend, so saved prices can't be extended with newly downloaded ones.
Log returns are unaffected by that rescaling (except on the ex-date
itself, which is exactly what the overlap check guards). prices.parquet is
rebuilt from returns for charting only.
"""
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from .backtest import (BURN_IN, DEFAULT_COST_BPS, PROB_COLS, REFIT_EVERY, FitRecord, apply_rule,
                       invested, walk_forward)
from .data import START, TICKER, fetch_prices
from .features import FEATURES, VOL_WINDOW, compute_features, log_returns
from .metrics import load_nber, nber_check, performance
from .model import HMM_PARAMS, LABELS, SEEDS, FitResult

log = logging.getLogger(__name__)

# Largest |difference| allowed between stored and re-downloaded log returns
# on overlapping days. Fresh downloads differ by at most ~1.9e-6 (measured
# 2026-11-24 across 6 full and 4 short downloads); a missed SPY dividend
# moves the ex-date return by ~3e-3.
RETURN_TOLERANCE = 2e-5
# Calendar days to download before the last stored date (~10+ sessions).
LOOKBACK_DAYS = 21
FETCH_ATTEMPTS = 3
FETCH_BACKOFF_SECONDS = 2.0
# A same-day bar fetched before this New York time may still be moving.
MARKET_TZ = "America/New_York"
BAR_FINAL_AFTER = "16:30"


class DataValidationError(Exception):
    """Downloaded or combined data failed a check; nothing was written."""


@dataclass(frozen=True)
class Paths:
    data_dir: Path
    models_dir: Path

    @property
    def returns(self): return self.data_dir / "returns.parquet"
    @property
    def prices(self): return self.data_dir / "prices.parquet"
    @property
    def signals(self): return self.data_dir / "signals.parquet"
    @property
    def fits(self): return self.data_dir / "fits.parquet"
    @property
    def metadata(self): return self.data_dir / "metadata.json"
    @property
    def nber(self): return self.data_dir / "nber_recessions.csv"
    @property
    def model(self): return self.models_dir / "latest.joblib"

    def outputs(self):
        return [self.returns, self.prices, self.signals, self.fits, self.model, self.metadata]


# --- fetching ---------------------------------------------------------------

def fetch_with_retry(fetch=None, attempts=FETCH_ATTEMPTS, backoff=FETCH_BACKOFF_SECONDS,
                     sleep=None, **kwargs) -> pd.DataFrame:
    """Call `fetch(**kwargs)`, retrying with exponential backoff (2 s, 4 s, ...).

    `fetch` defaults to yfinance via fetch_prices and `sleep` to time.sleep,
    looked up at call time so tests can replace them.
    """
    fetch = fetch or fetch_prices
    sleep = sleep or time.sleep
    for attempt in range(1, attempts + 1):
        try:
            return fetch(**kwargs)
        except Exception as exc:  # network errors from yfinance are not typed
            if attempt == attempts:
                raise RuntimeError(f"fetch failed after {attempts} attempts: {exc}") from exc
            wait = backoff * 2 ** (attempt - 1)
            log.warning("Fetch attempt %d/%d failed (%s); retrying in %.0fs",
                        attempt, attempts, exc, wait)
            sleep(wait)


def drop_incomplete_bar(prices: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Drop today's bar if it was fetched before the close is final."""
    now = pd.Timestamp.now(tz=MARKET_TZ) if now is None else now.tz_convert(MARKET_TZ)
    if len(prices) and prices.index[-1] == now.normalize().tz_localize(None) \
            and now.strftime("%H:%M") < BAR_FINAL_AFTER:
        log.info("Dropping today's bar %s: fetched before %s New York time",
                 prices.index[-1].date(), BAR_FINAL_AFTER)
        return prices.iloc[:-1]
    return prices


# --- validation -------------------------------------------------------------

def validate_prices(prices: pd.DataFrame) -> None:
    idx = prices.index
    if idx.has_duplicates:
        raise DataValidationError(f"duplicate dates: {idx[idx.duplicated()].date.tolist()}")
    if not idx.is_monotonic_increasing:
        raise DataValidationError("dates are not sorted")
    close = prices["Close"]
    if not np.isfinite(close).all():
        raise DataValidationError("non-finite prices")
    if (close <= 0).any():
        raise DataValidationError(f"non-positive prices on {idx[close <= 0].date.tolist()}")


def validate_sessions(index: pd.DatetimeIndex) -> None:
    """Dates must be exactly the NYSE sessions between the first and last date.

    Missing sessions are gaps; dates that aren't sessions (weekends,
    holidays) are bad data. Exchange holidays and special closures come
    from pandas_market_calendars.
    """
    if index.has_duplicates:
        raise DataValidationError(f"duplicate dates: {index[index.duplicated()].date.tolist()}")
    sessions = mcal.get_calendar("NYSE").valid_days(index[0], index[-1]).tz_localize(None)
    missing = sessions.difference(index)
    extra = index.difference(sessions)
    if len(missing):
        raise DataValidationError(f"missing trading days: {missing.date.tolist()[:10]}")
    if len(extra):
        raise DataValidationError(f"dates that are not NYSE sessions: {extra.date.tolist()[:10]}")


def check_overlap(stored: pd.Series, fresh: pd.Series, tol: float = RETURN_TOLERANCE) -> float:
    """Re-downloaded returns must match stored returns on overlapping dates."""
    common = stored.index.intersection(fresh.index)
    if common.empty:
        raise DataValidationError("download does not overlap the stored history")
    diff = (stored[common] - fresh[common]).abs()
    worst = float(np.max(diff.to_numpy()))  # NaN if any day is NaN, unlike Series.max
    if not worst <= tol:
        bad = diff[~(diff <= tol)]
        raise DataValidationError(
            f"stored and downloaded returns differ by up to {worst:.2e} (tolerance {tol:.0e}) "
            f"on {bad.index.date.tolist()}")
    return worst


def check_not_shrinking(old: pd.Series, new: pd.Series) -> None:
    """New history must keep every stored date and value, and not get shorter."""
    if len(new) < len(old):
        raise DataValidationError(f"history would shrink from {len(old)} to {len(new)} rows")
    head = new.iloc[: len(old)]
    if not head.index.equals(old.index) or not np.array_equal(head.to_numpy(), old.to_numpy()):
        raise DataValidationError("existing history would change")


# --- derived tables ---------------------------------------------------------

def prices_from_returns(returns: pd.Series, anchor_close: float) -> pd.DataFrame:
    """Adjusted close implied by the returns, scaled so the last close equals the anchor."""
    cum = returns.cumsum()
    return (anchor_close * np.exp(cum - cum.iloc[-1])).rename("Close").to_frame()


def build_signals(regimes: pd.DataFrame, returns: pd.Series, cost_bps: float) -> pd.DataFrame:
    """One row per out-of-sample day.

    `signal` is decided at that day's close; `position` is the previous
    day's signal, which earns that day's return. The first day has no
    position yet, so its position and return columns are empty.
    """
    strat = apply_rule(regimes, returns, cost_bps)
    out = regimes[[*PROB_COLS, "label", "refit_date"]].copy()
    out["signal"] = invested(out["label"])
    out = out.join(strat.rename(columns={"gross_ret": "strategy_gross_ret",
                                         "net_ret": "strategy_net_ret"}))
    return out


def settings(burn_in: int, refit_every: int, cost_bps: float) -> dict:
    return {
        "ticker": TICKER,
        "start": START,
        "features": FEATURES,
        "vol_window": VOL_WINDOW,
        "burn_in": burn_in,
        "refit_every": refit_every,
        "hmm_params": HMM_PARAMS,
        "seeds": list(SEEDS),
        "labels": list(LABELS),
        "labelling": "states ordered by raw mean 21-day volatility: lowest Calm, highest Stressed",
        "rule": "invested unless the most likely filtered state is Stressed; signal applied next day",
        "cost_bps_per_switch": cost_bps,
        "risk_free_rate": 0.0,
        "return_tolerance": RETURN_TOLERANCE,
    }


def summarise(signals: pd.DataFrame, fits: pd.DataFrame, nber_path: Path) -> dict:
    s = signals.dropna(subset=["position"])
    ok = fits[fits["ok"]]
    return {
        "metrics_note": "Daily log returns; Sharpe = annualised mean / annualised std "
                        "with a 0% risk-free rate.",
        "metrics": {
            "buy_and_hold": performance(s["bh_ret"]),
            "strategy_gross": performance(s["strategy_gross_ret"], s["position"], s["switch"]),
            "strategy_net": performance(s["strategy_net_ret"], s["position"], s["switch"]),
        },
        "evaluation": {"start": str(s.index[0].date()), "end": str(s.index[-1].date()),
                       "days": len(s)},
        "nber": nber_check(signals["label"], load_nber(nber_path)),
        "convergence": {
            "refits": len(fits),
            "failed_refits": int((~fits["ok"]).sum()),
            "failed_seed_fits": int(fits["failed_seeds"].sum()),
            "hit_max_iter": int(fits["hit_max_iter"].sum()),
            "with_nonmonotonic_warnings": int((fits["nonmonotonic_warnings"] > 0).sum()),
            "best_seed_not_42": int((ok["best_seed"] != 42).sum()),
        },
    }


def fits_frame(records: list[FitRecord]) -> pd.DataFrame:
    cols = list(FitRecord.__dataclass_fields__)
    df = pd.DataFrame([asdict(r) for r in records], columns=cols)
    df["refit_date"] = pd.to_datetime(df["refit_date"])
    return df


# --- writing ----------------------------------------------------------------

def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def write_all(paths: Paths, *, returns, prices, signals, fits, model_state, metadata) -> None:
    """Write every file to a temp name first, then swap them all in.

    If any write fails, the temp files are removed and the existing files
    are untouched. metadata.json is swapped last, so it only reports an
    update once every other file is in place.
    """
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.models_dir.mkdir(parents=True, exist_ok=True)
    writers = {
        paths.returns: lambda p: returns.to_frame("log_return").to_parquet(p),
        paths.prices: lambda p: prices.to_parquet(p),
        paths.signals: lambda p: signals.to_parquet(p),
        paths.fits: lambda p: fits.to_parquet(p),
        paths.model: lambda p: joblib.dump(model_state, p),
        paths.metadata: lambda p: p.write_text(json.dumps(metadata, indent=2, default=_json_default)),
    }
    tmp = {final: final.with_name(final.name + ".tmp") for final in writers}
    try:
        for final, write in writers.items():
            write(tmp[final])
    except BaseException:
        for t in tmp.values():
            t.unlink(missing_ok=True)
        raise
    for final in writers:  # dict order: metadata last
        os.replace(tmp[final], final)


# Rebuilt float columns may differ from stored ones in the last bit when the
# history was built on another platform (e.g. a different libm for log1p).
REBUILT_FLOAT_ATOL = 1e-12
EXACT_COLUMNS = ["label", "refit_date", "signal", "position", "switch"]


def keep_stored_rows(rebuilt: pd.DataFrame, stored: pd.DataFrame) -> pd.DataFrame:
    """Stored rows unchanged, followed by the newly computed rows.

    The rebuilt copies of the stored rows must agree with them: exactly on
    labels, signals and positions, and within REBUILT_FLOAT_ATOL on the
    float columns. Anything larger is a logic error, not float noise.
    """
    kept = rebuilt.iloc[: len(stored)]
    floats = [col for col in stored.columns if col not in EXACT_COLUMNS]
    try:
        pd.testing.assert_frame_equal(kept[EXACT_COLUMNS], stored[EXACT_COLUMNS],
                                      check_exact=True, check_freq=False)
        pd.testing.assert_frame_equal(kept[floats], stored[floats], check_exact=False,
                                      rtol=0, atol=REBUILT_FLOAT_ATOL, check_freq=False)
    except AssertionError as exc:
        raise DataValidationError(f"rebuilt signals changed existing rows: {exc}") from exc
    return pd.concat([stored, rebuilt.iloc[len(stored):]])


def _assemble(paths: Paths, returns, anchor_close, regimes, fits: pd.DataFrame, fit, refit_date,
              cfg: dict, stored_signals: pd.DataFrame | None = None) -> dict:
    signals = build_signals(regimes, returns, cfg["cost_bps_per_switch"])
    if stored_signals is not None:
        signals = keep_stored_rows(signals, stored_signals)
    state = {
        "fit": fit,
        "label_mapping": {label: int(fit.state_order[k]) for k, label in enumerate(LABELS)},
        "last_refit_date": pd.Timestamp(refit_date),
    }
    metadata = {
        "last_update_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_as_of": str(returns.index[-1].date()),
        "first_return_date": str(returns.index[0].date()),
        "last_refit_date": str(pd.Timestamp(refit_date).date()),
        "price_anchor": {"date": str(returns.index[-1].date()), "close": float(anchor_close)},
        "settings": cfg,
        **summarise(signals, fits, paths.nber),
    }
    return dict(returns=returns, prices=prices_from_returns(returns, anchor_close),
                signals=signals, fits=fits, model_state=state, metadata=metadata)


# --- entry points -----------------------------------------------------------

def build_history(paths: Paths, fetch=None, now=None, sleep=None,
                  burn_in: int = BURN_IN, refit_every: int = REFIT_EVERY,
                  cost_bps: float = DEFAULT_COST_BPS) -> dict:
    """Run the full walk-forward once and write every output file."""
    prices = fetch_with_retry(fetch, sleep=sleep, start=START)
    prices = drop_incomplete_bar(prices, now)
    validate_prices(prices)
    validate_sessions(prices.index)

    returns = log_returns(prices)
    regimes, fit_records, fit, refit_date = walk_forward(
        compute_features(returns), burn_in=burn_in, refit_every=refit_every)
    if fit is None:
        raise RuntimeError("no successful refit; nothing to save")

    out = _assemble(paths, returns, prices["Close"].iloc[-1], regimes, fits_frame(fit_records),
                    fit, refit_date, settings(burn_in, refit_every, cost_bps))
    write_all(paths, **out)
    log.info("History built through %s (%d signal rows)", out["metadata"]["data_as_of"],
             len(out["signals"]))
    return out["metadata"]


def _load(paths: Paths):
    missing = [p for p in paths.outputs() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing {', '.join(map(str, missing))}; run build_history first")
    metadata = json.loads(paths.metadata.read_text())
    returns = pd.read_parquet(paths.returns)["log_return"].rename("Returns")
    signals = pd.read_parquet(paths.signals)
    fits = pd.read_parquet(paths.fits)
    state = joblib.load(paths.model)
    return metadata, returns, signals, fits, state


def _check_saved_state(metadata, returns, signals, state, cfg_now) -> None:
    saved = metadata["settings"]
    for key in ("ticker", "start", "features", "vol_window", "hmm_params", "seeds", "labels",
                "rule", "labelling"):
        if saved[key] != cfg_now[key]:
            raise DataValidationError(
                f"saved setting {key!r} differs from the code; rebuild the history")
    if signals.index[-1] != returns.index[-1]:
        raise DataValidationError("signals and returns end on different dates")
    feat_rows = len(returns) - (VOL_WINDOW - 1)
    if feat_rows != saved["burn_in"] + len(signals):
        raise DataValidationError("signals don't line up with the burn-in and returns")
    if pd.Timestamp(state["last_refit_date"]) != signals["refit_date"].iloc[-1]:
        raise DataValidationError("saved model is not the latest refit in signals")


def daily_update(paths: Paths, fetch=None, now=None, sleep=None) -> str:
    """Append new trading days. Returns "updated" or "no_new_data".

    Raises on any failed fetch or check, before anything is written.
    """
    metadata, returns, signals, fits, state = _load(paths)
    cfg = metadata["settings"]
    _check_saved_state(metadata, returns, signals, state,
                       settings(cfg["burn_in"], cfg["refit_every"], cfg["cost_bps_per_switch"]))

    start = (returns.index[-1] - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    prices = fetch_with_retry(fetch, sleep=sleep, start=start)
    prices = drop_incomplete_bar(prices, now)
    validate_prices(prices)
    fresh = log_returns(prices)

    worst = check_overlap(returns, fresh)
    log.info("Overlap check passed: max |diff| %.2e over %d days", worst,
             len(returns.index.intersection(fresh.index)))

    new = fresh[fresh.index > returns.index[-1]]
    if new.empty:
        log.info("No new trading day after %s; nothing to do", returns.index[-1].date())
        return "no_new_data"

    combined = pd.concat([returns, new])
    check_not_shrinking(returns, combined)
    validate_sessions(combined.index)

    feat = compute_features(combined)
    new_regimes, new_records, fit, refit_date = walk_forward(
        feat, burn_in=cfg["burn_in"], refit_every=cfg["refit_every"],
        start_row=cfg["burn_in"] + len(signals),
        current=state["fit"], current_refit_date=state["last_refit_date"])

    regimes = pd.concat([signals[[*PROB_COLS, "label", "refit_date"]], new_regimes])
    if new_records:
        fits = pd.concat([fits, fits_frame(new_records)], ignore_index=True)

    out = _assemble(paths, combined, prices["Close"].iloc[-1], regimes, fits, fit,
                    refit_date, cfg, stored_signals=signals)

    write_all(paths, **out)
    log.info("Appended %d day(s) through %s%s", len(new), new.index[-1].date(),
             f"; {len(new_records)} refit(s)" if new_records else "")
    return "updated"
