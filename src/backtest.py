"""Expanding-window walk-forward with filtered regime probabilities."""
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import FEATURES
from .model import LABELS, FitResult, fit_hmm, filtered_proba

log = logging.getLogger(__name__)

BURN_IN = 1250
REFIT_EVERY = 10
DEFAULT_COST_BPS = 5.0

PROB_COLS = [f"p_{name.lower()}" for name in LABELS]


@dataclass
class FitRecord:
    refit_date: pd.Timestamp
    train_rows: int
    ok: bool
    n_iter: int | None
    hit_max_iter: bool
    nonmonotonic_warnings: int
    error: str | None
    best_seed: int | None = None
    log_likelihood: float = np.nan
    loglik_spread: float = np.nan
    failed_seeds: int = 0


def is_refit_row(i: int, burn_in: int = BURN_IN, refit_every: int = REFIT_EVERY) -> bool:
    return i >= burn_in and (i - burn_in) % refit_every == 0


def walk_forward(feat: pd.DataFrame, burn_in: int = BURN_IN, refit_every: int = REFIT_EVERY,
                 start_row: int | None = None, current: FitResult | None = None,
                 current_refit_date=pd.NaT):
    """Filtered regime probabilities for feature rows `start_row` .. end.

    Refits happen at rows burn_in, burn_in + refit_every, ... A refit at row
    t trains on rows [0, t) and is the latest model for rows t onwards until
    the next refit. Probabilities for row i use features up to and including
    row i only.

    The full backtest starts at row `burn_in` with no model. The daily
    update resumes at the first new row with the saved model, so both run
    the same loop and produce the same rows.

    If a refit raises, the error is logged and the previous model is kept.
    If there is no previous model, those rows get NaN probabilities.

    Returns (regimes DataFrame indexed by date, list[FitRecord], latest
    model, latest refit date).
    """
    X_all = feat[FEATURES].to_numpy()
    n = len(X_all)
    dates = feat.index
    start_row = burn_in if start_row is None else start_row
    if start_row < burn_in:
        raise ValueError(f"start_row {start_row} is inside the {burn_in}-row burn-in")

    rows, fits = [], []
    for i in range(start_row, n):
        if is_refit_row(i, burn_in, refit_every):
            try:
                fit = fit_hmm(X_all[:i])
            except (ValueError, np.linalg.LinAlgError) as exc:
                log.error("Refit at %s (%d rows) failed: %s", dates[i].date(), i, exc)
                fits.append(FitRecord(dates[i], i, False, None, False, 0, repr(exc)))
            else:
                log.info("Refit %s: best seed %d, loglik %.3f, spread across seeds %.3f",
                         dates[i].date(), fit.best_seed, fit.log_likelihood, fit.loglik_spread)
                if not fit.converged:
                    log.warning(
                        "Refit at %s did not converge cleanly: n_iter=%d hit_max_iter=%s "
                        "nonmonotonic_warnings=%d",
                        dates[i].date(), fit.n_iter, fit.hit_max_iter, fit.nonmonotonic_warnings,
                    )
                fits.append(FitRecord(dates[i], i, True, fit.n_iter, fit.hit_max_iter,
                                      fit.nonmonotonic_warnings, None, fit.best_seed,
                                      fit.log_likelihood, fit.loglik_spread, fit.failed_seeds))
                current, current_refit_date = fit, dates[i]

        if current is None:
            probs = np.full(len(LABELS), np.nan)
        else:
            probs = filtered_proba(current, X_all[: i + 1])
        rows.append((dates[i], current_refit_date, *probs))

    out = pd.DataFrame(rows, columns=["Date", "refit_date", *PROB_COLS]).set_index("Date")
    out["refit_date"] = pd.to_datetime(out["refit_date"])
    probs = out[PROB_COLS].to_numpy(dtype=float)
    valid = ~np.isnan(probs).any(axis=1)
    label = np.full(len(out), None, dtype=object)
    label[valid] = np.array(LABELS)[probs[valid].argmax(axis=1)]
    out["label"] = label
    return out, fits, current, current_refit_date


def invested(labels: pd.Series) -> pd.Series:
    """1 unless the most likely state is Stressed; 0 (cash) when there is no label."""
    return (labels.notna() & (labels != "Stressed")).astype(int)


def apply_rule(regimes: pd.DataFrame, returns: pd.Series,
               cost_bps: float = DEFAULT_COST_BPS) -> pd.DataFrame:
    """Invested unless the most likely state is Stressed (v2 rule).

    Turns daily labels into next-day positions and gross/net log returns.

    The signal decided at the close of day t is held during day t+1. The
    first day of the regime series therefore has no position, so results
    start on its second day. Position before the first day is cash, so the
    first entry counts as a switch. Each switch costs `cost_bps` of wealth.
    """
    signal = invested(regimes["label"])
    position = signal.shift(1).iloc[1:].astype(int)

    r = returns.reindex(position.index)
    switches = position.diff().fillna(position.iloc[0]).abs()
    gross = r * position
    net = gross + switches * np.log1p(-cost_bps / 1e4)

    return pd.DataFrame({
        "position": position,
        "switch": switches.astype(int),
        "bh_ret": r,
        "gross_ret": gross,
        "net_ret": net,
    })
