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


def walk_forward(feat: pd.DataFrame, burn_in: int = BURN_IN, refit_every: int = REFIT_EVERY):
    """Filtered regime probabilities for every day from `burn_in` onwards.

    A refit at row t trains on rows [0, t) and is the latest model for rows
    t .. t+refit_every-1. For each of those rows, probabilities use features
    up to and including that row only.

    If a refit raises, the error is logged and the previous model is kept.
    If there is no previous model, that block gets NaN probabilities.

    Returns (probs DataFrame indexed by date, list[FitRecord]).
    """
    X_all = feat[FEATURES].to_numpy()
    n = len(X_all)
    dates = feat.index

    rows, fits = [], []
    current: FitResult | None = None
    current_refit_date = pd.NaT

    for t in range(burn_in, n, refit_every):
        try:
            fit = fit_hmm(X_all[:t])
        except (ValueError, np.linalg.LinAlgError) as exc:
            log.error("Refit at %s (%d rows) failed: %s", dates[t].date(), t, exc)
            fits.append(FitRecord(dates[t], t, False, None, False, 0, repr(exc)))
        else:
            log.info("Refit %s: best seed %d, loglik %.3f, spread across seeds %.3f",
                     dates[t].date(), fit.best_seed, fit.log_likelihood, fit.loglik_spread)
            if not fit.converged:
                log.warning(
                    "Refit at %s did not converge cleanly: n_iter=%d hit_max_iter=%s "
                    "nonmonotonic_warnings=%d",
                    dates[t].date(), fit.n_iter, fit.hit_max_iter, fit.nonmonotonic_warnings,
                )
            fits.append(FitRecord(dates[t], t, True, fit.n_iter, fit.hit_max_iter,
                                  fit.nonmonotonic_warnings, None, fit.best_seed,
                                  fit.log_likelihood, fit.loglik_spread, fit.failed_seeds))
            current, current_refit_date = fit, dates[t]

        for i in range(t, min(t + refit_every, n)):
            if current is None:
                probs = np.full(len(LABELS), np.nan)
            else:
                probs = filtered_proba(current, X_all[: i + 1])
            rows.append((dates[i], current_refit_date, *probs))

    out = pd.DataFrame(rows, columns=["Date", "refit_date", *PROB_COLS]).set_index("Date")
    probs = out[PROB_COLS].to_numpy()
    valid = ~np.isnan(probs).any(axis=1)
    label = np.full(len(out), None, dtype=object)
    label[valid] = np.array(LABELS)[probs[valid].argmax(axis=1)]
    out["label"] = label
    return out, fits


def apply_rule(regimes: pd.DataFrame, returns: pd.Series,
               cost_bps: float = DEFAULT_COST_BPS) -> pd.DataFrame:
    """Invested unless the most likely state is Stressed (v2 rule).

    Turns daily labels into next-day positions and gross/net log returns.

    The signal decided at the close of day t is held during day t+1. The
    first day of the regime series therefore has no position, so results
    start on its second day. Position before the first day is cash, so the
    first entry counts as a switch. Each switch costs `cost_bps` of wealth.
    """
    signal = regimes["label"].map(lambda lab: lab is not None and lab != "Stressed").astype(int)
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
