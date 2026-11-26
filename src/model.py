"""Gaussian HMM fitting, state labelling and filtered probabilities."""
import logging
from dataclasses import dataclass, field

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from .features import FEATURES

log = logging.getLogger(__name__)

# Same settings as the v1 walk-forward loop; random_state is set per seed.
HMM_PARAMS = dict(
    n_components=3,
    covariance_type="full",
    n_iter=1000,
    tol=0.01,
    min_covar=0.001,
)

# EM is started from each seed and the fit with the highest log-likelihood
# is kept. A single start sometimes lands in a worse local optimum.
SEEDS = (42, 43, 44, 45, 46)

LABELS = ("Calm", "Normal", "Stressed")
VOL_COL = FEATURES.index("Volatility")


@dataclass
class FitResult:
    model: GaussianHMM
    scaler: StandardScaler
    # state_order[k] is the HMM state index carrying LABELS[k].
    state_order: np.ndarray
    n_iter: int
    hit_max_iter: bool
    nonmonotonic_warnings: int
    raw_state_vol: np.ndarray = field(repr=False)
    best_seed: int
    log_likelihood: float
    # Seed -> log-likelihood on the training window, NaN if that seed's fit raised.
    seed_loglik: dict[int, float] = field(repr=False)

    @property
    def converged(self) -> bool:
        return not self.hit_max_iter and self.nonmonotonic_warnings == 0

    @property
    def loglik_spread(self) -> float:
        ok = [v for v in self.seed_loglik.values() if np.isfinite(v)]
        return max(ok) - min(ok)

    @property
    def failed_seeds(self) -> int:
        return sum(not np.isfinite(v) for v in self.seed_loglik.values())


class _CountWarnings(logging.Handler):
    """Counts hmmlearn's 'Model is not converging' warnings during a fit."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.count = 0

    def emit(self, record):
        self.count += 1


def label_states(model: GaussianHMM, scaler: StandardScaler) -> tuple[np.ndarray, np.ndarray]:
    """Order states by their raw (unscaled) mean 21-day volatility.

    Returns (state_order, raw_vol) where state_order[0] is the Calm state
    (lowest volatility), [1] Normal and [2] Stressed (highest).
    """
    raw_means = scaler.inverse_transform(model.means_)
    raw_vol = raw_means[:, VOL_COL]
    return np.argsort(raw_vol, kind="stable"), raw_vol


def _fit_one(X: np.ndarray, seed: int) -> tuple[GaussianHMM, int]:
    model = GaussianHMM(**HMM_PARAMS, random_state=seed)
    counter = _CountWarnings()
    hmm_log = logging.getLogger("hmmlearn.base")
    hmm_log.addHandler(counter)
    try:
        model.fit(X)
    finally:
        hmm_log.removeHandler(counter)
    return model, counter.count


def fit_hmm(train_features: np.ndarray, seeds=SEEDS) -> FitResult:
    """Fit the scaler, then the HMM from each seed; keep the best log-likelihood.

    Ties go to the earlier seed. A seed whose fit raises is logged and
    skipped; if every seed raises, the last error is re-raised.
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(train_features)

    best = None
    seed_loglik = {}
    last_exc = None
    for seed in seeds:
        try:
            model, warnings = _fit_one(X, seed)
            ll = model.score(X)
        except (ValueError, np.linalg.LinAlgError) as exc:
            log.warning("Seed %d failed on %d rows: %s", seed, len(X), exc)
            seed_loglik[seed] = np.nan
            last_exc = exc
            continue
        seed_loglik[seed] = ll
        if best is None or ll > best[2]:
            best = (seed, model, ll, warnings)

    if best is None:
        raise last_exc

    seed, model, ll, warnings = best
    state_order, raw_vol = label_states(model, scaler)
    n_iter = model.monitor_.iter
    return FitResult(
        model=model,
        scaler=scaler,
        state_order=state_order,
        n_iter=n_iter,
        hit_max_iter=n_iter >= HMM_PARAMS["n_iter"],
        nonmonotonic_warnings=warnings,
        raw_state_vol=raw_vol,
        best_seed=seed,
        log_likelihood=ll,
        seed_loglik=seed_loglik,
    )


def filtered_proba(fit: FitResult, features_upto_t: np.ndarray) -> np.ndarray:
    """P(state at t | observations up to and including t), ordered as LABELS.

    predict_proba runs forward-backward over the sequence it is given; on a
    sequence that ends at t, the last row's posterior uses no data after t,
    so it equals the forward (filtered) probability at t.
    """
    X = fit.scaler.transform(features_upto_t)
    last = fit.model.predict_proba(X)[-1]
    return last[fit.state_order]
