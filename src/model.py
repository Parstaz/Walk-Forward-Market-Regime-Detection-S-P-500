"""Gaussian HMM fitting, state labelling and filtered probabilities."""
import logging
from dataclasses import dataclass, field

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from .features import FEATURES

log = logging.getLogger(__name__)

# Same settings as the v1 walk-forward loop.
HMM_PARAMS = dict(
    n_components=3,
    covariance_type="full",
    n_iter=1000,
    tol=0.01,
    min_covar=0.001,
    random_state=42,
)

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

    @property
    def converged(self) -> bool:
        return not self.hit_max_iter and self.nonmonotonic_warnings == 0


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


def fit_hmm(train_features: np.ndarray) -> FitResult:
    """Fit scaler + HMM on the training window and label the states.

    Raises whatever hmmlearn raises if the fit itself fails; the caller
    decides how to handle that.
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(train_features)
    model = GaussianHMM(**HMM_PARAMS)

    counter = _CountWarnings()
    hmm_log = logging.getLogger("hmmlearn.base")
    hmm_log.addHandler(counter)
    try:
        model.fit(X)
    finally:
        hmm_log.removeHandler(counter)

    state_order, raw_vol = label_states(model, scaler)
    n_iter = model.monitor_.iter
    return FitResult(
        model=model,
        scaler=scaler,
        state_order=state_order,
        n_iter=n_iter,
        hit_max_iter=n_iter >= HMM_PARAMS["n_iter"],
        nonmonotonic_warnings=counter.count,
        raw_state_vol=raw_vol,
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
