"""Replay the last trading days of the committed history through the daily update.

Uses the real files in data/ and models/ (built on macOS), so on CI it checks
that a Linux run reproduces the stored labels, positions and probabilities.
"""
import shutil

import joblib
import numpy as np
import pandas as pd
import pytest

from conftest import ROOT
from src.backtest import PROB_COLS
from src.features import FEATURES, compute_features
from src.model import LABELS, filtered_proba, fit_hmm
from src.pipeline import Paths, daily_update

REPLAY_DAYS = 15          # > refit interval, so the replay crosses at least one refit
PROB_TOLERANCE = 1e-6
LOGLIK_RTOL = 1e-9        # restored refit vs the stored log-likelihood


def no_sleep(_):
    pass


@pytest.fixture(scope="module")
def committed():
    data, models = ROOT / "data", ROOT / "models"
    return {
        "signals": pd.read_parquet(data / "signals.parquet"),
        "returns": pd.read_parquet(data / "returns.parquet")["log_return"].rename("Returns"),
        "prices": pd.read_parquet(data / "prices.parquet"),
        "fits": pd.read_parquet(data / "fits.parquet"),
    }


def restore_model_state(committed, cut: pd.Timestamp):
    """Refit the model that was current at `cut` and check it against the stored record.

    Only the latest model is saved, so the earlier one is rebuilt from the
    stored returns. It must reproduce the stored best seed, log-likelihood
    and probabilities, otherwise the replay would not start from the stored
    state and the test fails here.
    """
    fits = committed["fits"]
    record = fits[fits["refit_date"] <= cut].iloc[-1]
    feat = compute_features(committed["returns"])
    fit = fit_hmm(feat[FEATURES].to_numpy()[: int(record["train_rows"])])

    assert fit.best_seed == record["best_seed"], (fit.best_seed, record["best_seed"])
    ll_diff = abs(fit.log_likelihood - record["log_likelihood"])
    assert ll_diff <= LOGLIK_RTOL * abs(record["log_likelihood"]), ll_diff

    # Its probabilities on the stored days it covers must match the stored ones.
    signals = committed["signals"]
    covered = signals.index[(signals.index >= record["refit_date"]) & (signals.index <= cut)]
    rows = feat.index.get_indexer(covered)
    X = feat[FEATURES].to_numpy()
    probs = np.array([filtered_proba(fit, X[: i + 1]) for i in rows])
    restored_diff = float(np.abs(probs - signals.loc[covered, PROB_COLS].to_numpy()).max())
    assert restored_diff <= PROB_TOLERANCE, restored_diff

    state = {
        "fit": fit,
        "label_mapping": {label: int(fit.state_order[k]) for k, label in enumerate(LABELS)},
        "last_refit_date": pd.Timestamp(record["refit_date"]),
    }
    return state, record, ll_diff, restored_diff


def test_replay_last_days_of_committed_history(tmp_path, committed):
    signals, returns, fits = committed["signals"], committed["returns"], committed["fits"]
    replay_dates = signals.index[-REPLAY_DAYS:]
    cut = signals.index[-REPLAY_DAYS - 1]
    replay_refits = fits[fits["refit_date"] > cut]
    assert len(replay_refits) >= 1, "replay window does not cross a refit"

    # Copy the committed files and roll them back to `cut`.
    shutil.copytree(ROOT / "data", tmp_path / "data")
    shutil.copytree(ROOT / "models", tmp_path / "models")
    paths = Paths(tmp_path / "data", tmp_path / "models")
    signals.loc[:cut].to_parquet(paths.signals)
    returns.loc[:cut].to_frame("log_return").to_parquet(paths.returns)
    committed["prices"].loc[:cut].to_parquet(paths.prices)
    fits[fits["refit_date"] <= cut].to_parquet(paths.fits)
    state, record, ll_diff, restored_diff = restore_model_state(committed, cut)
    joblib.dump(state, paths.model)

    # Feed the removed days back in, one trading day at a time, from prices
    # rebuilt from the stored returns (no download).
    source = committed["prices"]

    for day in replay_dates:
        def fetch(start=None, **kwargs):
            return source.loc[pd.Timestamp(start): day]
        now = day.tz_localize("America/New_York") + pd.Timedelta(hours=18)
        assert daily_update(paths, fetch=fetch, now=now, sleep=no_sleep) == "updated"

    replayed = pd.read_parquet(paths.signals)
    pd.testing.assert_index_equal(replayed.index, signals.index)
    got, want = replayed.loc[replay_dates], signals.loc[replay_dates]

    prob_diff = (got[PROB_COLS] - want[PROB_COLS]).abs()
    max_prob_diff = float(prob_diff.to_numpy().max())
    new_fits = pd.read_parquet(paths.fits)
    new_fits = new_fits[new_fits["refit_date"] > cut].reset_index(drop=True)
    replay_ll_diff = (new_fits["log_likelihood"] - replay_refits["log_likelihood"].to_numpy()).abs()
    ret_diff = float((pd.read_parquet(paths.returns)["log_return"] - returns).abs().max())

    print(f"\nReplayed {len(replay_dates)} days {replay_dates[0].date()}..{replay_dates[-1].date()}, "
          f"restored model from refit {record['refit_date'].date()}")
    print(f"  restored refit: |loglik diff| {ll_diff:.3e}, max prob diff on its stored days "
          f"{restored_diff:.3e}")
    print(f"  replay refits: {[str(d.date()) for d in new_fits['refit_date']]}, best seeds "
          f"{new_fits['best_seed'].tolist()} (stored {replay_refits['best_seed'].tolist()}), "
          f"max |loglik diff| {replay_ll_diff.max():.3e}")
    print(f"  max |return diff| vs stored {ret_diff:.3e}")
    print(f"  MAX PROBABILITY DIFFERENCE vs stored: {max_prob_diff:.3e}")

    mismatched = got.index[(got["label"] != want["label"]).to_numpy()]
    assert mismatched.empty, (
        "labels differ from the stored history on "
        + "; ".join(f"{d.date()}: stored {want.at[d, 'label']} replayed {got.at[d, 'label']}, "
                    f"max prob diff {prob_diff.loc[d].max():.3e}" for d in mismatched))
    for col in ("signal", "position", "switch", "refit_date"):
        pd.testing.assert_series_equal(got[col], want[col], check_exact=True)
    assert new_fits["best_seed"].tolist() == replay_refits["best_seed"].tolist()
    assert max_prob_diff <= PROB_TOLERANCE, max_prob_diff
