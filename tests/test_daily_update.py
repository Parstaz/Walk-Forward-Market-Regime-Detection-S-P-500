import json
import shutil

import joblib
import numpy as np
import pandas as pd
import pytest

import src.pipeline as pipeline
from conftest import BUILD_AT, BURN_IN, REFIT_EVERY, ROOT, FakeFetch, snapshot
from scripts import daily_update as daily_script
from src.pipeline import DataValidationError, Paths, build_history, daily_update


def no_sleep(_):
    pass


def test_build_writes_every_file(built):
    for p in built.outputs():
        assert p.exists(), p
    meta = json.loads(built.metadata.read_text())
    assert meta["settings"]["risk_free_rate"] == 0.0
    assert "0% risk-free" in meta["metrics_note"]
    signals = pd.read_parquet(built.signals)
    assert list(signals.columns[:4]) == ["p_calm", "p_normal", "p_stressed", "label"]
    state = joblib.load(built.model)
    assert set(state["label_mapping"]) == {"Calm", "Normal", "Stressed"}
    assert state["last_refit_date"] == signals["refit_date"].iloc[-1]


# --- a failed run leaves the data untouched ---------------------------------

def test_failed_fetch_does_not_overwrite(built, truth):
    before = snapshot(built)
    waits = []

    def down(**kw):
        raise ConnectionError("yfinance unreachable")

    with pytest.raises(RuntimeError, match="after 3 attempts"):
        daily_update(built, fetch=down, sleep=waits.append)
    assert waits == [2.0, 4.0]
    assert snapshot(built) == before


def test_failed_fetch_script_exits_1_and_keeps_data(built, monkeypatch):
    before = snapshot(built)

    def down(**kw):
        raise ConnectionError("yfinance unreachable")

    monkeypatch.setattr(pipeline, "fetch_prices", down)
    monkeypatch.setattr(pipeline.time, "sleep", no_sleep)
    code = daily_script.main(["--data-dir", str(built.data_dir),
                              "--models-dir", str(built.models_dir)])
    assert code == 1
    assert snapshot(built) == before


def test_empty_download_does_not_overwrite(built, truth):
    before = snapshot(built)
    fetch = FakeFetch(truth, truth.index[BUILD_AT + 2], edit=lambda df: df.iloc[0:0])
    with pytest.raises(Exception):
        daily_update(built, fetch=fetch, now=fetch.now, sleep=no_sleep)
    assert snapshot(built) == before


def test_missed_dividend_does_not_overwrite(built, truth):
    """Prices before an ex-date not re-adjusted: the overlap check must fail."""
    before = snapshot(built)
    ex_date = truth.index[BUILD_AT - 4]

    def unadjusted(df):
        df.loc[df.index < ex_date, "Close"] *= 1.003
        return df

    fetch = FakeFetch(truth, truth.index[BUILD_AT + 2], edit=unadjusted)
    with pytest.raises(DataValidationError, match="differ"):
        daily_update(built, fetch=fetch, now=fetch.now, sleep=no_sleep)
    assert snapshot(built) == before


def test_gap_in_new_days_does_not_overwrite(built, truth):
    before = snapshot(built)
    skipped = truth.index[BUILD_AT + 1]
    fetch = FakeFetch(truth, truth.index[BUILD_AT + 3], edit=lambda df: df.drop(skipped))
    with pytest.raises(DataValidationError, match="missing trading days"):
        daily_update(built, fetch=fetch, now=fetch.now, sleep=no_sleep)
    assert snapshot(built) == before


def test_duplicate_dates_do_not_overwrite(built, truth):
    before = snapshot(built)
    fetch = FakeFetch(truth, truth.index[BUILD_AT + 2], edit=lambda df: pd.concat([df, df.iloc[-1:]]))
    with pytest.raises(DataValidationError, match="duplicate"):
        daily_update(built, fetch=fetch, now=fetch.now, sleep=no_sleep)
    assert snapshot(built) == before


def test_failed_write_does_not_overwrite(built, truth, monkeypatch):
    """A crash while writing temp files must leave the real files as they were."""
    before = snapshot(built)

    def broken_dump(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(pipeline.joblib, "dump", broken_dump)
    fetch = FakeFetch(truth, truth.index[BUILD_AT + 2])
    with pytest.raises(OSError, match="disk full"):
        daily_update(built, fetch=fetch, now=fetch.now, sleep=no_sleep)
    assert snapshot(built) == before
    assert not list(built.data_dir.glob("*.tmp")) and not list(built.models_dir.glob("*.tmp"))


def test_changed_method_settings_refuse_to_update(built, truth):
    meta = json.loads(built.metadata.read_text())
    meta["settings"]["seeds"] = [42]
    built.metadata.write_text(json.dumps(meta))
    before = snapshot(built)
    fetch = FakeFetch(truth, truth.index[BUILD_AT + 2])
    with pytest.raises(DataValidationError, match="seeds"):
        daily_update(built, fetch=fetch, now=fetch.now, sleep=no_sleep)
    assert snapshot(built) == before


# --- history built on another platform ----------------------------------------

def _nudge_stored_net_return(paths: Paths, how) -> tuple[pd.Timestamp, float]:
    """Alter one stored net return on a switch day, as a different libm might."""
    signals = pd.read_parquet(paths.signals)
    day = signals.index[signals["switch"] == 1][-1]
    signals.loc[day, "strategy_net_ret"] = how(signals.loc[day, "strategy_net_ret"])
    signals.to_parquet(paths.signals)
    return day, signals.loc[day, "strategy_net_ret"]


def test_last_bit_difference_in_stored_floats_is_kept(built, truth):
    """Stored rows built elsewhere may differ in the last bit; the update keeps them as stored."""
    day, nudged = _nudge_stored_net_return(built, lambda v: np.nextafter(v, np.inf))
    fetch = FakeFetch(truth, truth.index[BUILD_AT + 1])
    assert daily_update(built, fetch=fetch, now=fetch.now, sleep=no_sleep) == "updated"
    assert pd.read_parquet(built.signals).loc[day, "strategy_net_ret"] == nudged


def test_real_difference_in_stored_floats_still_fails(built, truth):
    _nudge_stored_net_return(built, lambda v: v + 1e-9)
    before = snapshot(built)
    fetch = FakeFetch(truth, truth.index[BUILD_AT + 1])
    with pytest.raises(DataValidationError, match="changed existing rows"):
        daily_update(built, fetch=fetch, now=fetch.now, sleep=no_sleep)
    assert snapshot(built) == before


# --- normal runs --------------------------------------------------------------

def test_no_new_day_exits_cleanly_without_changes(built, truth):
    before = snapshot(built)
    fetch = FakeFetch(truth, truth.index[BUILD_AT - 1])
    assert daily_update(built, fetch=fetch, now=fetch.now, sleep=no_sleep) == "no_new_data"
    assert snapshot(built) == before


def test_intraday_bar_is_not_appended(built, truth):
    before = snapshot(built)
    fetch = FakeFetch(truth, truth.index[BUILD_AT])
    midday = pd.Timestamp(truth.index[BUILD_AT]).tz_localize("America/New_York") + pd.Timedelta(hours=12)
    assert daily_update(built, fetch=fetch, now=midday, sleep=no_sleep) == "no_new_data"
    assert snapshot(built) == before


def test_rescaled_download_still_appends(built, truth):
    """After a dividend yfinance rescales all history; returns, and the update, are unaffected."""
    fetch = FakeFetch(truth, truth.index[BUILD_AT + 1], scale=0.99)
    assert daily_update(built, fetch=fetch, now=fetch.now, sleep=no_sleep) == "updated"
    returns = pd.read_parquet(built.returns)
    assert returns.index[-1] == truth.index[BUILD_AT + 1]
    prices = pd.read_parquet(built.prices)
    assert prices["Close"].iloc[-1] == pytest.approx(truth["Close"].iloc[BUILD_AT + 1] * 0.99)


def test_daily_updates_match_full_rebuild(built, truth, tmp_path):
    """Day-by-day updates, across several refits, give exactly the full-history result."""
    last = len(truth) - 1
    for i in range(BUILD_AT, last + 1):
        fetch = FakeFetch(truth, truth.index[i])
        assert daily_update(built, fetch=fetch, now=fetch.now, sleep=no_sleep) == "updated"

    full = Paths(tmp_path / "full" / "data", tmp_path / "full" / "models")
    full.data_dir.mkdir(parents=True)
    shutil.copy(ROOT / "data" / "nber_recessions.csv", full.nber)
    fetch = FakeFetch(truth, truth.index[last])
    build_history(full, fetch=fetch, now=fetch.now, burn_in=BURN_IN, refit_every=REFIT_EVERY)

    for name in ("returns.parquet", "signals.parquet", "fits.parquet"):
        a = pd.read_parquet(built.data_dir / name)
        b = pd.read_parquet(full.data_dir / name)
        pd.testing.assert_frame_equal(a, b, check_exact=True)

    refits = pd.read_parquet(built.fits)
    assert (refits["refit_date"] > truth.index[BUILD_AT]).sum() >= 3  # the test crossed refits

    pd.testing.assert_frame_equal(pd.read_parquet(built.prices), pd.read_parquet(full.prices),
                                  rtol=1e-12)
    ma = json.loads(built.metadata.read_text())
    mb = json.loads(full.metadata.read_text())
    for key in ("metrics", "nber", "convergence", "data_as_of", "last_refit_date", "settings"):
        assert ma[key] == mb[key], key
