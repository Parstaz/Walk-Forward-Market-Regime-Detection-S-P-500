import numpy as np
import pandas as pd
import pytest

from src.features import log_returns
from src.pipeline import (DataValidationError, check_not_shrinking, check_overlap,
                          drop_incomplete_bar, fetch_with_retry, validate_prices,
                          validate_sessions)


def prices(dates, values=None):
    idx = pd.DatetimeIndex(pd.to_datetime(dates), name="Date")
    values = np.linspace(100, 101, len(idx)) if values is None else values
    return pd.DataFrame({"Close": values}, index=idx)


WEEK = ["2024-03-04", "2024-03-05", "2024-03-06", "2024-03-07", "2024-03-08"]


class TestValidatePrices:
    def test_clean_passes(self):
        validate_prices(prices(WEEK))

    def test_duplicate_dates_fail(self):
        with pytest.raises(DataValidationError, match="duplicate"):
            validate_prices(prices(WEEK + ["2024-03-08"]))

    def test_unsorted_dates_fail(self):
        with pytest.raises(DataValidationError, match="sorted"):
            validate_prices(prices(list(reversed(WEEK))))

    @pytest.mark.parametrize("bad", [0.0, -5.0])
    def test_non_positive_prices_fail(self, bad):
        with pytest.raises(DataValidationError, match="non-positive"):
            validate_prices(prices(WEEK, [100, 101, bad, 102, 103]))

    def test_nan_price_fails(self):
        with pytest.raises(DataValidationError, match="non-finite"):
            validate_prices(prices(WEEK, [100, 101, np.nan, 102, 103]))


class TestValidateSessions:
    def test_normal_holidays_pass(self):
        # 2024-03-29 Good Friday and 2024-12-25 Christmas are NYSE holidays.
        validate_sessions(pd.DatetimeIndex(["2024-03-27", "2024-03-28", "2024-04-01"]))
        validate_sessions(pd.DatetimeIndex(["2024-12-23", "2024-12-24", "2024-12-26"]))

    def test_special_closure_passes(self):
        # Markets were closed 2001-09-11 to 2001-09-14.
        validate_sessions(pd.DatetimeIndex(["2001-09-10", "2001-09-17"]))

    def test_missing_session_fails(self):
        with pytest.raises(DataValidationError, match="missing trading days"):
            validate_sessions(pd.DatetimeIndex(["2024-03-04", "2024-03-05", "2024-03-07"]))

    def test_weekend_date_fails(self):
        with pytest.raises(DataValidationError, match="not NYSE sessions"):
            validate_sessions(pd.DatetimeIndex(["2024-03-08", "2024-03-09", "2024-03-11"]))


class TestOverlap:
    def stored(self):
        return log_returns(prices(WEEK, [100, 101, 99, 100, 102]))

    def test_identical_passes(self):
        assert check_overlap(self.stored(), self.stored()) == 0.0

    def test_rescaled_history_passes(self):
        # yfinance re-adjusting all prices by a constant leaves returns unchanged.
        fresh = log_returns(prices(WEEK, np.array([100, 101, 99, 100, 102]) * 0.987))
        assert check_overlap(self.stored(), fresh) < 1e-12

    def test_download_noise_passes(self):
        fresh = self.stored() + 2e-6
        check_overlap(self.stored(), fresh)

    def test_missed_dividend_fails(self):
        fresh = self.stored().copy()
        fresh.iloc[2] -= 3e-3  # ~ one SPY quarterly dividend
        with pytest.raises(DataValidationError, match="differ"):
            check_overlap(self.stored(), fresh)

    def test_nan_fails(self):
        fresh = self.stored().copy()
        fresh.iloc[1] = np.nan
        with pytest.raises(DataValidationError, match="differ"):
            check_overlap(self.stored(), fresh)

    def test_no_overlap_fails(self):
        s = self.stored()
        fresh = pd.Series([0.01], index=pd.DatetimeIndex(["2024-04-01"]))
        with pytest.raises(DataValidationError, match="overlap"):
            check_overlap(s, fresh)


class TestNotShrinking:
    def old(self):
        return pd.Series([0.01, -0.02, 0.03], index=pd.DatetimeIndex(WEEK[:3]))

    def test_append_passes(self):
        new = pd.concat([self.old(), pd.Series([0.0], index=pd.DatetimeIndex([WEEK[3]]))])
        check_not_shrinking(self.old(), new)

    def test_shorter_fails(self):
        with pytest.raises(DataValidationError, match="shrink"):
            check_not_shrinking(self.old(), self.old().iloc[:2])

    def test_changed_value_fails(self):
        new = self.old().copy()
        new.iloc[0] = 0.011
        with pytest.raises(DataValidationError, match="change"):
            check_not_shrinking(self.old(), new)


class TestIncompleteBar:
    def frame(self):
        return prices(WEEK)

    def test_intraday_bar_dropped(self):
        now = pd.Timestamp("2024-03-08 14:00", tz="America/New_York")
        assert drop_incomplete_bar(self.frame(), now).index[-1] == pd.Timestamp("2024-03-07")

    def test_after_close_kept(self):
        now = pd.Timestamp("2024-03-08 17:30", tz="America/New_York")
        assert len(drop_incomplete_bar(self.frame(), now)) == 5

    def test_utc_clock_converted(self):
        # 22:30 UTC is 17:30 New York in winter.
        now = pd.Timestamp("2024-03-08 22:30", tz="UTC")
        assert len(drop_incomplete_bar(self.frame(), now)) == 5


class TestRetry:
    def test_succeeds_on_third_attempt(self):
        waits, calls = [], []

        def flaky(**kw):
            calls.append(kw)
            if len(calls) < 3:
                raise ConnectionError("boom")
            return "ok"

        assert fetch_with_retry(flaky, sleep=waits.append, start="x") == "ok"
        assert waits == [2.0, 4.0]
        assert calls == [{"start": "x"}] * 3

    def test_gives_up_after_three(self):
        waits = []

        def broken(**kw):
            raise ConnectionError("down")

        with pytest.raises(RuntimeError, match="after 3 attempts"):
            fetch_with_retry(broken, sleep=waits.append)
        assert waits == [2.0, 4.0]
