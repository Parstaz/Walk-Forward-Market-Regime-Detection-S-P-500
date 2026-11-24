"""Price data loading."""
import pandas as pd
import yfinance as yf

TICKER = "SPY"
START = "2000-01-01"


def fetch_prices(ticker: str = TICKER, start: str = START, end: str | None = None) -> pd.DataFrame:
    """Download daily adjusted closes from yfinance.

    `end` defaults to None, meaning "up to the latest available bar".
    Returns a DataFrame with a single `Close` column and a tz-naive
    DatetimeIndex named `Date`.
    """
    raw = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
        threads=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")

    prices = raw[["Close"]].astype("float64").dropna()
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None).normalize()
    prices.index.name = "Date"
    return prices
