"""Feature engineering: same two features as v1."""
import numpy as np
import pandas as pd

VOL_WINDOW = 21
FEATURES = ["Returns", "Volatility"]


def log_returns(prices: pd.DataFrame) -> pd.Series:
    """Daily log returns of the Close column; the first date is dropped."""
    close = prices["Close"]
    return np.log(close / close.shift(1)).iloc[1:].rename("Returns")


def compute_features(returns: pd.Series) -> pd.DataFrame:
    """Daily log returns and their 21-day rolling volatility.

    Rows without a full volatility window are dropped, as in v1. Each row
    depends only on returns up to that date, so appending new returns never
    changes existing rows.
    """
    df = returns.rename("Returns").to_frame()
    df["Volatility"] = df["Returns"].rolling(window=VOL_WINDOW).std()
    return df.dropna()
