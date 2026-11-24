"""Feature engineering: same two features as v1."""
import numpy as np
import pandas as pd

VOL_WINDOW = 21
FEATURES = ["Returns", "Volatility"]


def compute_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Add daily log returns and 21-day rolling volatility of log returns.

    Rows without a full volatility window are dropped, as in v1.
    """
    df = prices[["Close"]].copy()
    df["Returns"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Volatility"] = df["Returns"].rolling(window=VOL_WINDOW).std()
    return df.dropna()
