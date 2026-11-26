"""Performance metrics and the NBER recession check.

All return inputs are daily log returns. Sharpe uses a zero risk-free rate,
matching v1: annualised mean log return / annualised std.
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def performance(log_ret: pd.Series, position: pd.Series | None = None,
                switches: pd.Series | None = None) -> dict:
    log_ret = log_ret.dropna()
    wealth = np.exp(log_ret.cumsum())
    years = (log_ret.index[-1] - log_ret.index[0]).days / 365.25
    vol = log_ret.std() * np.sqrt(TRADING_DAYS)
    drawdown = wealth / wealth.cummax() - 1

    return {
        "sharpe": log_ret.mean() * TRADING_DAYS / vol if vol > 0 else np.nan,
        "ann_vol": vol,
        "total_return": wealth.iloc[-1] - 1,
        "cagr": wealth.iloc[-1] ** (1 / years) - 1,
        "max_drawdown": drawdown.min(),
        "switches": int(switches.sum()) if switches is not None else 0,
        "pct_invested": float(position.mean()) if position is not None else 1.0,
    }


def load_nber(path) -> pd.DataFrame:
    """Recession intervals as day ranges.

    NBER dates peaks and troughs by month. Following the convention of
    FRED's USREC series, the recession runs from the first day of the month
    after the peak to the last day of the trough month.
    """
    df = pd.read_csv(path, comment="#", parse_dates=["peak", "trough"])
    df["start"] = df["peak"] + pd.offsets.MonthBegin(1)
    df["end"] = df["trough"] + pd.offsets.MonthEnd(0)
    return df


def recession_mask(index: pd.DatetimeIndex, nber: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=index)
    for start, end in zip(nber["start"], nber["end"]):
        mask |= (index >= start) & (index <= end)
    return mask


def nber_check(labels: pd.Series, nber: pd.DataFrame) -> dict:
    """Share of recession / non-recession days whose most likely state is Stressed."""
    labels = labels.dropna()
    in_rec = recession_mask(labels.index, nber)
    stressed = labels == "Stressed"

    per_recession = []
    for _, row in nber.iterrows():
        m = (labels.index >= row["start"]) & (labels.index <= row["end"])
        if m.any():
            per_recession.append({
                "recession": f"{row['start']:%Y-%m} to {row['end']:%Y-%m}",
                "days": int(m.sum()),
                "pct_stressed": float(stressed[m].mean()),
            })

    return {
        "recession_days": int(in_rec.sum()),
        "non_recession_days": int((~in_rec).sum()),
        "pct_recession_days_stressed": float(stressed[in_rec].mean()),
        "pct_non_recession_days_stressed": float(stressed[~in_rec].mean()),
        "per_recession": per_recession,
    }
