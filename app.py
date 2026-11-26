"""SPY market regime monitor (Streamlit).

Reads only the files the daily pipeline writes to data/. No yfinance calls,
no model fitting, no model file loading.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA = Path(__file__).resolve().parent / "data"
REPO_URL = "https://github.com/Parstaz/Walk-Forward-Market-Regime-Detection-S-P-500"
MARKET_TZ = "America/New_York"
STALE_AFTER_SESSIONS = 4

# Categorical slots 1-3 of the reference palette (validated all-pairs in both
# modes), plus chart ink. Light and dark are separate steps, not a flip.
PALETTE = {
    "light": {"Calm": "#2a78d6", "Normal": "#1baf7a", "Stressed": "#eb6834",
              "ink": "#0b0b0b", "muted": "#898781", "grid": "#e1e0d9", "benchmark": "#898781",
              "nber": "rgba(82,81,78,0.16)", "stressed_band": "rgba(235,104,52,0.16)"},
    "dark": {"Calm": "#3987e5", "Normal": "#199e70", "Stressed": "#d95926",
             "ink": "#ffffff", "muted": "#898781", "grid": "#2c2c2a", "benchmark": "#898781",
             "nber": "rgba(195,194,183,0.16)", "stressed_band": "rgba(217,89,38,0.22)"},
}
LABELS = ["Calm", "Normal", "Stressed"]
PROB_COLS = {"Calm": "p_calm", "Normal": "p_normal", "Stressed": "p_stressed"}

st.set_page_config(page_title="SPY Regime Monitor", page_icon="📈", layout="wide")


# --- loading (cached per file version) --------------------------------------

def _mtime(name: str) -> float:
    return (DATA / name).stat().st_mtime


@st.cache_data(show_spinner=False)
def load_metadata(_mtime: float) -> dict:
    return json.loads((DATA / "metadata.json").read_text())


@st.cache_data(show_spinner=False)
def load_parquet(name: str, _mtime: float) -> pd.DataFrame:
    return pd.read_parquet(DATA / name)


@st.cache_data(show_spinner=False)
def load_nber(_mtime: float) -> pd.DataFrame:
    # Same convention as src/metrics.py: month after the peak to the end of the trough month.
    df = pd.read_csv(DATA / "nber_recessions.csv", comment="#", parse_dates=["peak", "trough"])
    df["start"] = df["peak"] + pd.offsets.MonthBegin(1)
    df["end"] = df["trough"] + pd.offsets.MonthEnd(0)
    return df


@st.cache_resource(show_spinner=False)
def nyse_calendar():
    import pandas_market_calendars as mcal  # imported lazily: only needed for two dates
    return mcal.get_calendar("NYSE")


@st.cache_data(show_spinner=False, ttl=3600)
def sessions_between(start: str, end: str) -> pd.DatetimeIndex:
    return nyse_calendar().valid_days(start, end).tz_localize(None)


# --- derived values ---------------------------------------------------------

def runs(mask: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """(first, last) date of each contiguous True run."""
    m = mask.to_numpy()
    if not m.any():
        return []
    edges = np.diff(np.concatenate([[0], m.astype(int), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    idx = mask.index
    return [(idx[s], idx[e]) for s, e in zip(starts, ends)]


def schedule(meta: dict, signals: pd.DataFrame, fits: pd.DataFrame):
    """Next refit date, trading days until then, and trading days the data is behind."""
    cfg = meta["settings"]
    as_of = signals.index[-1]
    last_row = cfg["burn_in"] + len(signals) - 1
    next_row = int(fits["train_rows"].max()) + cfg["refit_every"]
    ahead = next_row - last_row

    future = sessions_between(str((as_of + pd.Timedelta(days=1)).date()),
                              str((as_of + pd.Timedelta(days=40)).date()))
    next_refit = future[ahead - 1] if ahead >= 1 and len(future) >= ahead else None

    today = pd.Timestamp.now(tz=MARKET_TZ).normalize().tz_localize(None)
    missed = sessions_between(str((as_of + pd.Timedelta(days=1)).date()), str(today.date()))
    behind = int((missed < today).sum())  # completed sessions not yet in the data
    return next_refit, ahead, behind


def log_dollar_axis(lo: float, hi: float) -> dict:
    """Log y-axis with $ labels at 1-2-5 steps covering [lo, hi]."""
    ticks = []
    for exp in range(int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi))) + 1):
        ticks += [m * 10.0 ** exp for m in (1, 2, 5)]
    ticks = [t for t in ticks if lo / 1.5 <= t <= hi * 1.5]
    return dict(type="log", title=None, tickvals=ticks, ticktext=[f"${t:,g}" for t in ticks])


def fmt_date(ts) -> str:
    return pd.Timestamp(ts).strftime("%a %-d %b %Y")


def base_layout(fig: go.Figure, c: dict, height: int, yaxis: dict):
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None,
                    traceorder="normal"),
        xaxis=dict(showgrid=False, linecolor=c["grid"]),
        yaxis=dict(gridcolor=c["grid"], zeroline=False, **yaxis),
    )
    return fig


# --- page -------------------------------------------------------------------

meta = load_metadata(_mtime("metadata.json"))
signals = load_parquet("signals.parquet", _mtime("signals.parquet"))
prices = load_parquet("prices.parquet", _mtime("prices.parquet"))
fits = load_parquet("fits.parquet", _mtime("fits.parquet"))
nber = load_nber(_mtime("nber_recessions.csv"))

mode = "dark" if st.context.theme.type == "dark" else "light"
c = PALETTE[mode]

as_of = signals.index[-1]
latest = signals.iloc[-1]
next_refit, sessions_to_refit, behind = schedule(meta, signals, fits)

st.title("SPY market regime monitor")
st.markdown(f"**Data as of {fmt_date(as_of)}** (close) · "
            f"pipeline last ran {pd.Timestamp(meta['last_update_utc']):%Y-%m-%d %H:%M} UTC")

if behind > STALE_AFTER_SESSIONS:
    st.warning(f"The data is {behind} trading days old. The daily update has probably "
               "been failing; the regime shown below may be out of date.", icon="⚠️")

# Current state
cols = st.columns(4)
cols[0].metric("Current regime", latest["label"])
cols[1].metric("Position for the next session",
               "Invested" if latest["signal"] == 1 else "In cash",
               help="The rule: hold SPY unless the most likely regime at today's close is "
                    "Stressed. Today's signal is applied from the next trading day.")
cols[2].metric("Last refit", fmt_date(latest["refit_date"]))
cols[3].metric("Next refit due",
               fmt_date(next_refit) if next_refit is not None else "—",
               help=f"The model is refit every {meta['settings']['refit_every']} trading days, "
                    "in the update after that day's close.")
if next_refit is not None:
    cols[3].caption(f"in {sessions_to_refit} trading day{'s' if sessions_to_refit != 1 else ''}")

_net, _bh = meta["metrics"]["strategy_net"], meta["metrics"]["buy_and_hold"]
st.markdown(f"Since {fmt_date(meta['evaluation']['start'])}, out of sample: "
            f"Sharpe **{_net['sharpe']:.2f}** after costs vs {_bh['sharpe']:.2f} buy-and-hold · "
            f"max drawdown **{_net['max_drawdown']:.1%}** vs {_bh['max_drawdown']:.1%}.")

st.caption("Regime probabilities at the latest close (filtered: each uses only data up to that day)")
pcols = st.columns(3)
for col, label in zip(pcols, LABELS):
    col.metric(f"P({label})", f"{latest[PROB_COLS[label]]:.1%}")

# Price with Stressed regimes and NBER recessions
st.subheader("SPY price, Stressed regimes and NBER recessions")
fig = go.Figure()
fig.add_trace(go.Scatter(x=prices.index, y=prices["Close"], name="SPY (adjusted close)",
                         line=dict(color=c["ink"], width=1.5),
                         hovertemplate="$%{y:,.2f}<extra></extra>"))
for i, (s, e) in enumerate(runs(signals["label"] == "Stressed")):
    fig.add_vrect(x0=s, x1=e + pd.Timedelta(days=1), fillcolor=c["stressed_band"], line_width=0,
                  layer="below", showlegend=i == 0, name="Stressed regime", legendgroup="stressed")
visible_nber = nber[nber["end"] >= prices.index[0]]
for i, row in enumerate(visible_nber.itertuples()):
    fig.add_vrect(x0=row.start, x1=row.end, fillcolor=c["nber"], line_width=0, layer="below",
                  showlegend=i == 0, name="NBER recession", legendgroup="nber")
base_layout(fig, c, 420, log_dollar_axis(prices["Close"].min(), prices["Close"].max()))
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
st.caption(f"Regimes start {fmt_date(signals.index[0])}, after the "
           f"{meta['settings']['burn_in']:,}-day training burn-in. Log scale.")

# Filtered probabilities, last 12 months
st.subheader("Regime probabilities, last 12 months")
recent = signals.loc[as_of - pd.DateOffset(months=12):]
fig = go.Figure()
for label in LABELS:
    fig.add_trace(go.Scatter(
        x=recent.index, y=recent[PROB_COLS[label]] * 100, name=label, stackgroup="p",
        mode="lines", line=dict(width=0.5, color=c[label]), fillcolor=c[label],
        hovertemplate=f"{label} %{{y:.0f}}%<extra></extra>"))
base_layout(fig, c, 300, dict(range=[0, 100], ticksuffix="%", title=None))
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

# Equity curve
st.subheader("Strategy vs buy-and-hold")
held = signals.dropna(subset=["position"])
wealth = np.exp(held[["strategy_net_ret", "strategy_gross_ret", "bh_ret"]].cumsum())
fig = go.Figure()
fig.add_trace(go.Scatter(x=wealth.index, y=wealth["bh_ret"], name="Buy-and-hold SPY",
                         line=dict(color=c["benchmark"], width=2),
                         hovertemplate="$%{y:.2f}<extra></extra>"))
fig.add_trace(go.Scatter(x=wealth.index, y=wealth["strategy_net_ret"],
                         name=f"Strategy, after {meta['settings']['cost_bps_per_switch']:g} bps per switch",
                         line=dict(color=c["Calm"], width=2),
                         hovertemplate="$%{y:.2f}<extra></extra>"))
base_layout(fig, c, 380, log_dollar_axis(wealth.min().min(), wealth.max().max()))
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
st.caption(f"Growth of $1 from {fmt_date(wealth.index[0])}, out of sample. Log scale.")

# Metrics
st.subheader("Performance, out of sample")
m = meta["metrics"]
order = [("buy_and_hold", "Buy-and-hold"), ("strategy_gross", "Strategy (gross)"),
         ("strategy_net", "Strategy (net)")]
rows = [("Sharpe ratio", "sharpe", "{:.2f}"), ("Annual volatility", "ann_vol", "{:.1%}"),
        ("Total return", "total_return", "{:.0%}"), ("CAGR", "cagr", "{:.2%}"),
        ("Max drawdown", "max_drawdown", "{:.1%}"), ("Switches", "switches", "{:.0f}"),
        ("Time invested", "pct_invested", "{:.0%}")]
table = pd.DataFrame({name: [f.format(m[key][k]) for _, k, f in rows] for key, name in order},
                     index=[r[0] for r in rows])
st.table(table)
ev, nb = meta["evaluation"], meta["nber"]
st.caption(f"{ev['start']} to {ev['end']} ({ev['days']:,} trading days). "
           "Sharpe uses daily log returns and a 0% risk-free rate.")

st.markdown(
    f"**NBER check:** the model's most likely state was Stressed on "
    f"**{nb['pct_recession_days_stressed']:.0%}** of recession days and "
    f"**{nb['pct_non_recession_days_stressed']:.0%}** of other days. "
    + " ".join(f"{r['recession']}: {r['pct_stressed']:.0%} of {r['days']} days."
               for r in nb["per_recession"])
    + " Only two recessions fall in the test period, so treat this as a sanity check, "
      "not proof.")

with st.expander("Data tables"):
    st.markdown("Regime probabilities, last 12 months")
    st.dataframe(recent[[*PROB_COLS.values(), "label", "signal"]].iloc[::-1]
                 .style.format({v: "{:.1%}" for v in PROB_COLS.values()}), height=260)
    st.markdown("Growth of $1")
    st.dataframe(wealth.rename(columns={"strategy_net_ret": "strategy_net",
                                        "strategy_gross_ret": "strategy_gross",
                                        "bh_ret": "buy_and_hold"}).iloc[::-1]
                 .style.format("${:.2f}"), height=260)

# Method
st.subheader("How it works")
s = meta["settings"]
st.markdown(f"""
- **Data.** Daily SPY prices from {s['start'][:4]} onwards. Two inputs per day: the day's
  log return and the volatility of returns over the past {s['vol_window']} trading days.
- **Model.** A hidden Markov model with three states, fitted only on data before each day
  (an expanding window). It is refit every {s['refit_every']} trading days, trying
  {len(s['seeds'])} random starting points and keeping the best fit.
- **Regimes.** After each refit, the states are named by their average volatility:
  lowest is **Calm**, middle is **Normal**, highest is **Stressed**.
- **Probabilities.** Each day's regime probabilities use only information available at
  that day's close, never later data.
- **Rule.** Hold SPY unless the most likely regime is Stressed, in which case hold cash.
  The signal from one close is applied from the next trading day, and each switch costs
  {s['cost_bps_per_switch']:g} basis points.
- **Updates.** A scheduled job adds each new trading day after the close. This page only
  reads the saved results; it never fetches prices or fits the model.
""")

st.divider()
st.markdown(f"[Source code on GitHub]({REPO_URL}) · Data as of {as_of:%Y-%m-%d} · "
            "A research project, not investment advice.")
