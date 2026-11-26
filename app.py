"""SPY market regime monitor (Streamlit).

Reads only the files the daily pipeline writes to data/. No yfinance calls,
no model fitting, no model file loading.

Theme colours and fonts live in .streamlit/config.toml. The page only picks
regime colour shades per mode itself; everything else follows the live theme.
"""
import json
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

DATA = Path(__file__).resolve().parent / "data"
REPO_URL = "https://github.com/Parstaz/Walk-Forward-Market-Regime-Detection-S-P-500"
MARKET_TZ = "America/New_York"
STALE_AFTER_SESSIONS = 4

# Only the regime shades are picked per mode (via st.context.theme). Everything
# else follows the live theme on the browser side, so a theme switch never
# leaves stale colours: chart backgrounds, gridlines, axis text and hover come
# from Streamlit's Plotly theme; custom text and lines derive from the current
# text colour; the few fixed line colours below pass on both backgrounds.
#
# Contrast (WCAG) is checked against both page backgrounds (#F6F8FA, #0B0F14),
# so a shade still reads if the mode changes before the next rerun. Regime
# colours are >= 3:1 for chart marks; light-mode Calm green is 3.1:1, so green
# *text* in light mode uses calm_text (#15803D, 4.7:1).
REGIME_SHADES = {
    "light": dict(Calm="#16A34A", Normal="#B45309", Stressed="#DC2626", calm_text="#15803D",
                  stressed_band="rgba(220,38,38,0.10)"),
    "dark": dict(Calm="#22C55E", Normal="#F59E0B", Stressed="#EF4444", calm_text="#22C55E",
                 stressed_band="rgba(239,68,68,0.16)"),
}
ACCENT = "#2563EB"        # strategy line: 4.9:1 on light, 3.7:1 on dark
SPY_GRAY = "#71777F"      # SPY price and buy-and-hold lines: 4.2:1 on both backgrounds
NBER_BAND = "rgba(125,133,144,0.18)"
ICONS = {"Calm": "▲", "Normal": "●", "Stressed": "▼"}
MONO = "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace"

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


def text_color(c: dict, label: str) -> str:
    return c["calm_text"] if label == "Calm" else c[label]


def inject_css() -> None:
    """All custom CSS. Targets only our own rm-* markup; colours derive from the
    current text colour so they follow a live theme switch."""
    st.html(f"""<style>
    .rm-num {{ font-family: {MONO}; font-variant-numeric: tabular-nums; }}
    .rm-muted {{ color: color-mix(in srgb, currentColor 70%, transparent); font-size: 0.85rem; }}
    .rm-ticker {{ display: flex; flex-wrap: wrap; align-items: baseline; gap: 0.25rem 1rem; }}
    .rm-sym {{ font-size: 2rem; font-weight: 700; letter-spacing: -0.02em; }}
    .rm-close {{ font-size: 1.75rem; font-weight: 600; }}
    .rm-chg {{ font-size: 1.1rem; font-weight: 600; }}
    .rm-badge {{ display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem 1.5rem;
                padding: 1rem 1.25rem; border-radius: 0.75rem; border: 1px solid; }}
    .rm-badge-main {{ font-size: 1.9rem; font-weight: 700; }}
    .rm-badge-sub {{ color: color-mix(in srgb, currentColor 70%, transparent); font-size: 0.9rem; }}
    .rm-probs {{ display: flex; flex-wrap: wrap; gap: 0.25rem 1.25rem; margin-left: auto; }}
    .rm-table {{ width: 100%; border-collapse: collapse; font-size: 0.92rem; }}
    .rm-table th, .rm-table td {{ padding: 0.45rem 0.6rem;
        border-bottom: 1px solid color-mix(in srgb, currentColor 15%, transparent); }}
    .rm-table th {{ color: color-mix(in srgb, currentColor 70%, transparent); font-weight: 500;
                   text-align: right; }}
    .rm-table th:first-child, .rm-table td:first-child {{ text-align: left; }}
    .rm-table td {{ text-align: right; }}
    .rm-scroll {{ overflow-x: auto; }}
    </style>""")


def style_fig(fig: go.Figure, height: int) -> go.Figure:
    """Transparent backgrounds and no fixed ink colours, so Streamlit's Plotly theme
    supplies background, gridlines, axis text and hover per the live theme."""
    fig.update_layout(
        height=height,
        margin=dict(l=4, r=4, t=4, b=4),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        hoverlabel=dict(font=dict(family=MONO, size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, title=None,
                    traceorder="normal", bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(showgrid=False, showline=False, zeroline=False, ticks="", automargin=True,
                     hoverformat="%a %-d %b %Y")
    fig.update_yaxes(showgrid=True, gridwidth=1, showline=False, zeroline=False, ticks="",
                     automargin=True, tickfont=dict(family=MONO))
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", theme="streamlit",
                    config={"displayModeBar": False, "responsive": True})


def panel():
    return st.container(border=True)


def regime_text(label: str) -> str:
    return f"{ICONS[label]} {label}"


# --- page -------------------------------------------------------------------

meta = load_metadata(_mtime("metadata.json"))
signals = load_parquet("signals.parquet", _mtime("signals.parquet"))
prices = load_parquet("prices.parquet", _mtime("prices.parquet"))
returns = load_parquet("returns.parquet", _mtime("returns.parquet"))["log_return"]
fits = load_parquet("fits.parquet", _mtime("fits.parquet"))
nber = load_nber(_mtime("nber_recessions.csv"))

# Regime shades only; see REGIME_SHADES.
c = REGIME_SHADES["dark" if st.context.theme.type == "dark" else "light"]
inject_css()

as_of = signals.index[-1]
latest = signals.iloc[-1]
next_refit, sessions_to_refit, behind = schedule(meta, signals, fits)
s = meta["settings"]
m = meta["metrics"]
ev, nb = meta["evaluation"], meta["nber"]

# 1. Ticker header
close = prices["Close"].iloc[-1]
chg = np.expm1(returns.iloc[-1])
up = chg >= 0
st.html(f"""
<div class="rm-ticker">
  <span class="rm-sym">SPY</span>
  <span class="rm-close rm-num">${close:,.2f}</span>
  <span class="rm-chg rm-num" style="color:{c['calm_text'] if up else c['Stressed']}">
    {'▲' if up else '▼'} {chg:+.2%}</span>
  <span class="rm-muted">1 day</span>
</div>
<div class="rm-muted">Data as of {fmt_date(as_of)} close · pipeline last ran
  {pd.Timestamp(meta['last_update_utc']):%Y-%m-%d %H:%M} UTC</div>
""")

if behind > STALE_AFTER_SESSIONS:
    st.warning(f"The data is {behind} trading days old. The daily update has probably "
               "been failing; the regime shown below may be out of date.", icon="⚠️")

# 2. Regime badge and KPI cards
label = latest["label"]
col = c[label]
probs = " ".join(
    f'<span class="rm-num" style="color:{text_color(c, lab)}">{regime_text(lab)} '
    f'{latest[PROB_COLS[lab]]:.1%}</span>' for lab in LABELS)
st.html(f"""
<div class="rm-badge" style="border-color:{col}; background:{col}1a">
  <div>
    <div class="rm-badge-sub">Current regime</div>
    <div class="rm-badge-main" style="color:{text_color(c, label)}">{regime_text(label)}</div>
  </div>
  <div>
    <div class="rm-badge-sub">Probability</div>
    <div class="rm-badge-main rm-num">{latest[PROB_COLS[label]]:.1%}</div>
  </div>
  <div class="rm-probs">{probs}</div>
</div>
<div class="rm-muted" style="margin-top:0.35rem">Filtered probabilities at the latest close:
  each uses only data up to that day.</div>
""")

net, bh = m["strategy_net"], m["buy_and_hold"]
kpis = st.columns(5)
kpis[0].metric("Position for the next session",
               "Invested" if latest["signal"] == 1 else "In cash", border=True,
               help="The rule: hold SPY unless the most likely regime at today's close is "
                    "Stressed. Today's signal is applied from the next trading day.")
kpis[1].metric("Last refit", fmt_date(latest["refit_date"]), border=True)
kpis[2].metric("Next refit due", fmt_date(next_refit) if next_refit is not None else "—",
               delta=(f"in {sessions_to_refit} trading day{'s' if sessions_to_refit != 1 else ''}"
                      if next_refit is not None else None),
               delta_color="off", delta_arrow="off", border=True,
               help=f"The model is refit every {s['refit_every']} trading days, "
                    "in the update after that day's close.")
kpis[3].metric("Sharpe, after costs", f"{net['sharpe']:.2f}",
               delta=f"{net['sharpe'] - bh['sharpe']:+.2f} vs B&H {bh['sharpe']:.2f}",
               delta_color="off", delta_arrow="off", border=True)
kpis[4].metric("Max drawdown, after costs", f"{net['max_drawdown']:.1%}",
               delta=f"{(net['max_drawdown'] - bh['max_drawdown']) * 100:+.1f} pts vs "
                     f"B&H {bh['max_drawdown']:.1%}",
               delta_color="off", delta_arrow="off", border=True)
st.caption(f"Out of sample since {fmt_date(ev['start'])}; B&H = buy-and-hold SPY. "
           "Sharpe uses daily log returns and a 0% risk-free rate.")

# 3. Price, Stressed periods, NBER recessions, and the regime strip
with panel():
    st.markdown("**SPY price** · Stressed regimes and NBER recessions shaded")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.94, 0.06],
                        vertical_spacing=0.02)
    fig.add_trace(go.Scatter(x=prices.index, y=prices["Close"], name="SPY",
                             line=dict(color=SPY_GRAY, width=1.4), showlegend=False,
                             hovertemplate="SPY $%{y:,.2f}<extra></extra>"), row=1, col=1)
    for i, (start, end) in enumerate(runs(signals["label"] == "Stressed")):
        fig.add_vrect(x0=start, x1=end + pd.Timedelta(days=1), fillcolor=c["stressed_band"],
                      line_width=0, layer="below", row=1, col=1, showlegend=i == 0,
                      name=f"{ICONS['Stressed']} Stressed regime", legendgroup="stressed")
    visible_nber = nber[nber["end"] >= prices.index[0]]
    for i, row in enumerate(visible_nber.itertuples()):
        fig.add_vrect(x0=row.start, x1=row.end, fillcolor=NBER_BAND, line_width=0,
                      layer="below", row=1, col=1, showlegend=i == 0, name="NBER recession",
                      legendgroup="nber")
    codes = signals["label"].map({lab: k for k, lab in enumerate(LABELS)})
    fig.add_trace(go.Heatmap(
        x=signals.index, y=[0], z=[codes.to_numpy(dtype=float)], showscale=False,
        zmin=0, zmax=2, colorscale=[[0, c["Calm"]], [0.33, c["Calm"]], [0.34, c["Normal"]],
                                    [0.66, c["Normal"]], [0.67, c["Stressed"]], [1, c["Stressed"]]],
        customdata=[signals["label"].map(regime_text).to_numpy()],
        hovertemplate="%{customdata}<extra></extra>", name="Regime"), row=2, col=1)
    style_fig(fig, 440)
    fig.update_yaxes(log_dollar_axis(prices["Close"].min(), prices["Close"].max()), row=1, col=1)
    fig.update_yaxes(visible=False, showgrid=False, row=2, col=1)
    show(fig)
    st.caption(f"Strip below the chart: daily regime ({' · '.join(regime_text(x) for x in LABELS)}). "
               f"Regimes start {fmt_date(signals.index[0])}, after the {s['burn_in']:,}-day "
               "training burn-in. Log scale.")

# 4. Filtered probabilities, last 12 months
recent = signals.loc[as_of - pd.DateOffset(months=12):]
with panel():
    st.markdown("**Regime probabilities, last 12 months**")
    fig = go.Figure()
    for lab in LABELS:
        fig.add_trace(go.Scatter(
            x=recent.index, y=recent[PROB_COLS[lab]] * 100, name=regime_text(lab),
            stackgroup="p", mode="lines", line=dict(width=0.5, color=c[lab]), fillcolor=c[lab],
            hovertemplate=f"{regime_text(lab)} %{{y:.0f}}%<extra></extra>"))
    style_fig(fig, 300)
    fig.update_yaxes(range=[0, 100], ticksuffix="%")
    show(fig)

# 5. Equity curve
held = signals.dropna(subset=["position"])
wealth = np.exp(held[["strategy_net_ret", "strategy_gross_ret", "bh_ret"]].cumsum())
with panel():
    st.markdown("**Strategy vs buy-and-hold** · growth of $1, log scale")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wealth.index, y=wealth["bh_ret"], name="Buy-and-hold SPY",
                             line=dict(color=SPY_GRAY, width=1.6),
                             hovertemplate="Buy-and-hold $%{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=wealth.index, y=wealth["strategy_net_ret"],
                             name=f"Strategy, after {s['cost_bps_per_switch']:g} bps per switch",
                             line=dict(color=ACCENT, width=2),
                             hovertemplate="Strategy $%{y:.2f}<extra></extra>"))
    style_fig(fig, 360)
    fig.update_yaxes(log_dollar_axis(wealth.min().min(), wealth.max().max()))
    show(fig)
    st.caption(f"From {fmt_date(wealth.index[0])}, out of sample.")

# 6. Metrics, NBER check, data tables, method, footer
left, right = st.columns([3, 2])
with left, panel():
    st.markdown("**Performance, out of sample**")
    order = [("buy_and_hold", "Buy-and-hold"), ("strategy_gross", "Strategy (gross)"),
             ("strategy_net", "Strategy (net)")]
    rows = [("Sharpe ratio", "sharpe", "{:.2f}"), ("Annual volatility", "ann_vol", "{:.1%}"),
            ("Total return", "total_return", "{:.0%}"), ("CAGR", "cagr", "{:.2%}"),
            ("Max drawdown", "max_drawdown", "{:.1%}"), ("Switches", "switches", "{:.0f}"),
            ("Time invested", "pct_invested", "{:.0%}")]
    head_html = "".join(f"<th>{escape(name)}</th>" for _, name in order)
    body_html = "".join(
        f"<tr><td>{escape(title)}</td>"
        + "".join(f'<td class="rm-num">{fmt.format(m[key][k])}</td>' for key, _ in order)
        + "</tr>" for title, k, fmt in rows)
    st.html(f'<div class="rm-scroll"><table class="rm-table"><thead><tr><th></th>{head_html}'
            f'</tr></thead><tbody>{body_html}</tbody></table></div>')
    st.caption(f"{ev['start']} to {ev['end']} ({ev['days']:,} trading days). "
               "Sharpe uses daily log returns and a 0% risk-free rate.")

with right, panel():
    st.markdown(f"**NBER check** · share of days whose most likely state was "
                f"{regime_text('Stressed')}")
    a, b = st.columns(2)
    a.metric("Recession days", f"{nb['pct_recession_days_stressed']:.0%}")
    b.metric("Other days", f"{nb['pct_non_recession_days_stressed']:.0%}")
    st.markdown(" · ".join(f"{r['recession']}: **{r['pct_stressed']:.0%}** of {r['days']} days"
                           for r in nb["per_recession"]))
    st.caption("Only two recessions fall in the test period, so treat this as a sanity check, "
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

with panel():
    st.markdown("**How it works**")
    st.markdown(f"""
- **Data.** Daily SPY prices from {s['start'][:4]} onwards. Two inputs per day: the day's
  log return and the volatility of returns over the past {s['vol_window']} trading days.
- **Model.** A hidden Markov model with three states, fitted only on data before each day
  (an expanding window). It is refit every {s['refit_every']} trading days, trying
  {len(s['seeds'])} random starting points and keeping the best fit.
- **Regimes.** After each refit, the states are named by their average volatility:
  lowest is **{regime_text('Calm')}**, middle is **{regime_text('Normal')}**, highest is
  **{regime_text('Stressed')}**.
- **Probabilities.** Each day's regime probabilities use only information available at
  that day's close, never later data.
- **Rule.** Hold SPY unless the most likely regime is Stressed, in which case hold cash.
  The signal from one close is applied from the next trading day, and each switch costs
  {s['cost_bps_per_switch']:g} basis points.
- **Updates.** A scheduled job adds each new trading day after the close. This page only
  reads the saved results; it never fetches prices or fits the model.
""")

st.html(f'<div class="rm-muted" style="padding:0.5rem 0 1rem">'
        f'<a href="{REPO_URL}">Source code on GitHub</a> · Data as of {as_of:%Y-%m-%d} · '
        "A research project, not investment advice.</div>")
