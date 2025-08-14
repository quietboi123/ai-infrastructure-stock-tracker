# AI Infrastructure Portfolio Tracker — with Benchmark label, Re-weight Events, and Configurable Demo $
# Streamlit app to track a starter "AI Infrastructure" portfolio and suggest rebalancing trades
# Author: ChatGPT

import io
import math
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(page_title="AI Infra Portfolio Tracker", layout="wide")
st.title("📈 AI Infrastructure Portfolio Tracker")
st.caption("US-listed, Robinhood-friendly tickers. Tracks performance, re-weight events, and suggests rebalancing.")

# -----------------------------
# Defaults (can be edited in the UI)
# -----------------------------
DEFAULT_WEIGHTS = {
    # Inside the chip
    "SMH": 0.20,
    "MU": 0.10,
    "AMAT": 0.10,
    # Inside the data center
    "VRT": 0.15,
    "SMCI": 0.15,
    # Outside the fence
    "ETN": 0.10,
    "SPXC": 0.10,
    "NEE": 0.10,
}
DEFAULT_CASH = 0.00
DEFAULT_BENCH = {"SMH": 1.0}

# -----------------------------
# Session State (for re-weight events)
# -----------------------------
if "rebalance_events" not in st.session_state:
    # List of dicts: {"date": datetime.date, "weights": {ticker: weight}}
    st.session_state.rebalance_events = []

def _normalize_weights(weight_dict: dict) -> dict:
    w = pd.Series(weight_dict, dtype=float)
    w = w[w > 0]
    if w.empty:
        return {}
    w = w / w.sum()
    return w.to_dict()

def _bench_label(bench_weights: dict) -> str:
    if not bench_weights:
        return "—"
    parts = [f"{t}: {w*100:.1f}%" for t, w in bench_weights.items()]
    return ", ".join(parts)

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("⚙️ Settings")

# Model weights
st.sidebar.subheader("Model Weights")
weights = {}
for t, w in DEFAULT_WEIGHTS.items():
    weights[t] = st.sidebar.number_input(
        f"Target weight – {t}", min_value=0.0, max_value=1.0, value=float(w), step=0.01
    )
weights = _normalize_weights(weights)

# Benchmark
st.sidebar.divider()
bench_choice = st.sidebar.selectbox("Benchmark preset", ["SMH (100%)", "Custom"])
if bench_choice == "SMH (100%)":
    bench = DEFAULT_BENCH.copy()
else:
    st.sidebar.caption("Set custom benchmark weights (normalized to 100%).")
    bench = {}
    for t in weights.keys():
        bench[t] = st.sidebar.number_input(
            f"Benchmark weight – {t}", min_value=0.0, max_value=1.0, value=0.0, step=0.01
        )
    bench = _normalize_weights(bench)

# Lookback / Rebalance params
st.sidebar.divider()
lookback_days = st.sidebar.slider("Performance lookback (days)", min_value=30, max_value=365*2, value=180, step=10)
rebalance_tol = st.sidebar.slider("Rebalance tolerance (drift %)", min_value=0.0, max_value=10.0, value=3.0, step=0.5)
min_trade_usd = st.sidebar.number_input("Min trade size ($)", min_value=0.0, value=50.0, step=25.0)
fractional_round = st.sidebar.selectbox("Round shares to", options=[3, 2, 1, 0], index=0, help="Robinhood allows fractional shares; choose rounding precision.")

# Demo holdings
st.sidebar.divider()
use_demo = st.sidebar.toggle("Use demo holdings", value=False)
demo_amount = st.sidebar.number_input("Demo amount ($)", min_value=0.0, value=1750.0, step=50.0, help="Used only when 'Use demo holdings' is toggled on.")

# -----------------------------
# Re-weight Events (Working Memory)
# -----------------------------
st.subheader("Re-weight Events (Working Memory)")
col_e1, col_e2, col_e3, col_e4 = st.columns([1,1,1,2])
with col_e1:
    event_date = st.date_input("Event date", value=date.today())
with col_e2:
    if st.button("➕ Record event (use current target weights)"):
        st.session_state.rebalance_events.append({"date": event_date, "weights": _normalize_weights(weights)})
with col_e3:
    if st.button("↩️ Undo last event"):
        if st.session_state.rebalance_events:
            st.session_state.rebalance_events = st.session_state.rebalance_events[:-1]
with col_e4:
    if st.button("🧹 Clear all events"):
        st.session_state.rebalance_events = []

if st.session_state.rebalance_events:
    ev_df = pd.DataFrame([
        {"Date": e["date"].isoformat(), **{k: v for k, v in e["weights"].items()}}
        for e in sorted(st.session_state.rebalance_events, key=lambda x: x["date"])
    ])
    st.dataframe(ev_df, use_container_width=True)
else:
    st.caption("No events recorded yet. Tip: add events whenever you *actually* rebalanced. Performance will respect these dates.")

# -----------------------------
# Holdings Input
# -----------------------------
st.subheader("Holdings")
st.caption("Enter current shares and cash. Prices update live via Yahoo Finance.")

# Pull prices first
all_tickers = sorted(weights.keys())
prices = {}
price_errors = []
for t in all_tickers:
    try:
        tk = yf.Ticker(t)
        # Prefer fast_info; fallback to history
        px = getattr(tk, "fast_info", {}).get("lastPrice")
        if px is None or (isinstance(px, float) and not math.isfinite(px)):
            hist = tk.history(period="1d")
            px = float(hist["Close"].iloc[-1])
        prices[t] = float(px)
    except Exception:
        price_errors.append(t)
        prices[t] = np.nan

if price_errors:
    st.warning("Some tickers failed to fetch via yfinance. Check symbols and try again: " + ", ".join(price_errors))

holdings = []
if use_demo:
    # Allocate demo capital to target weights at current prices
    for t in all_tickers:
        target_val = demo_amount * weights[t]
        sh = target_val / prices[t] if prices[t] and not np.isnan(prices[t]) else 0.0
        holdings.append({"Ticker": t, "Shares": round(sh, 3)})
    cash = 0.0
else:
    for t in all_tickers:
        holdings.append({"Ticker": t, "Shares": 0.0})
    cash = DEFAULT_CASH

# Make editable table for shares
hold_df = pd.DataFrame(holdings).set_index("Ticker")
hold_df = st.data_editor(
    hold_df,
    num_rows="fixed",
    column_config={"Shares": st.column_config.NumberColumn(format="%.6f")},
    width=700,
)

cash = st.number_input("Cash ($)", min_value=0.0, value=float(cash), step=50.0)

# Compute current values
vals = []
for t in all_tickers:
    sh = float(hold_df.loc[t, "Shares"]) if t in hold_df.index else 0.0
    px = prices.get(t, np.nan)
    val = sh * px if not np.isnan(px) else np.nan
    vals.append({"Ticker": t, "Shares": sh, "Price": px, "Value": val, "TargetWeight": weights[t]})

val_df = pd.DataFrame(vals).set_index("Ticker")
val_df["Weight"] = val_df["Value"] / (val_df["Value"].sum() + cash)
val_df["DriftPct"] = (val_df["Weight"] - val_df["TargetWeight"]) * 100

# Display holdings table with computed fields
show_cols = ["Shares", "Price", "Value", "Weight", "TargetWeight", "DriftPct"]
fmt = {
    "Shares": "{:.6f}",
    "Price": "${:,.2f}",
    "Value": "${:,.2f}",
    "Weight": "{:.2%}",
    "TargetWeight": "{:.2%}",
    "DriftPct": "{:+.2f}%",
}
st.dataframe(val_df[show_cols].style.format(fmt))

portfolio_value = float(val_df["Value"].sum() + cash)
st.metric("Portfolio Value", f"${portfolio_value:,.2f}")

# -----------------------------
# Performance vs Benchmark (with re-weight events)
# -----------------------------
@st.cache_data(ttl=600)
def load_history(tickers, days):
    if not tickers:
        return pd.DataFrame()
    end = datetime.now()
    start = end - timedelta(days=int(days) + 7)
    # Use auto_adjust so we can just rely on 'Close'
    raw = yf.download(
        list(tickers),
        start=start.date(),
        end=end.date(),
        auto_adjust=True,
        progress=False,
        group_by="column"
    )
    if raw is None or len(raw) == 0:
        return pd.DataFrame()
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(name="Close")
    # If single-index columns
    if not isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"].to_frame() if "Close" in raw.columns else raw.iloc[:, [0]].copy()
        if len(tickers) == 1:
            closes.columns = [list(tickers)[0]]
        return closes.dropna(how="all")
    # MultiIndex: pick Close level
    lvl0 = set(raw.columns.get_level_values(0))
    close_key = "Close" if "Close" in lvl0 else ("Adj Close" if "Adj Close" in lvl0 else list(lvl0)[0])
    closes = raw[close_key].copy()
    if isinstance(closes.columns, pd.MultiIndex):
        closes.columns = [c[1] for c in closes.columns]
    keep = [t for t in tickers if t in closes.columns]
    return closes[keep].dropna(how="all")

hist_tickers = set(all_tickers) | set(bench.keys())
px_hist = load_history(hist_tickers, lookback_days)

def compute_nav_with_events(px: pd.DataFrame, ret_weights_events: list, tickers_for_weights: list):
    """
    px: price history (columns = tickers)
    ret_weights_events: list of {"date": datetime.date, "weights": dict}, sorted by date
    tickers_for_weights: the tickers we consider for weights (e.g., holdings tickers)
    Returns a NAV series starting at 1.0, using piecewise-constant weights across event segments.
    """
    if px.empty:
        return pd.Series(dtype=float)

    # Daily returns
    ret = px.pct_change().fillna(0.0)

    # Build a weights schedule over the index
    idx = ret.index
    # If no events, treat as single event at the first date
    events = sorted(ret_weights_events, key=lambda e: e["date"]) if ret_weights_events else []
    if not events:
        events = [{"date": idx[0].date(), "weights": _normalize_weights({k: DEFAULT_WEIGHTS.get(k, 0) for k in tickers_for_weights})}]
    # Filter events within the index range; ensure one at/ before start
    start_date = idx[0].date()
    events_in = [e for e in events if e["date"] <= idx[-1].date()]
    if not any(e["date"] <= start_date for e in events_in):
        # Prepend a start event using the earliest available event's weights
        base_w = events_in[0]["weights"] if events_in else _normalize_weights({k: DEFAULT_WEIGHTS.get(k, 0) for k in tickers_for_weights})
        events_in = [{"date": start_date, "weights": base_w}] + events_in
    # Collapse duplicate dates by keeping the last on that day
    dedup = {}
    for e in events_in:
        dedup[e["date"]] = e["weights"]
    events_sorted = [{"date": d, "weights": dedup[d]} for d in sorted(dedup.keys())]

    # Create a DataFrame of weights aligned to index
    wdf = pd.DataFrame(0.0, index=idx, columns=px.columns)
    current_w = None
    event_ptr = 0
    for ts in idx:
        d = ts.date()
        # Advance event pointer if next event date has arrived
        while event_ptr < len(events_sorted) and events_sorted[event_ptr]["date"] <= d:
            # Normalize and store only tickers present in px
            ww = {t: events_sorted[event_ptr]["weights"].get(t, 0.0) for t in px.columns}
            current_w = _normalize_weights(ww)
            event_ptr += 1
        if current_w is None:
            # Before first event on index; use zeros
            continue
        for t, w in current_w.items():
            if t in wdf.columns:
                wdf.at[ts, t] = w

    # Compute weighted daily returns
    port_ret = (ret * wdf).sum(axis=1)
    nav = (1 + port_ret).cumprod()
    return nav

if not px_hist.empty:
    # Subset columns for portfolio and for benchmark
    port_cols = [t for t in weights.keys() if t in px_hist.columns]
    bench_cols = [t for t in bench.keys() if t in px_hist.columns]

    # Build events list for portfolio performance using current session events
    events_for_perf = st.session_state.rebalance_events.copy()
    # If you want the *current* target weights to apply from today onward (even if not added as event),
    # you can append a "today" event implicitly:
    # events_for_perf.append({"date": date.today(), "weights": _normalize_weights(weights)})

    # Compute NAVs
    port_nav = compute_nav_with_events(px_hist[port_cols], events_for_perf, port_cols)
    # Benchmark NAV uses constant weights (no re-weight events)
    bench_w_series = pd.Series(bench)
    if not bench_w_series.empty:
        bench_w_series = bench_w_series.loc[bench_cols] / bench_w_series.loc[bench_cols].sum() if bench_cols else bench_w_series
    ret = px_hist[bench_cols].pct_change().fillna(0.0)
    bench_nav = (1 + (ret * bench_w_series).sum(axis=1)).cumprod() if not bench_w_series.empty else pd.Series(index=ret.index, data=1.0)

    st.subheader("Performance")
    st.caption(f"Benchmark: {_bench_label(bench)}")
    colA, colB = st.columns(2)
    if not port_nav.empty:
        colA.metric("Portfolio return", f"{(port_nav.iloc[-1]-1):.2%}")
    if not bench_nav.empty:
        colB.metric("Benchmark return", f"{(bench_nav.iloc[-1]-1):.2%}", delta=f"{(port_nav.iloc[-1]-bench_nav.iloc[-1]):+.2%}")

    # Align for chart
    chart_df = pd.DataFrame({"Portfolio": port_nav, "Benchmark": bench_nav}).dropna(how="all")
    st.line_chart(chart_df)
else:
    st.info("Price history not available yet. Try a shorter lookback or check symbols.")

# -----------------------------
# Rebalancing Engine (trade ideas)
# -----------------------------
st.subheader("🔁 Rebalance Suggestions")

# Target values and trades
target_values = {t: portfolio_value * weights[t] for t in all_tickers}
current_values = {t: float(val_df.loc[t, "Value"]) for t in all_tickers}

trade_rows = []
new_cash = cash
for t in all_tickers:
    px = prices[t]
    cur = current_values[t]
    tgt = target_values[t]
    delta = tgt - cur
    drift = float(val_df.loc[t, "DriftPct"]) if t in val_df.index else 0.0

    action = "HOLD"
    shares = 0.0
    notional = 0.0

    # Only trade if drift exceeds tolerance and trade meets min size
    if abs(drift) >= rebalance_tol and abs(delta) >= min_trade_usd and px and not np.isnan(px):
        shares = round(delta / px, int(fractional_round))
        if shares != 0:
            notional = shares * px
            action = "BUY" if shares > 0 else "SELL"
            new_cash -= notional

    trade_rows.append({
        "Ticker": t,
        "Action": action,
        "Shares": shares,
        "Price": px,
        "Notional": notional,
        "Current $": cur,
        "Target $": tgt,
        "Drift %": drift,
    })

trades_df = pd.DataFrame(trade_rows).set_index("Ticker")

st.dataframe(trades_df[["Action", "Shares", "Price", "Notional", "Current $", "Target $", "Drift %"]].style.format({
    "Shares": "{:.6f}",
    "Price": "${:,.2f}",
    "Notional": "${:,.2f}",
    "Current $": "${:,.2f}",
    "Target $": "${:,.2f}",
    "Drift %": "{:+.2f}%",
}))

# Download trade ticket
csv_buf = io.StringIO()
trades_df.reset_index().to_csv(csv_buf, index=False)
st.download_button("📥 Download trade ticket (CSV)", data=csv_buf.getvalue(), file_name=f"rebalance_trades_{datetime.now().date()}.csv", mime="text/csv")

# Cash after proposed trades
st.metric("Projected cash after trades", f"${new_cash:,.2f}")

# -----------------------------
# Notes
# -----------------------------
st.info(
    """
    **Notes**
    - This tool does not connect to your broker. Enter your current shares and cash manually.
    - Prices come from Yahoo Finance via `yfinance` and can differ slightly from Robinhood.
    - Re-weight events: add one whenever you actually rebalance; performance will respect those dates.
    - Use fractional shares on Robinhood to match suggested quantities (or round as you prefer).
    - Rebalance tolerance avoids churning small trades. Increase it if you prefer fewer trades.
    """
)
st.caption("Built by ChatGPT. Educational use only; not investment advice.")
