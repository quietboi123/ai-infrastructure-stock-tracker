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
        closes = raw["Close"].to_frame() if "Close" in raw.columns else raw.iloc[:, [0]]()_
