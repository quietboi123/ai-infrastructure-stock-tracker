# AI Infrastructure Portfolio Tracker
# Streamlit app to track a starter "AI Infrastructure" portfolio and suggest rebalancing trades
# Author: ChatGPT
# 
# Features
# - Default model weights based on "AI infrastructure" thesis (US-listed only)
# - Live prices via yfinance
# - Enter current holdings (shares) + cash; computes current allocation & drift
# - Rebalancing suggestions to target weights with tolerance bands
# - Downloadable CSV trade ticket
# - Benchmark vs SMH (or custom)
# - Mobile-friendly Streamlit layout

import io
import math
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(page_title="AI Infra Portfolio Tracker", layout="wide")
st.title("📈 AI Infrastructure Portfolio Tracker")
st.caption("US-listed, Robinhood-friendly tickers. Tracks performance and suggests rebalancing.")

# -----------------------------
# Defaults (can be edited in the UI)
# -----------------------------
DEFAULT_WEIGHTS = {
    # Inside the chip
    "SMH": 0.20,  # Semiconductor ETF core
    "MU": 0.10,   # Micron (HBM)
    "AMAT": 0.10, # Applied Materials (semicap)
    # Inside the data center
    "VRT": 0.15,  # Vertiv (power & cooling)
    "SMCI": 0.15, # Super Micro Computer (AI servers)
    # Outside the fence
    "ETN": 0.10,  # Eaton (electrical)
    "SPXC": 0.10, # SPX Technologies (transformers)
    "NEE": 0.10,  # NextEra Energy (utility)
}

DEFAULT_CASH = 0.00
DEFAULT_BENCH = {"SMH": 1.0}

# Helper to normalize weights to 1.0

def _normalize_weights(weight_dict: dict) -> dict:
    w = pd.Series(weight_dict, dtype=float)
    w = w[w > 0]
    if w.empty:
        return {}
    w = w / w.sum()
    return w.to_dict()

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("⚙️ Settings")

st.sidebar.subheader("Model Weights")
weights = {}
for t, w in DEFAULT_WEIGHTS.items():
    weights[t] = st.sidebar.number_input(
        f"Target weight – {t}", min_value=0.0, max_value=1.0, value=float(w), step=0.01
    )

if st.sidebar.button("Normalize Weights to 100%"):
    weights = _normalize_weights(weights)
else:
    # Always keep a normalized copy for calculations
    weights = _normalize_weights(weights)

st.sidebar.divider()
bench_choice = st.sidebar.selectbox("Benchmark preset", ["SMH (100%)", "Custom"])
if bench_choice == "SMH (100%)":
    bench = DEFAULT_BENCH.copy()
else:
    st.sidebar.caption("Set custom benchmark weights (normalized to 100%).")
    bench = {}
    for t in weights.keys():
        bench[t] = st.sidebar.number_input(
            f"Benchmark weight – {t}", min_value=0.0, max_value=1.0, value=float(0.0), step=0.01
        )
    bench = _normalize_weights(bench)

st.sidebar.divider()
lookback_days = st.sidebar.slider("Performance lookback (days)", min_value=30, max_value=365*2, value=180, step=10)
rebalance_tol = st.sidebar.slider("Rebalance tolerance (drift %)", min_value=0.0, max_value=10.0, value=3.0, step=0.5)
min_trade_usd = st.sidebar.number_input("Min trade size ($)", min_value=0.0, value=50.0, step=25.0)
fractional_round = st.sidebar.selectbox("Round shares to", options=[3, 2, 1, 0], index=0, help="Robinhood allows fractional shares; choose rounding precision.")

st.sidebar.divider()
use_demo = st.sidebar.toggle("Use demo holdings (simulate $1,750 funded today)", value=False)

# -----------------------------
# Holdings Input
# -----------------------------
st.subheader("Holdings")
st.caption("Enter current shares and cash. Prices update live via Yahoo Finance.")

cols = st.columns([1,1,1,1,1])
cols[0].markdown("**Ticker**")
cols[1].markdown("**Shares**")
cols[2].markdown("**Last Price**")
cols[3].markdown("**Value ($)**")
cols[4].markdown("**Target Weight**")

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
    except Exception as e:
        price_errors.append((t, str(e)))
        prices[t] = np.nan

if price_errors:
    st.warning("Some tickers failed to fetch via yfinance. Check symbols and try again: " + ", ".join([p[0] for p in price_errors]))

holdings = []

if use_demo:
    demo_capital = 1750.0
    # Allocate demo capital to target weights at current prices
    for t in all_tickers:
        target_val = demo_capital * weights[t]
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
# Performance vs Benchmark
# -----------------------------

@st.cache_data(ttl=600)
def load_history(tickers, days):
    if not tickers:
        return pd.DataFrame()
    end = datetime.now()
    start = end - timedelta(days=int(days)+7)
    data = yf.download(list(tickers), start=start.date(), end=end.date(), progress=False)["Adj Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    data = data.dropna(how="all")
    return data

hist_tickers = set(all_tickers) | set(bench.keys())
px_hist = load_history(hist_tickers, lookback_days)

if not px_hist.empty:
    # Compute portfolio & benchmark NAVs
    w_series = pd.Series(weights)
    b_series = pd.Series(bench)

    # Convert to normalized weights over the intersection present in px_hist
    common_tickers = [t for t in w_series.index if t in px_hist.columns]
    w_series = w_series.loc[common_tickers]
    w_series = w_series / w_series.sum()

    common_b = [t for t in b_series.index if t in px_hist.columns]
    b_series = b_series.loc[common_b]
    b_series = b_series / b_series.sum()

    ret = px_hist.pct_change().fillna(0.0)
    port_nav = (1 + (ret[common_tickers] * w_series).sum(axis=1)).cumprod()
    bench_nav = (1 + (ret[common_b] * b_series).sum(axis=1)).cumprod()

    st.subheader("Performance")
    colA, colB = st.columns(2)
    port_total = port_nav.iloc[-1] - 1
    bench_total = bench_nav.iloc[-1] - 1
    colA.metric("Portfolio (model weights) return", f"{port_total:.2%}")
    colB.metric("Benchmark return", f"{bench_total:.2%}", delta=f"{(port_total-bench_total):+.2%}")

    st.line_chart(pd.DataFrame({"Portfolio": port_nav, "Benchmark": bench_nav}))
else:
    st.info("Price history not available yet. Try a shorter lookback or check symbols.")

# -----------------------------
# Rebalancing Engine
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
trades_export = trades_df.reset_index()
trades_export.to_csv(csv_buf, index=False)
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
    - Use fractional shares on Robinhood to match suggested quantities (or round as you prefer).
    - Rebalance tolerance avoids churning small trades. Increase it if you prefer fewer trades.
    """
)

st.caption("Built by ChatGPT. Educational use only; not investment advice.")
