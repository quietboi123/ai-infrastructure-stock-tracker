# AI Infrastructure Portfolio Tracker
# Streamlit app to track a starter "AI Infrastructure" portfolio and suggest rebalancing trades
# Author: ChatGPT

import io
import math
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
weights = _normalize_weights(weights) if not st.sidebar.button("Normalize Weights to 100%") else _normalize_weights(weights)

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

st.sidebar.divider()
lookback_days = st.sidebar.slider("Performance lookback (days)", 30, 365*2, 180, 10)
rebalance_tol = st.sidebar.slider("Rebalance tolerance (drift %)", 0.0, 10.0, 3.0, 0.5)
min_trade_usd = st.sidebar.number_input("Min trade size ($)", 0.0, value=50.0, step=25.0)
fractional_round = st.sidebar.selectbox("Round shares to", [3, 2, 1, 0], index=0)
st.sidebar.divider()
use_demo = st.sidebar.toggle("Use demo holdings (simulate $1,750 funded today)", value=False)

# -----------------------------
# Holdings Input
# -----------------------------
st.subheader("Holdings")
all_tickers = sorted(weights.keys())
prices = {}
price_errors = []
for t in all_tickers:
    try:
        tk = yf.Ticker(t)
        px = getattr(tk, "fast_info", {}).get("lastPrice")
        if px is None or (isinstance(px, float) and not math.isfinite(px)):
            hist = tk.history(period="1d")
            px = float(hist["Close"].iloc[-1])
        prices[t] = float(px)
    except Exception as e:
        price_errors.append(t)
        prices[t] = np.nan
if price_errors:
    st.warning("Failed to fetch: " + ", ".join(price_errors))

holdings = []
if use_demo:
    demo_capital = 1750.0
    for t in all_tickers:
        target_val = demo_capital * weights[t]
        sh = target_val / prices[t] if prices[t] and not np.isnan(prices[t]) else 0.0
        holdings.append({"Ticker": t, "Shares": round(sh, 3)})
    cash = 0.0
else:
    for t in all_tickers:
        holdings.append({"Ticker": t, "Shares": 0.0})
    cash = DEFAULT_CASH

hold_df = pd.DataFrame(holdings).set_index("Ticker")
hold_df = st.data_editor(hold_df, num_rows="fixed", column_config={"Shares": st.column_config.NumberColumn(format="%.6f")}, width=700)
cash = st.number_input("Cash ($)", 0.0, value=float(cash), step=50.0)

vals = []
for t in all_tickers:
    sh = float(hold_df.loc[t, "Shares"]) if t in hold_df.index else 0.0
    px = prices.get(t, np.nan)
    val = sh * px if not np.isnan(px) else np.nan
    vals.append({"Ticker": t, "Shares": sh, "Price": px, "Value": val, "TargetWeight": weights[t]})
val_df = pd.DataFrame(vals).set_index("Ticker")
val_df["Weight"] = val_df["Value"] / (val_df["Value"].sum() + cash)
val_df["DriftPct"] = (val_df["Weight"] - val_df["TargetWeight"]) * 100
st.dataframe(val_df[["Shares","Price","Value","Weight","TargetWeight","DriftPct"]].style.format({
    "Shares": "{:.6f}", "Price": "${:,.2f}", "Value": "${:,.2f}", "Weight": "{:.2%}", "TargetWeight": "{:.2%}", "DriftPct": "{:+.2f}%"
}))
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
    raw = yf.download(list(tickers), start=start.date(), end=end.date(), auto_adjust=True, progress=False, group_by="column")
    if raw is None or len(raw) == 0:
        return pd.DataFrame()
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(name="Close")
    if not isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"].to_frame() if "Close" in raw.columns else raw.iloc[:, [0]].copy()
        if len(tickers) == 1:
            closes.columns = [list(tickers)[0]]
        return closes.dropna(how="all")
    lvl0 = set(raw.columns.get_level_values(0))
    close_key = "Close" if "Close" in lvl0 else ("Adj Close" if "Adj Close" in lvl0 else list(lvl0)[0])
    closes = raw[close_key].copy()
    if isinstance(closes.columns, pd.MultiIndex):
        closes.columns = [c[1] for c in closes.columns]
    keep = [t for t in tickers if t in closes.columns]
    return closes[keep].dropna(how="all")

hist_tickers = set(all_tickers) | set(bench.keys())
px_hist = load_history(hist_tickers, lookback_days)

if not px_hist.empty:
    w_series = pd.Series(weights)
    b_series = pd.Series(bench)
    common_tickers = [t for t in w_series.index if t in px_hist.columns]
    w_series = w_series.loc[common_tickers] / w_series.loc[common_tickers].sum()
    common_b = [t for t in b_series.index if t in px_hist.columns]
    b_series = b_series.loc[common_b] / b_series.loc[common_b].sum()
    ret = px_hist.pct_change().fillna(0.0)
    port_nav = (1 + (ret[common_tickers] * w_series).sum(axis=1)).cumprod()
    bench_nav = (1 + (ret[common_b] * b_series).sum(axis=1)).cumprod()
    st.subheader("Performance")
    colA, colB = st.columns(2)
    colA.metric("Portfolio return", f"{port_nav.iloc[-1]-1:.2%}")
    colB.metric("Benchmark return", f"{bench_nav.iloc[-1]-1:.2%}", delta=f"{(port_nav.iloc[-1]-bench_nav.iloc[-1]):+.2%}")
    st.line_chart(pd.DataFrame({"Portfolio": port_nav, "Benchmark": bench_nav}))
else:
    st.info("Price history not available yet. Try a shorter lookback or check symbols.")

# -----------------------------
# Rebalancing Engine
# -----------------------------
st.subheader("🔁 Rebalance Suggestions")
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
    action, shares, notional = "HOLD", 0.0, 0.0
    if abs(drift) >= rebalance_tol and abs(delta) >= min_trade_usd and px and not np.isnan(px):
        shares = round(delta / px, int(fractional_round))
        if shares != 0:
            notional = shares * px
            action = "BUY" if shares > 0 else "SELL"
            new_cash -= notional
    trade_rows.append({"Ticker": t, "Action": action, "Shares": shares, "Price": px, "Notional": notional, "Current $": cur, "Target $": tgt, "Drift %": drift})
trades_df = pd.DataFrame(trade_rows).set_index("Ticker")
st.dataframe(trades_df[["Action","Shares","Price","Notional","Current $","Target $","Drift %"]].style.format({
    "Shares": "{:.6f}", "Price": "${:,.2f}", "Notional": "${:,.2f}", "Current $": "${:,.2f}", "Target $": "${:,.2f}", "Drift %": "{:+.2f}%"
}))
csv_buf = io.StringIO()
trades_df.reset_index().to_csv(csv_buf, index=False)
st.download_button("📥 Download trade ticket (CSV)", data=csv_buf.getvalue(), file_name=f"rebalance_trades_{datetime.now().date()}.csv", mime="text/csv")
st.metric("Projected cash after trades", f"${new_cash:,.2f}")

# -----------------------------
# Notes
# -----------------------------
st.info("""**Notes**
- This tool does not connect to your broker. Enter your current shares and cash manually.
- Prices come from Yahoo Finance via `yfinance` and can differ slightly from Robinhood.
- Use fractional shares on Robinhood to match suggested quantities.
- Rebalance tolerance avoids churning small trades."""
)
st.caption("Built by ChatGPT. Educational use only; not investment advice.")
