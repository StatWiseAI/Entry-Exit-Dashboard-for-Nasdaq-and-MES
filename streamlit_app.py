"""
NQ/ES Trade Decision — Streamlit App
=====================================
Run locally:   streamlit run streamlit_app.py
Deploy:        push to GitHub → connect at share.streamlit.app

Architecture
------------
  [CSV upload]  ──┐
  [Topstep API] ──┤──► run_strategy() ──► render_decision()
  [TradingView] ──┘

When live APIs are connected, replace the `load_data_*` functions below.
The dashboard rendering code never changes.
"""

import time
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ── Optional live connectors (imported only when configured) ──────────────────
try:
    from api.topstep_connector import TopstepConnector
    TOPSTEP_AVAILABLE = True
except ImportError:
    TOPSTEP_AVAILABLE = False

try:
    from api.tradingview_connector import get_latest_tv_bars
    TRADINGVIEW_AVAILABLE = True
except ImportError:
    TRADINGVIEW_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NQ/ES Trade Decision",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal dark-mode CSS override ───────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stSidebar"] { background: #111114; }
  .block-container { padding-top: 1.2rem; }
  .signal-box { border-radius: 10px; padding: 20px 24px; margin-bottom: 12px; }
  .signal-long  { background: rgba(34,197,94,.12);  border: 1px solid rgba(34,197,94,.4); }
  .signal-short { background: rgba(239,68,68,.12);  border: 1px solid rgba(239,68,68,.4); }
  .signal-wait  { background: rgba(39,39,42,.6);    border: 1px solid #3f3f46; }
  .signal-warn  { background: rgba(245,158,11,.10); border: 1px solid rgba(245,158,11,.35); }
  .big-label { font-size: 28px; font-weight: 800; font-family: 'Syne', sans-serif; }
  .reason    { font-size: 13px; color: #a1a1aa; margin-top: 6px; font-family: monospace; }
  .level-row { display:flex; justify-content:space-between; padding:8px 0;
               border-bottom:1px solid #27272a; font-family:monospace; font-size:13px; }
  .kpi-grid  { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:8px; }
  .kpi-card  { background:#18181c; border:1px solid #27272a; border-radius:8px;
               padding:10px 14px; }
  .kpi-label { font-size:10px; color:#52525b; text-transform:uppercase;
               letter-spacing:.08em; font-family:monospace; }
  .kpi-value { font-size:18px; font-weight:700; font-family:monospace; margin-top:3px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY ENGINE  (same logic as nq_es_strategy.py)
# ══════════════════════════════════════════════════════════════════════════════
TF_DEFAULTS = {
    "1min":  dict(pullback_thresh=0.0005, hold_bars=10, sl_pct=0.002, tp_pct=0.004,
                  corr_window=20, ema_fast=20, ema_slow=50, momentum_bars=3, corr_thresh=0.70),
    "5min":  dict(pullback_thresh=0.001,  hold_bars=8,  sl_pct=0.003, tp_pct=0.006,
                  corr_window=20, ema_fast=20, ema_slow=50, momentum_bars=3, corr_thresh=0.70),
    "30min": dict(pullback_thresh=0.002,  hold_bars=5,  sl_pct=0.005, tp_pct=0.010,
                  corr_window=20, ema_fast=20, ema_slow=50, momentum_bars=3, corr_thresh=0.70),
    "1hour": dict(pullback_thresh=0.003,  hold_bars=4,  sl_pct=0.007, tp_pct=0.014,
                  corr_window=20, ema_fast=20, ema_slow=50, momentum_bars=3, corr_thresh=0.70),
}
NQ_POINT_VALUE = 20  # $20 per NQ point (E-mini)


def run_strategy(es_df: pd.DataFrame, nq_df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Core strategy — accepts pre-loaded DataFrames."""
    for df in [es_df, nq_df]:
        df.columns = df.columns.str.lower()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
    
    # Re-index after setting timestamp as index
    es = es_df.copy()
    nq = nq_df.copy()
    if "timestamp" in es.columns:
        es["timestamp"] = pd.to_datetime(es["timestamp"])
        es = es.set_index("timestamp")
    if "timestamp" in nq.columns:
        nq["timestamp"] = pd.to_datetime(nq["timestamp"])
        nq = nq.set_index("timestamp")
    
    es = es.sort_index()
    nq = nq.sort_index()

    df = pd.DataFrame(index=nq.index)
    df["NQ_open"]  = nq["open"]
    df["NQ_high"]  = nq["high"]
    df["NQ_low"]   = nq["low"]
    df["NQ_close"] = nq["close"]
    df["NQ_vol"]   = nq["volume"]
    df["ES_close"] = es["close"]
    df = df.dropna()

    cw = params["corr_window"]
    df["ret_NQ"]  = df["NQ_close"].pct_change()
    df["ret_ES"]  = df["ES_close"].pct_change()
    df["corr_20"] = df["ret_NQ"].rolling(cw).corr(df["ret_ES"])
    df["ratio"]   = df["NQ_close"] / df["ES_close"]
    df["ratio_ma"]= df["ratio"].rolling(cw).mean()
    df["nq_stronger"] = df["ratio"] > df["ratio_ma"]
    df["ema_fast"]= df["NQ_close"].ewm(span=params["ema_fast"]).mean()
    df["ema_slow"]= df["NQ_close"].ewm(span=params["ema_slow"]).mean()
    df["trend_nq"]= np.where(df["ema_fast"] > df["ema_slow"], 1, -1)
    df["pullback"]= (df["NQ_close"] - df["ema_fast"]) / df["ema_fast"]
    df["momentum"]= df["NQ_close"] - df["NQ_close"].shift(params["momentum_bars"])

    pt, ct = params["pullback_thresh"], params["corr_thresh"]
    df["long_signal"]  = ((df["trend_nq"]==1) & (df["pullback"]<-pt) &
                          (df["momentum"]>0)   & (df["nq_stronger"]) & (df["corr_20"]>ct))
    df["short_signal"] = ((df["trend_nq"]==-1) & (df["pullback"]>pt) &
                          (df["momentum"]<0)   & (~df["nq_stronger"]) & (df["corr_20"]>ct))
    return df.dropna()


def backtest(df: pd.DataFrame, params: dict) -> list:
    trades, in_trade, trade = [], False, None
    for ts, row in df.iterrows():
        if in_trade:
            ep = trade["entry_price"]; d = trade["direction"]
            exited = False
            if d == "LONG":
                if row["NQ_low"]  <= trade["sl"]: trade.update(exit_price=trade["sl"],  exit_time=str(ts), exit_reason="SL");   exited=True
                elif row["NQ_high"]>= trade["tp"]: trade.update(exit_price=trade["tp"],  exit_time=str(ts), exit_reason="TP");   exited=True
                elif trade["bars_held"] >= params["hold_bars"]: trade.update(exit_price=row["NQ_close"], exit_time=str(ts), exit_reason="TIME"); exited=True
            else:
                if row["NQ_high"] >= trade["sl"]: trade.update(exit_price=trade["sl"],  exit_time=str(ts), exit_reason="SL");   exited=True
                elif row["NQ_low"] <= trade["tp"]: trade.update(exit_price=trade["tp"],  exit_time=str(ts), exit_reason="TP");   exited=True
                elif trade["bars_held"] >= params["hold_bars"]: trade.update(exit_price=row["NQ_close"], exit_time=str(ts), exit_reason="TIME"); exited=True
            if exited:
                pts = (trade["exit_price"]-ep) if d=="LONG" else (ep-trade["exit_price"])
                trade["pnl_pts"] = round(pts, 2)
                trade["pnl_usd"] = round(pts * NQ_POINT_VALUE, 2)
                trades.append(trade); in_trade = False
            else:
                trade["bars_held"] += 1; continue
        if not in_trade:
            ep = float(row["NQ_close"])
            if row["long_signal"]:
                trade = {"direction":"LONG", "entry_price":ep, "entry_time":str(ts), "bars_held":1,
                         "sl":round(ep*(1-params["sl_pct"]),2), "tp":round(ep*(1+params["tp_pct"]),2)}
                in_trade = True
            elif row["short_signal"]:
                trade = {"direction":"SHORT", "entry_price":ep, "entry_time":str(ts), "bars_held":1,
                         "sl":round(ep*(1+params["sl_pct"]),2), "tp":round(ep*(1-params["tp_pct"]),2)}
                in_trade = True
    return trades


def signal_score(row, params) -> int:
    return sum([
        row["trend_nq"] == 1,
        float(row["corr_20"]) > params["corr_thresh"],
        bool(row["nq_stronger"]),
        abs(float(row["pullback"])) > params["pullback_thresh"] * 0.5,
        float(row["momentum"]) != 0,
    ])


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def load_from_uploaded(nq_file, es_file) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load NQ + ES from Streamlit uploaded file objects."""
    nq_df = pd.read_csv(nq_file)
    es_df = pd.read_csv(es_file)
    return nq_df, es_df


def load_from_topstep(tf: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """
    ── LIVE DATA: Topstep / Tradovate ──────────────────────────────────────
    Topstep uses the Tradovate API under the hood.
    Steps to activate:
      1. Install:  pip install tradovate-api  (or use requests)
      2. Set TOPSTEP_USERNAME and TOPSTEP_PASSWORD in Streamlit secrets
         (Settings → Secrets in share.streamlit.app)
      3. Uncomment and complete the connector in api/topstep_connector.py
      4. Return two DataFrames with columns: timestamp, open, high, low, close, volume

    Tradovate API docs: https://api.tradovate.com
    Topstep help:       https://help.topstep.com/en/articles/api
    ────────────────────────────────────────────────────────────────────────
    """
    if not TOPSTEP_AVAILABLE:
        return None
    try:
        connector = TopstepConnector(
            username=st.secrets["TOPSTEP_USERNAME"],
            password=st.secrets["TOPSTEP_PASSWORD"],
        )
        nq_df = connector.get_bars("NQU4", tf)
        es_df = connector.get_bars("ESU4", tf)
        return nq_df, es_df
    except Exception as e:
        st.sidebar.warning(f"Topstep connection failed: {e}")
        return None


def load_from_tradingview(tf: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """
    ── LIVE DATA: TradingView Webhook ──────────────────────────────────────
    TradingView Pine Script sends bar data to your webhook URL via alerts.
    Steps to activate:
      1. Deploy this app publicly (Streamlit Cloud or Railway)
      2. Add the webhook URL to your TradingView alert:
           https://YOUR_APP.streamlit.app/webhook
      3. The Pine Script template is in api/tradingview_connector.py
      4. Bars accumulate in st.session_state["tv_bars"][symbol][tf]

    TradingView webhook docs: https://www.tradingview.com/support/solutions/43000529348
    ────────────────────────────────────────────────────────────────────────
    """
    if not TRADINGVIEW_AVAILABLE:
        return None
    try:
        nq_bars = get_latest_tv_bars("NQ1!", tf)
        es_bars = get_latest_tv_bars("ES1!", tf)
        if nq_bars is None or es_bars is None:
            return None
        return pd.DataFrame(nq_bars), pd.DataFrame(es_bars)
    except Exception as e:
        st.sidebar.warning(f"TradingView data unavailable: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# RENDER HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def render_decision(latest, params: dict):
    is_long  = bool(latest["long_signal"])
    is_short = bool(latest["short_signal"])
    score    = signal_score(latest, params)
    approaching = not is_long and not is_short and score >= 4

    if is_long:
        css = "signal-long"
        label = "▲ ENTER LONG NOW"
        color = "#22c55e"
        reason = (f"NQ uptrend — pulled back {abs(latest['pullback'])*100:.2f}% below EMA{params['ema_fast']}, "
                  f"momentum up, ES correlation {latest['corr_20']:.2f}.")
    elif is_short:
        css = "signal-short"
        label = "▼ ENTER SHORT NOW"
        color = "#ef4444"
        reason = (f"NQ downtrend — extended {abs(latest['pullback'])*100:.2f}% above EMA{params['ema_fast']}, "
                  f"momentum down, ES correlation {latest['corr_20']:.2f}.")
    elif approaching:
        css = "signal-warn"
        label = "⚡ SIGNAL APPROACHING — WATCH NEXT BAR"
        color = "#f59e0b"
        reason = (f"{score}/5 conditions met. Pullback at {abs(latest['pullback'])*100:.3f}% "
                  f"(threshold {params['pullback_thresh']*100:.2f}%).")
    else:
        css = "signal-wait"
        label = "— WAIT  ·  NO TRADE"
        color = "#52525b"
        reason = f"Only {score}/5 conditions aligned. No statistical edge. Stay flat."

    st.markdown(f"""
    <div class="signal-box {css}">
      <div class="big-label" style="color:{color}">{label}</div>
      <div class="reason">{reason}</div>
    </div>
    """, unsafe_allow_html=True)

    return is_long, is_short


def render_trade_plan(latest, params: dict, is_long: bool, is_short: bool):
    if not is_long and not is_short:
        return
    ep = float(latest["NQ_close"])
    sl = round(ep*(1-params["sl_pct"]),2) if is_long else round(ep*(1+params["sl_pct"]),2)
    tp = round(ep*(1+params["tp_pct"]),2) if is_long else round(ep*(1-params["tp_pct"]),2)
    risk_pts   = abs(ep-sl)
    reward_pts = abs(ep-tp)

    st.markdown("**Trade Plan**")
    rows = [
        ("Entry price",   f"{ep:,.2f}",  "← enter at this bar's close"),
        ("Stop-Loss",     f"{sl:,.2f}",  f"  {risk_pts:.1f} pts · ${risk_pts*NQ_POINT_VALUE:,.0f} risk"),
        ("Take-Profit",   f"{tp:,.2f}",  f"  {reward_pts:.1f} pts · ${reward_pts*NQ_POINT_VALUE:,.0f} reward"),
        ("Risk : Reward", f"1 : {params['tp_pct']/params['sl_pct']:.1f}", ""),
        ("Max hold",      f"{params['hold_bars']} bars", "exit at market if TP/SL not hit"),
    ]
    for label, value, note in rows:
        col_a, col_b, col_c = st.columns([2, 2, 3])
        col_a.markdown(f"<span style='color:#52525b;font-size:12px;font-family:monospace'>{label}</span>", unsafe_allow_html=True)
        color = "#3b82f6" if "Entry" in label else "#ef4444" if "Stop" in label else "#22c55e" if "Take" in label else "#f59e0b" if "Risk" in label else "#a1a1aa"
        col_b.markdown(f"<span style='color:{color};font-size:16px;font-weight:700;font-family:monospace'>{value}</span>", unsafe_allow_html=True)
        col_c.markdown(f"<span style='color:#52525b;font-size:11px;font-family:monospace'>{note}</span>", unsafe_allow_html=True)


def render_readiness(latest, params: dict):
    score = signal_score(latest, params)
    pct   = int(score / 5 * 100)
    color = "#22c55e" if pct >= 100 else "#f59e0b" if pct >= 60 else "#52525b"

    st.markdown(f"**Signal Readiness — {pct}%**")
    st.progress(pct / 100)

    checks = [
        ("Trend aligned (EMA fast > slow)",      latest["trend_nq"] == 1),
        ("High NQ/ES correlation",               float(latest["corr_20"]) > params["corr_thresh"]),
        ("NQ outperforming ES",                  bool(latest["nq_stronger"])),
        ("Pullback in range",                    abs(float(latest["pullback"])) > params["pullback_thresh"] * 0.5),
        ("Momentum direction correct",           float(latest["momentum"]) != 0),
    ]
    for label, passed in checks:
        icon = "✅" if passed else "⬜"
        st.markdown(f"<span style='font-family:monospace;font-size:12px'>{icon}  {label}</span>",
                    unsafe_allow_html=True)


def render_backtest_table(trades: list):
    if not trades:
        st.info("No signals fired on this timeframe in the loaded data. Try 5m or 1m.")
        return

    wins     = [t for t in trades if t["pnl_usd"] > 0]
    total_pnl = sum(t["pnl_usd"] for t in trades)
    wr        = len(wins) / len(trades) * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trades",      len(trades))
    c2.metric("Win Rate",    f"{wr:.0f}%")
    c3.metric("Net P&L",     f"${total_pnl:,.0f}", delta=f"${total_pnl:,.0f}")
    g = sum(t["pnl_usd"] for t in wins)
    l = abs(sum(t["pnl_usd"] for t in trades if t["pnl_usd"]<=0))
    c4.metric("Profit Factor", f"{g/l:.2f}" if l > 0 else "∞")

    df_trades = pd.DataFrame([{
        "Dir":        t["direction"],
        "Entry Time": t["entry_time"][:16],
        "Entry ▶":   t["entry_price"],
        "Stop-Loss":  t["sl"],
        "Take-Profit":t["tp"],
        "Exit Time":  (t.get("exit_time") or "")[:16],
        "Exit ▶":    t.get("exit_price"),
        "Reason":     t.get("exit_reason"),
        "Pts":        t.get("pnl_pts"),
        "P&L $":      t.get("pnl_usd"),
    } for t in trades])

    st.dataframe(
        df_trades.style
        .map(lambda v: "color: #22c55e" if v == "LONG" else "color: #ef4444" if v == "SHORT" else "",
             subset=["Dir"])
        .map(lambda v: "color: #22c55e; font-weight: bold" if isinstance(v, (int,float)) and v > 0
             else "color: #ef4444; font-weight: bold" if isinstance(v, (int,float)) and v < 0 else "",
             subset=["P&L $", "Pts"])
        .format({"Entry ▶": "{:,.2f}", "Stop-Loss": "{:,.2f}",
                 "Take-Profit": "{:,.2f}", "Exit ▶": "{:,.2f}",
                 "Pts": "{:+.2f}", "P&L $": "${:+,.0f}"}),
        use_container_width=True,
        height=280,
    )

    # Equity curve
    cum = 0; equity = []
    for t in trades:
        cum += t["pnl_usd"]; equity.append(cum)

    eq_df = pd.DataFrame({"Trade #": range(1, len(equity)+1), "Cumulative P&L": equity})
    st.markdown("**Equity Curve**")
    st.line_chart(eq_df.set_index("Trade #"), color="#22c55e" if equity[-1] >= 0 else "#ef4444")


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def sidebar() -> tuple:
    st.sidebar.markdown("## ⚙️ Settings")

    # Data source
    st.sidebar.markdown("### Data Source")
    data_source = st.sidebar.radio(
        "Select source",
        ["📁 Upload CSV files", "🔴 Topstep (live)", "📺 TradingView (webhook)"],
        index=0,
        help="Live sources require API setup. See README for instructions."
    )

    # Timeframe
    st.sidebar.markdown("### Timeframe")
    tf = st.sidebar.selectbox("Timeframe", ["1min", "5min", "30min", "1hour"], index=2)

    # Auto-refresh (for live sources)
    auto_refresh = False
    refresh_interval = 30
    if "live" in data_source.lower() or "tradingview" in data_source.lower():
        st.sidebar.markdown("### Auto-refresh")
        auto_refresh     = st.sidebar.checkbox("Enable auto-refresh", value=True)
        refresh_interval = st.sidebar.slider("Interval (seconds)", 10, 300, 30)

    # Advanced params
    st.sidebar.markdown("### Strategy Parameters")
    with st.sidebar.expander("Override defaults"):
        params = TF_DEFAULTS[tf].copy()
        params["pullback_thresh"] = st.slider(
            "Pullback threshold", 0.0001, 0.010, params["pullback_thresh"], 0.0001,
            format="%.4f", help="How far price must pull back from EMA to qualify")
        params["sl_pct"] = st.slider(
            "Stop-Loss %", 0.001, 0.02, params["sl_pct"], 0.001, format="%.3f")
        params["tp_pct"] = st.slider(
            "Take-Profit %", 0.001, 0.04, params["tp_pct"], 0.001, format="%.3f")
        params["hold_bars"] = st.slider(
            "Max hold (bars)", 1, 20, params["hold_bars"])
        params["corr_thresh"] = st.slider(
            "Correlation threshold", 0.3, 0.99, params["corr_thresh"], 0.01)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "[![GitHub](https://img.shields.io/badge/GitHub-View_Source-black?logo=github)]"
        "(https://github.com/YOUR_USERNAME/nq-es-dashboard)",
        unsafe_allow_html=True
    )
    st.sidebar.caption("⚠️ Not financial advice. Use at your own risk.")

    return data_source, tf, params, auto_refresh, refresh_interval


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    data_source, tf, params, auto_refresh, refresh_interval = sidebar()

    st.markdown("# NQ/ES Trade Decision")
    st.markdown(
        f"<span style='font-family:monospace;font-size:12px;color:#52525b'>"
        f"Timeframe: {tf} · Updated: {datetime.now().strftime('%H:%M:%S')}</span>",
        unsafe_allow_html=True
    )

    # ── Load data ─────────────────────────────────────────────────────────────
    nq_df, es_df = None, None

    if "Upload" in data_source:
        col1, col2 = st.columns(2)
        with col1:
            nq_file = st.file_uploader("NQ CSV", type="csv", key="nq_upload",
                                        help="Must have: timestamp, open, high, low, close, volume")
        with col2:
            es_file = st.file_uploader("ES CSV", type="csv", key="es_upload")

        if nq_file and es_file:
            nq_df, es_df = load_from_uploaded(nq_file, es_file)
            st.success(f"Loaded: NQ {len(nq_df)} rows · ES {len(es_df)} rows")
        else:
            st.info("👆 Upload both NQ and ES CSV files to begin. "
                    "Use FRD-format files: timestamp, open, high, low, close, volume")
            _show_sample_format()
            return

    elif "Topstep" in data_source:
        if not TOPSTEP_AVAILABLE:
            st.error("Topstep connector not installed. See `api/topstep_connector.py`.")
            st.code("pip install tradovate-api\n# Then configure api/topstep_connector.py")
            return
        result = load_from_topstep(tf)
        if result is None:
            st.error("Could not connect to Topstep. Check your credentials in Streamlit secrets.")
            return
        nq_df, es_df = result
        st.success(f"🔴 Live Topstep feed — {tf} — {len(nq_df)} bars loaded")

    elif "TradingView" in data_source:
        if not TRADINGVIEW_AVAILABLE:
            st.error("TradingView connector not available. See `api/tradingview_connector.py`.")
            return
        result = load_from_tradingview(tf)
        if result is None:
            st.warning("No TradingView data received yet. "
                       "Check your alert webhook URL and Pine Script setup.")
            return
        nq_df, es_df = result

    # ── Run strategy ──────────────────────────────────────────────────────────
    with st.spinner("Running strategy…"):
        df      = run_strategy(es_df, nq_df, params)
        trades  = backtest(df, params)
        latest  = df.iloc[-1]

    # ── Layout: left (decision) / right (chart + trades) ─────────────────────
    left, right = st.columns([1, 2], gap="large")

    with left:
        is_long, is_short = render_decision(latest, params)
        st.markdown("---")
        render_trade_plan(latest, params, is_long, is_short)
        st.markdown("---")
        render_readiness(latest, params)
        st.markdown("---")

        # Quick stats
        wins = [t for t in trades if t["pnl_usd"] > 0]
        wr   = len(wins)/len(trades)*100 if trades else 0
        pnl  = sum(t["pnl_usd"] for t in trades)
        st.markdown("**Historical Performance**")
        st.markdown(f"""
        <div class="kpi-grid">
          <div class="kpi-card">
            <div class="kpi-label">Win Rate</div>
            <div class="kpi-value" style="color:{'#22c55e' if wr>=50 else '#ef4444'}">{wr:.0f}%</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Net P&L</div>
            <div class="kpi-value" style="color:{'#22c55e' if pnl>=0 else '#ef4444'}">${pnl:,.0f}</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Total Trades</div>
            <div class="kpi-value" style="color:#a1a1aa">{len(trades)}</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Last bar</div>
            <div class="kpi-value" style="color:#a1a1aa;font-size:12px">{str(df.index[-1])[:16]}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with right:
        # Price chart
        st.markdown("**NQ Price  ·  EMA20 / EMA50  ·  ▲▼ Signal markers**")

        chart_data = df[["NQ_close", "ema_fast", "ema_slow"]].tail(200).copy()
        chart_data.columns = ["NQ Close", "EMA 20", "EMA 50"]
        st.line_chart(chart_data, color=["#a1a1aa", "#f59e0b", "#ef4444"])

        # Signal markers
        long_bars  = df[df["long_signal"]].tail(50)
        short_bars = df[df["short_signal"]].tail(50)
        if len(long_bars):
            st.success(f"▲ Last LONG signal: {str(long_bars.index[-1])[:16]}  "
                       f"@ {long_bars['NQ_close'].iloc[-1]:,.2f}")
        if len(short_bars):
            st.error(f"▼ Last SHORT signal: {str(short_bars.index[-1])[:16]}  "
                     f"@ {short_bars['NQ_close'].iloc[-1]:,.2f}")

        st.markdown("---")
        st.markdown("**Backtest — All Signals in Loaded Data**")
        render_backtest_table(trades)

    # ── Auto-refresh ──────────────────────────────────────────────────────────
    if auto_refresh and ("live" in data_source.lower() or "tradingview" in data_source.lower()):
        st.markdown(f"<span style='color:#52525b;font-size:11px;font-family:monospace'>"
                    f"Auto-refreshing every {refresh_interval}s…</span>", unsafe_allow_html=True)
        time.sleep(refresh_interval)
        st.rerun()


def _show_sample_format():
    st.markdown("**Expected CSV format:**")
    st.code(
        "timestamp,open,high,low,close,volume\n"
        "2026-04-01 09:30,24100.25,24150.00,24080.50,24130.75,1250\n"
        "2026-04-01 10:00,24130.75,24200.00,24120.00,24185.50,980",
        language="csv"
    )


if __name__ == "__main__":
    main()
