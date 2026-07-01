"""
streamlit_app.py — AI-Powered NQ/ES Trade Decision System
==========================================================
Three theoretical streams in one interface:
  1. SCM decision models (EOQ sizing, Newsvendor SL/TP, safety stock, bullwhip)
  2. Conversational AI with persistent trader memory (Claude claude-sonnet-4-6)
  3. Real-time external market intelligence (VIX, macro calendar, news, order flow)

Run locally:  streamlit run streamlit_app.py
Deploy:       push to GitHub -> connect at share.streamlit.app
"""

import os, time, json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NQ/ES AI Trade Decision",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stSidebar"]{background:#0d0f13}
.block-container{padding-top:1rem}
.decision-card{border-radius:12px;padding:20px 24px;margin-bottom:12px;border:1px solid}
.dc-long {background:rgba(34,197,94,.10);border-color:rgba(34,197,94,.35)}
.dc-short{background:rgba(239,68,68,.10);border-color:rgba(239,68,68,.35)}
.dc-wait {background:rgba(39,39,42,.5); border-color:#3f3f46}
.dc-warn {background:rgba(245,158,11,.08);border-color:rgba(245,158,11,.3)}
.big-sig {font-size:28px;font-weight:800;margin-bottom:6px}
.reason  {font-size:13px;color:#a1a1aa;line-height:1.55;font-family:monospace}
.note    {font-size:12px;color:#71717a;margin-top:8px;font-style:italic}
.warn-box{background:rgba(239,68,68,.07);border:1px solid rgba(239,68,68,.25);border-radius:8px;
          padding:8px 12px;font-size:12px;color:#fca5a5;margin-top:8px;font-family:monospace}
.kpi-card{background:#18181c;border:1px solid #27272a;border-radius:8px;padding:10px 14px}
.kpi-l   {font-size:10px;color:#52525b;text-transform:uppercase;letter-spacing:.08em;font-family:monospace}
.kpi-v   {font-size:18px;font-weight:700;font-family:monospace;margin-top:3px}
.chat-user{background:#1c1c1f;border-radius:8px;padding:10px 14px;margin:6px 0;
           font-family:monospace;font-size:13px;color:#e4e4e7}
.chat-agent{background:#18181b;border-left:3px solid #22c55e;border-radius:0 8px 8px 0;
            padding:10px 14px;margin:6px 0;font-family:monospace;font-size:13px;color:#a1a1aa}
.conf-bar{height:6px;border-radius:3px;margin-top:6px}
</style>
""", unsafe_allow_html=True)

# ── Strategy engine import ────────────────────────────────────────────────────
try:
    from nq_es_strategy import run_strategy, build_signal_context, backtest, performance_summary, TF_DEFAULTS
except ImportError:
    st.error("nq_es_strategy.py not found. Place it in the same directory.")
    st.stop()

# ── Agent import (graceful degradation if Anthropic not configured) ───────────
AGENT_AVAILABLE = False
try:
    from agent.trader_agent import (
        TraderAgent, TraderMemory, TraderProfile, SessionState,
        SignalContext, MarketContext, TradeRecord, SCMDecisionModels
    )
    from agent.intelligence_layer import build_market_context
    AGENT_AVAILABLE = True
except ImportError:
    pass

NQ_POINT_VALUE = 20


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════════
def _init_state():
    defaults = {
        "agent":           None,
        "memory":          None,
        "chat_messages":   [],
        "last_signal":     None,
        "last_market_ctx": None,
        "last_decision":   None,
        "df":              None,
        "trades":          [],
        "perf":            {},
        "tf":              "30min",
        "data_loaded":     False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


def _get_agent() -> "TraderAgent | None":
    if not AGENT_AVAILABLE:
        return None
    if st.session_state.agent is None:
        # Load whichever key is available into env so llm_client.py can find it
        for key_name in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY"):
            if not os.environ.get(key_name):
                try:
                    val = st.secrets.get(key_name, "")
                    if val:
                        os.environ[key_name] = val
                except Exception:
                    pass
        mem = TraderMemory(Path("trader_memory.json"))
        st.session_state.agent  = TraderAgent(mem)
        st.session_state.memory = mem
    return st.session_state.agent


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    st.sidebar.markdown("## ⚙️ Settings")

    # ── Data source ──
    st.sidebar.markdown("### Data source")
    source = st.sidebar.radio(
        "Source", ["📁 Upload CSV", "🔴 Topstep live", "📺 TradingView webhook"],
        index=0, label_visibility="collapsed"
    )

    # ── Timeframe ──
    st.sidebar.markdown("### Timeframe")
    tf = st.sidebar.selectbox("Timeframe", list(TF_DEFAULTS.keys()), index=3,
                               label_visibility="collapsed")

    # ── Auto-refresh ──
    auto_refresh = False; refresh_sec = 30
    if source != "📁 Upload CSV":
        st.sidebar.markdown("### Auto-refresh")
        auto_refresh = st.sidebar.toggle("Enable", value=True)
        refresh_sec  = st.sidebar.slider("Seconds", 10, 300, 30)

    # ── Strategy params ──
    st.sidebar.markdown("### Strategy parameters")
    params = TF_DEFAULTS[tf].copy()
    with st.sidebar.expander("Override defaults"):
        params["pullback_thresh"] = st.slider("Pullback threshold",   0.0001, 0.010, params["pullback_thresh"], 0.0001, format="%.4f")
        params["sl_pct"]          = st.slider("Stop-Loss %",          0.001,  0.020, params["sl_pct"],          0.001,  format="%.3f")
        params["tp_pct"]          = st.slider("Take-Profit %",        0.001,  0.040, params["tp_pct"],          0.001,  format="%.3f")
        params["hold_bars"]       = st.slider("Max hold (bars)",      1,      20,    params["hold_bars"])
        params["corr_thresh"]     = st.slider("Correlation threshold",0.3,    0.99,  params["corr_thresh"],     0.01)

    # ── Trader profile (for agent personalisation) ──
    if AGENT_AVAILABLE:
        st.sidebar.markdown("### Trader profile")
        with st.sidebar.expander("Edit profile"):
            agent = _get_agent()
            if agent:
                p = agent.memory.profile
                new_name    = st.text_input("Name",            p.name)
                new_account = st.number_input("Account ($)",   value=p.account_size_usd, step=5000.0)
                new_risk    = st.slider("Risk per trade (%)",  0.1, 5.0, p.risk_per_trade_pct, 0.1)
                new_style   = st.text_area("Trading style",   p.style_notes, height=60)
                new_biases  = st.text_input("Known biases (comma-separated)", ", ".join(p.known_biases))
                new_state   = st.selectbox("Emotional state today",
                                           ["neutral","elevated","fatigued","overconfident"],
                                           index=["neutral","elevated","fatigued","overconfident"].index(agent.memory.session.emotional_state))
                if st.button("Save profile"):
                    agent.update_profile(
                        name=new_name, account_size_usd=new_account,
                        risk_per_trade_pct=new_risk, style_notes=new_style,
                        known_biases=[b.strip() for b in new_biases.split(",") if b.strip()],
                    )
                    agent.set_emotional_state(new_state)
                    st.success("Profile saved.")

    st.sidebar.markdown("---")

    # ── AI provider status badge ──────────────────────────────────────────────
    if AGENT_AVAILABLE:
        try:
            from agent.llm_client import provider_status
            ps = provider_status()
            badge_color = ps["color"]
            badge_text  = ps["badge"]
            model_text  = ps["model"]
            badge_html  = (
                "<div style='font-family:monospace;font-size:11px;padding:8px 10px;"
                f"background:#18181c;border:1px solid #27272a;border-radius:6px;"
                f"color:{badge_color}'>"
                f"{badge_text}<br>"
                f"<span style='color:#52525b'>Model: {model_text}</span>"
                "</div>"
            )
            st.sidebar.markdown(badge_html, unsafe_allow_html=True)
            if ps["status"] == "unconfigured":
                st.sidebar.info(
                    "**Set up AI agent (free):**\n\n"
                    "1. Get free key → [aistudio.google.com](https://aistudio.google.com/app/apikey)\n"
                    "2. Add to Streamlit secrets:\n"
                    "   `GEMINI_API_KEY = \"AIza...\"`\n\n"
                    "Upgrade later: add `ANTHROPIC_API_KEY`"
                )
        except Exception:
            pass

    st.sidebar.markdown("---")
    st.sidebar.caption("⚠️ Not financial advice. Rules-based + AI decision support only.")

    return source, tf, params, auto_refresh, refresh_sec


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_data(source: str, tf: str, params: dict):
    if source == "📁 Upload CSV":
        col1, col2 = st.columns(2)
        with col1:
            nq_file = st.file_uploader("NQ CSV", type="csv", key="nq_upload",
                                        help="Columns: timestamp, open, high, low, close, volume")
        with col2:
            es_file = st.file_uploader("ES CSV", type="csv", key="es_upload")
        if not nq_file or not es_file:
            st.info("Upload both NQ and ES CSV files to begin.")
            _show_csv_format()
            return None, None
        return pd.read_csv(nq_file), pd.read_csv(es_file)

    elif source == "🔴 Topstep live":
        try:
            from api.topstep_connector import TopstepConnector
            conn  = TopstepConnector(st.secrets["TOPSTEP_USERNAME"], st.secrets["TOPSTEP_PASSWORD"])
            nq_df = conn.get_bars("NQ", tf, n_bars=500)
            es_df = conn.get_bars("ES", tf, n_bars=500)
            st.success(f"🔴 Live Topstep — {tf} — {len(nq_df)} bars")
            return nq_df, es_df
        except Exception as e:
            st.error(f"Topstep connection failed: {e}")
            return None, None

    elif source == "📺 TradingView webhook":
        try:
            from api.tradingview_connector import get_latest_tv_bars
            nq_df = get_latest_tv_bars("NQ", tf)
            es_df = get_latest_tv_bars("ES", tf)
            if nq_df is None or es_df is None:
                st.warning("No TradingView data yet. Check your Pine Script alerts and webhook URL.")
                return None, None
            return nq_df, es_df
        except Exception as e:
            st.error(f"TradingView data unavailable: {e}")
            return None, None

    return None, None


def _show_csv_format():
    st.markdown("**Expected CSV format (FRD / standard OHLCV):**")
    st.code("timestamp,open,high,low,close,volume\n"
            "2026-04-01 09:30,24100.25,24150.00,24080.50,24130.75,1250\n"
            "2026-04-01 10:00,24130.75,24200.00,24120.00,24185.50,980", language="csv")


# ══════════════════════════════════════════════════════════════════════════════
# DECISION RENDERING
# ══════════════════════════════════════════════════════════════════════════════
def render_decision_card(decision: dict | None, signal_ctx: dict):
    """Render the main decision card — agent output if available, else rule-based."""
    if decision:
        action     = decision.get("action", "WAIT")
        headline   = decision.get("headline", "Stand by")
        reason     = decision.get("reason", "")
        note       = decision.get("agent_note", "")
        confidence = decision.get("confidence_pct", 0)
        warnings   = decision.get("warnings", [])
    else:
        # Fallback: pure rule-based
        sig = signal_ctx.get("signal", "NONE")
        action     = "ENTER_LONG" if sig == "LONG" else "ENTER_SHORT" if sig == "SHORT" else "WAIT"
        score      = signal_ctx.get("score", 0)
        approaching= signal_ctx.get("approaching", False)
        if approaching: action = "APPROACHING"
        headline   = {
            "ENTER_LONG":  "▲ Enter long now",
            "ENTER_SHORT": "▼ Enter short now",
            "APPROACHING": "⚡ Signal approaching — watch next bar",
            "WAIT":        "— Wait  ·  no trade",
        }.get(action, "— Wait")
        confidence = int(signal_ctx.get("confidence", 0) * 100)
        reason     = f"{score}/5 conditions aligned." if action == "WAIT" else \
                     f"All 5 conditions met on {signal_ctx.get('tf','?')} timeframe."
        note       = ""; warnings = []

    is_long   = "LONG"  in action
    is_short  = "SHORT" in action
    is_wait   = action == "WAIT"
    is_near   = "APPROACH" in action

    css   = "dc-long" if is_long else "dc-short" if is_short else "dc-warn" if is_near else "dc-wait"
    color = "#22c55e" if is_long else "#ef4444" if is_short else "#f59e0b" if is_near else "#52525b"

    st.markdown(f"""
    <div class="decision-card {css}">
      <div class="big-sig" style="color:{color}">{headline}</div>
      <div class="reason">{reason}</div>
      {"<div class='note'>" + note + "</div>" if note else ""}
      {"".join(f"<div class='warn-box'>⚠ {w}</div>" for w in warnings)}
    </div>
    """, unsafe_allow_html=True)

    # Confidence bar
    bar_color = "#22c55e" if confidence >= 70 else "#f59e0b" if confidence >= 40 else "#52525b"
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px">
      <div style="font-family:monospace;font-size:11px;color:#52525b;min-width:80px">Signal strength</div>
      <div style="flex:1;background:#27272a;border-radius:3px;height:6px">
        <div class="conf-bar" style="width:{confidence}%;background:{bar_color}"></div>
      </div>
      <div style="font-family:monospace;font-size:11px;color:{bar_color};min-width:32px;text-align:right">{confidence}%</div>
    </div>
    """, unsafe_allow_html=True)

    return is_long, is_short


def render_trade_plan(signal_ctx: dict, decision: dict | None, params: dict):
    """Show the exact trade plan when a signal is active."""
    sig = signal_ctx.get("signal", "NONE")
    if sig == "NONE":
        return

    # Use SCM-sized plan from agent if available, else raw signal levels
    if decision and decision.get("trade_plan"):
        tp_data = decision["trade_plan"]
        ep      = tp_data.get("entry",          signal_ctx["entry_price"])
        sl      = tp_data.get("sl",             signal_ctx["sl"])
        tp      = tp_data.get("tp",             signal_ctx["tp"])
        contr   = tp_data.get("contracts",      1)
        rr      = tp_data.get("rr_ratio",       signal_ctx["rr_ratio"])
        hold    = tp_data.get("max_hold_bars",  params["hold_bars"])
        nv_note = tp_data.get("newsvendor_note","")
    else:
        ep    = signal_ctx["entry_price"]; sl = signal_ctx["sl"]; tp = signal_ctx["tp"]
        contr = 1; rr = signal_ctx["rr_ratio"]; hold = params["hold_bars"]; nv_note = ""

    risk_pts   = abs(ep - sl);   reward_pts = abs(ep - tp)
    risk_usd   = round(risk_pts  * NQ_POINT_VALUE * contr, 0)
    reward_usd = round(reward_pts* NQ_POINT_VALUE * contr, 0)

    st.markdown("**Trade plan**")
    rows = [
        ("Entry",         f"{ep:,.2f}",          "← enter at this bar's close"),
        ("Stop-Loss",     f"{sl:,.2f}",           f"{risk_pts:.1f} pts · ${risk_usd:,.0f} ({contr} contract{'s' if contr!=1 else ''})"),
        ("Take-Profit",   f"{tp:,.2f}",           f"{reward_pts:.1f} pts · ${reward_usd:,.0f}"),
        ("Risk : Reward", f"1 : {rr:.1f}",        ""),
        ("Max hold",      f"{hold} bars",          "exit at market if not hit"),
        ("Contracts",     str(contr),              "SCM EOQ-sized" if decision else "default"),
    ]
    if nv_note:
        rows.append(("Newsvendor",    nv_note[:60],      "SL/TP calibration"))

    for label, value, note in rows:
        vc = "#3b82f6" if "Entry" in label else "#ef4444" if "Stop" in label \
             else "#22c55e" if "Take" in label else "#f59e0b" if "Risk" in label else "#a1a1aa"
        c1, c2, c3 = st.columns([2,2,3])
        c1.markdown(f"<span style='font-family:monospace;font-size:11px;color:#52525b'>{label}</span>", unsafe_allow_html=True)
        c2.markdown(f"<span style='font-family:monospace;font-size:15px;font-weight:700;color:{vc}'>{value}</span>", unsafe_allow_html=True)
        c3.markdown(f"<span style='font-family:monospace;font-size:10px;color:#52525b'>{note}</span>", unsafe_allow_html=True)


def render_readiness(signal_ctx: dict, params: dict):
    score = signal_ctx.get("score", 0)
    pct   = int(score / 5 * 100)
    color = "#22c55e" if pct >= 100 else "#f59e0b" if pct >= 60 else "#52525b"
    st.markdown(f"**Signal readiness — {pct}%**")
    st.progress(pct / 100)
    for c in signal_ctx.get("conditions", []):
        icon = "✅" if c["pass"] else "⬜"
        st.markdown(f"<span style='font-family:monospace;font-size:11px'>{icon}  {c['label']}</span>",
                    unsafe_allow_html=True)


def render_market_context(ctx: dict | None):
    if not ctx:
        st.caption("Market context: not available (configure live APIs)")
        return
    st.markdown("**External market intelligence**")
    items = [
        ("VIX",        f"{ctx.get('vix_level','—')} ({ctx.get('vix_regime','?')})"),
        ("Put/Call",   str(ctx.get("put_call_ratio","—"))),
        ("Session",    ctx.get("session_type","—")),
        ("Macro",      ctx.get("macro_event_name","None") if ctx.get("macro_event_near") else "None nearby"),
        ("News risk",  ctx.get("news_risk","—")),
        ("Order flow", ctx.get("order_flow_note","—") or "—"),
    ]
    for label, val in items:
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;font-family:monospace;font-size:11px;"
            f"padding:3px 0;border-bottom:1px solid #1c1c1f'>"
            f"<span style='color:#52525b'>{label}</span>"
            f"<span style='color:#a1a1aa'>{val}</span></div>",
            unsafe_allow_html=True
        )
    if ctx.get("news_summary"):
        st.caption(f"News: {ctx['news_summary'][:120]}…")


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST TABLE
# ══════════════════════════════════════════════════════════════════════════════
def render_backtest(trades: list, perf: dict):
    if not trades:
        st.info("No signals fired on this timeframe/data. Try 5m or 1m.")
        return

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Trades",       perf.get("total_trades", 0))
    c2.metric("Win rate",     f"{perf.get('win_rate',0)}%")
    c3.metric("Net P&L",      f"${perf.get('total_pnl_usd',0):,.0f}")
    c4.metric("Profit factor",str(perf.get("profit_factor","—")))

    df_t = pd.DataFrame([{
        "Dir":       t["direction"],
        "Entry":     t["entry_time"][:16],
        "Entry $":   t["entry_price"],
        "SL":        t["sl"],
        "TP":        t["tp"],
        "Exit":      (t.get("exit_time") or "")[:16],
        "Exit $":    t.get("exit_price"),
        "Reason":    t.get("exit_reason"),
        "Pts":       t.get("pnl_pts"),
        "P&L $":     t.get("pnl_usd"),
        "Conf %":    round(t.get("confidence",0)*100),
    } for t in trades])

    st.dataframe(
        df_t.style
        .map(lambda v: "color:#22c55e" if v=="LONG" else "color:#ef4444" if v=="SHORT" else "", subset=["Dir"])
        .map(lambda v: ("color:#22c55e;font-weight:bold" if isinstance(v,(int,float)) and v>0
                        else "color:#ef4444;font-weight:bold" if isinstance(v,(int,float)) and v<0 else ""),
             subset=["P&L $","Pts"])
        .format({"Entry $":"{:,.2f}","SL":"{:,.2f}","TP":"{:,.2f}",
                 "Exit $":"{:,.2f}","Pts":"{:+.2f}","P&L $":"${:+,.0f}","Conf %":"{}%"}),
        use_container_width=True, height=260,
    )

    # Equity curve
    cum=0; eq=[]; labels=[]
    for i,t in enumerate(trades):
        cum+=t["pnl_usd"]; eq.append(cum); labels.append(i+1)
    st.markdown("**Equity curve**")
    st.line_chart(pd.DataFrame({"Cumulative P&L ($)": eq}, index=labels))


# ══════════════════════════════════════════════════════════════════════════════
# CHAT INTERFACE
# ══════════════════════════════════════════════════════════════════════════════
def render_chat(signal_ctx: dict, market_ctx: dict | None):
    st.markdown("### 💬 Ask the agent")
    st.caption("The agent knows your profile, today's session, and the current market context.")

    if not AGENT_AVAILABLE:
        st.warning("Agent module not found. Ensure `agent/` folder is present.")
        return

    try:
        from agent.llm_client import provider_status, active_provider
        ps = provider_status()
    except ImportError:
        st.warning("llm_client.py not found in agent/ folder.")
        return

    if ps["status"] == "unconfigured":
        st.info(
            "**Enable the AI agent in 2 minutes (free):**\n\n"
            "1. Get a free Gemini API key at [aistudio.google.com](https://aistudio.google.com/app/apikey)\n"
            "2. In Streamlit Cloud → your app → **Settings → Secrets**, add:\n"
            "```toml\nGEMINI_API_KEY = \"AIza...\"\n```\n"
            "3. Save and reboot the app.\n\n"
            "To upgrade to Claude later, just add `ANTHROPIC_API_KEY = \"sk-ant-...\"` — "
            "the system switches automatically."
        )
        return

    st.caption(f"Using {ps['badge']} · {ps['model']}")

    agent = _get_agent()
    if not agent:
        return

    for msg in st.session_state.chat_messages[-12:]:
        css = "chat-user" if msg["role"]=="user" else "chat-agent"
        st.markdown(f"<div class='{css}'>{msg['content']}</div>", unsafe_allow_html=True)

    user_input = st.chat_input("Ask anything: 'Should I take this trade?', 'Why did the last signal fail?', 'Am I overtrading?'")
    if user_input:
        st.session_state.chat_messages.append({"role":"user","content":user_input})

        sig_obj = None
        if signal_ctx:
            try:
                from agent.trader_agent import SignalContext
                sig_obj = SignalContext(**{k:v for k,v in signal_ctx.items()
                                           if k in SignalContext.__dataclass_fields__})
            except Exception:
                pass

        ctx_obj = None
        if market_ctx:
            try:
                from agent.trader_agent import MarketContext
                ctx_obj = MarketContext(**{k:v for k,v in market_ctx.items()
                                           if k in MarketContext.__dataclass_fields__})
            except Exception:
                pass

        with st.spinner("Agent thinking…"):
            reply = agent.chat(user_input, sig_obj, ctx_obj)

        st.session_state.chat_messages.append({"role":"assistant","content":reply})
        st.rerun()

    # Quick prompts
    st.markdown("**Quick questions:**")
    quick = [
        "Should I take this trade given my recent losses?",
        "What does my win rate tell me about my SL/TP settings?",
        "Is this a bullwhip situation — should I wait?",
        "How many contracts should I trade given my account size?",
        "Explain why the signal has not fired yet on this timeframe.",
    ]
    cols = st.columns(2)
    for i, q in enumerate(quick):
        if cols[i%2].button(q, key=f"qp_{i}", use_container_width=True):
            st.session_state.chat_messages.append({"role":"user","content":q})
            agent_reply = agent.chat(q)
            st.session_state.chat_messages.append({"role":"assistant","content":agent_reply})
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    source, tf, params, auto_refresh, refresh_sec = render_sidebar()

    st.markdown("# NQ/ES — AI Trade Decision System")
    ts_str = datetime.now().strftime("%H:%M:%S")
    st.markdown(
        f"<span style='font-family:monospace;font-size:11px;color:#52525b'>"
        f"Timeframe: {tf}  ·  {source}  ·  {ts_str}</span>",
        unsafe_allow_html=True
    )

    # Load data
    nq_df, es_df = load_data(source, tf, params)
    if nq_df is None or es_df is None:
        return

    # Run strategy
    with st.spinner("Running strategy…"):
        try:
            df     = run_strategy(es_df, nq_df, params)
            sig    = build_signal_context(df, params, tf)
            trades = backtest(df, params)
            perf   = performance_summary(trades)
            st.session_state.df     = df
            st.session_state.trades = trades
            st.session_state.perf   = perf
        except ValueError as e:
            st.error(str(e)); return

    # Get market context
    market_ctx = None
    if AGENT_AVAILABLE:
        try:
            market_ctx = build_market_context()
            st.session_state.last_market_ctx = market_ctx
        except Exception:
            pass

    # Get agent decision
    decision = None
    agent    = _get_agent()
    _llm_ready = False
    if AGENT_AVAILABLE:
        try:
            from agent.llm_client import provider_status
            _llm_ready = provider_status()["status"] == "active"
        except Exception:
            pass

    if agent and _llm_ready:
        with st.spinner("Agent analysing…"):
            try:
                from agent.trader_agent import SignalContext, MarketContext
                sig_obj = SignalContext(**{k:v for k,v in sig.items()
                                           if k in SignalContext.__dataclass_fields__})
                ctx_obj = MarketContext(**(market_ctx or {})) if market_ctx else None
                dec_obj = agent.decide(sig_obj, ctx_obj)
                decision = {
                    "action":         dec_obj.action,
                    "confidence_pct": dec_obj.confidence_pct,
                    "headline":       dec_obj.headline,
                    "reason":         dec_obj.reason,
                    "agent_note":     dec_obj.agent_note,
                    "warnings":       dec_obj.warnings,
                    "trade_plan":     dec_obj.trade_plan,
                    "scm_sizing":     dec_obj.scm_sizing,
                }
                st.session_state.last_decision = decision
            except Exception as e:
                st.caption(f"Agent decision unavailable: {e}")

    # ── Layout ────────────────────────────────────────────────────────────────
    left, right = st.columns([1, 2], gap="large")

    with left:
        is_long, is_short = render_decision_card(decision, sig)
        st.markdown("---")
        render_trade_plan(sig, decision, params)
        st.markdown("---")
        render_readiness(sig, params)
        st.markdown("---")
        render_market_context(market_ctx)
        st.markdown("---")

        # Session summary
        if agent:
            mem = agent.memory
            st.markdown("**Today's session**")
            session_pnl = mem.session.session_pnl
            pnl_color = "#22c55e" if session_pnl >= 0 else "#ef4444"
            st.markdown(f"""
            <div class="kpi-card" style="margin-bottom:8px">
              <div class="kpi-l">Session P&L</div>
              <div class="kpi-v" style="color:{pnl_color}">${session_pnl:+,.0f}</div>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px">
              <div class="kpi-card"><div class="kpi-l">Trades today</div>
                <div class="kpi-v" style="color:#a1a1aa">{mem.session.trades_taken}</div></div>
              <div class="kpi-card"><div class="kpi-l">Emotional state</div>
                <div class="kpi-v" style="color:#a1a1aa;font-size:13px">{mem.session.emotional_state}</div></div>
            </div>
            """, unsafe_allow_html=True)

    with right:
        tab1, tab2, tab3 = st.tabs(["📈 Chart", "📋 Backtest", "💬 Agent chat"])

        with tab1:
            st.markdown("**NQ price · EMA20 / EMA50 · ▲▼ Signal markers**")
            chart = df[["NQ_close","ema_fast","ema_slow"]].tail(200).copy()
            chart.columns = ["NQ Close","EMA 20","EMA 50"]
            st.line_chart(chart, color=["#a1a1aa","#f59e0b","#ef4444"])

            longs  = df[df["long_signal"]].tail(20)
            shorts = df[df["short_signal"]].tail(20)
            if len(longs):
                st.success(f"▲ Last LONG signal: {str(longs.index[-1])[:16]}  @ {longs['NQ_close'].iloc[-1]:,.2f}")
            if len(shorts):
                st.error  (f"▼ Last SHORT signal: {str(shorts.index[-1])[:16]}  @ {shorts['NQ_close'].iloc[-1]:,.2f}")

            # Correlation subplot
            with st.expander("Rolling NQ/ES correlation"):
                corr_data = df[["corr_20"]].tail(200).copy()
                corr_data.columns = ["Correlation (20-bar)"]
                st.line_chart(corr_data, color=["#a78bfa"])
                st.caption(f"Threshold: {params['corr_thresh']}  ·  Current: {sig['corr_20']:.4f}")

        with tab2:
            render_backtest(trades, perf)

        with tab3:
            render_chat(sig, market_ctx)

    # Auto-refresh
    if auto_refresh and source != "📁 Upload CSV":
        st.markdown(f"<span style='color:#52525b;font-size:10px;font-family:monospace'>↻ Refreshing in {refresh_sec}s…</span>",
                    unsafe_allow_html=True)
        time.sleep(refresh_sec)
        st.rerun()


if __name__ == "__main__":
    main()
