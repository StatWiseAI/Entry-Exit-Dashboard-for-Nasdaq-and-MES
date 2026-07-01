"""
agent/trader_agent.py
=====================
Conversational AI decision-support agent for NQ/ES futures trading.

THEORETICAL FRAMEWORK
─────────────────────
This agent integrates three streams previously unconnected in trading analytics:

1. SUPPLY CHAIN MANAGEMENT DECISION MODELS
   • Economic Order Quantity (EOQ) → optimal position size given volatility "cost"
   • Newsvendor Model → asymmetric SL/TP optimisation under demand uncertainty
   • Safety Stock → minimum account buffer before entering a trade
   • Bullwhip Effect → why amplified reactions to news destroy edge; agent flags this

2. CONVERSATIONAL AI WITH PERSISTENT MEMORY
   • Trader profile: risk tolerance, session goals, known biases, time-of-day patterns
   • Session memory: current P&L, emotional state markers, trades taken today
   • Outcome feedback: every closed trade updates the memory, improving future advice
   • Socratic mode: agent challenges the trader's intent before confirming a signal

3. REAL-TIME EXTERNAL MARKET INTELLIGENCE
   • Macro calendar awareness (Fed, CPI, NFP windows)
   • Sentiment proxies (VIX level, put/call ratio, AAII fear index)
   • News context (web search for breaking events before any signal)
   • Order flow context (CVD trend, DOM imbalance if available)

HOW IT WORKS
────────────
  strategy_engine  →  signal dict
  intelligence_layer → context dict
  trader_memory    → profile dict
         ↓
  TraderAgent.decide()
         ↓
  DecisionOutput (ENTER / WAIT / EXIT + plain-language reason + trade plan)
         ↓
  Dashboard renders the big green/red button
  Agent answers follow-up questions conversationally
  Trade outcome is journalled back into memory
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.llm_client import chat_complete, active_provider, active_model

NQ_POINT_VALUE = 20   # $20 per NQ point (E-mini)
MEMORY_FILE    = Path(__file__).parent / "trader_memory.json"


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TraderProfile:
    """Persistent trader identity — updated by the agent over time."""
    name:                str   = "Trader"
    risk_per_trade_pct:  float = 1.0       # % of account at risk per trade
    account_size_usd:    float = 50_000.0  # funded account size
    preferred_tfs:       list  = field(default_factory=lambda: ["5min", "30min"])
    known_biases:        list  = field(default_factory=list)  # e.g. ["FOMO on breakouts", "cuts winners short"]
    trading_hours:       str   = "08:00-15:00 CT"
    max_daily_loss_pct:  float = 3.0       # hard stop for the day
    max_daily_trades:    int   = 5
    style_notes:         str   = ""        # e.g. "trend-follower, avoids news windows"


@dataclass
class SessionState:
    """Intraday state — resets each morning."""
    date:            str   = ""
    trades_taken:    int   = 0
    session_pnl:     float = 0.0
    daily_loss_hit:  bool  = False
    emotional_state: str   = "neutral"     # neutral / elevated / fatigued / overconfident
    last_signal_ts:  str   = ""
    consecutive_losses: int = 0


@dataclass
class TradeRecord:
    """Single completed trade, persisted for learning."""
    direction:    str
    entry_price:  float
    exit_price:   float
    sl:           float
    tp:           float
    entry_time:   str
    exit_time:    str
    exit_reason:  str          # TP / SL / TIME / MANUAL
    pnl_pts:      float
    pnl_usd:      float
    tf:           str
    signal_score: int
    context_note: str = ""    # agent's note on why this trade was taken


@dataclass
class MarketContext:
    """External intelligence snapshot at decision time."""
    timestamp:         str   = ""
    vix_level:         float | None = None
    vix_regime:        str   = "unknown"   # low / normal / elevated / fear
    put_call_ratio:    float | None = None
    macro_event_near:  bool  = False
    macro_event_name:  str   = ""
    macro_minutes_away:int   = 9999
    news_summary:      str   = ""          # 1-2 sentence summary from web search
    news_risk:         str   = "low"       # low / moderate / high
    order_flow_note:   str   = ""          # CVD trend, DOM imbalance
    session_type:      str   = "regular"   # pre-market / regular / overnight


@dataclass
class SignalContext:
    """Output from the strategy engine, ready for the agent."""
    signal:       str    = "NONE"          # LONG / SHORT / NONE
    score:        int    = 0               # 0-5 conditions met
    approaching:  bool   = False           # 4/5 met, watch next bar
    confidence:   float  = 0.0            # 0.0-1.0, from SCM scorer
    tf:           str    = "30min"
    timestamp:    str    = ""
    entry_price:  float  = 0.0
    sl:           float  = 0.0
    tp:           float  = 0.0
    sl_pct:       float  = 0.005
    tp_pct:       float  = 0.010
    hold_bars:    int    = 5
    rr_ratio:     float  = 2.0
    risk_usd:     float  = 0.0
    reward_usd:   float  = 0.0
    pullback_pct: float  = 0.0
    corr_20:      float  = 0.0
    trend:        int    = 0
    conditions:   list   = field(default_factory=list)


@dataclass
class DecisionOutput:
    """Final output from the agent — drives the dashboard."""
    action:         str   = "WAIT"          # ENTER_LONG / ENTER_SHORT / WAIT / EXIT / REDUCE
    confidence_pct: int   = 0               # 0-100
    headline:       str   = ""              # one bold sentence for the button
    reason:         str   = ""              # plain-language 2-3 sentence explanation
    trade_plan:     dict  = field(default_factory=dict)
    warnings:       list  = field(default_factory=list)  # risk flags
    agent_note:     str   = ""              # personalised comment from the agent
    scm_sizing:     dict  = field(default_factory=dict)  # SCM-derived position advice
    timestamp:      str   = ""


# ══════════════════════════════════════════════════════════════════════════════
# TRADER MEMORY  (persistent JSON store)
# ══════════════════════════════════════════════════════════════════════════════

class TraderMemory:
    """
    Persistent memory store for the trader.
    Loaded on startup, updated after every trade and session.
    In production, replace with a database (Supabase, SQLite, etc.)
    """

    def __init__(self, path: Path = MEMORY_FILE):
        self.path    = path
        self.profile = TraderProfile()
        self.session = SessionState()
        self.trades:  list[TradeRecord] = []
        self.chat_history: list[dict]   = []
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                p = data.get("profile", {})
                self.profile = TraderProfile(**{
                    k: v for k, v in p.items()
                    if k in TraderProfile.__dataclass_fields__
                })
                s = data.get("session", {})
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                if s.get("date") == today:
                    self.session = SessionState(**{
                        k: v for k, v in s.items()
                        if k in SessionState.__dataclass_fields__
                    })
                else:
                    self.session = SessionState(date=today)

                self.trades = [
                    TradeRecord(**{k: v for k, v in t.items()
                                   if k in TradeRecord.__dataclass_fields__})
                    for t in data.get("trades", [])
                ]
                self.chat_history = data.get("chat_history", [])[-40:]
            except Exception:
                pass  # start fresh on corrupt file

    def save(self):
        data = {
            "profile":      asdict(self.profile),
            "session":      asdict(self.session),
            "trades":       [asdict(t) for t in self.trades[-200:]],
            "chat_history": self.chat_history[-40:],
        }
        self.path.write_text(json.dumps(data, indent=2))

    def record_trade(self, trade: TradeRecord):
        self.trades.append(trade)
        self.session.trades_taken    += 1
        self.session.session_pnl     += trade.pnl_usd
        if trade.pnl_usd < 0:
            self.session.consecutive_losses += 1
        else:
            self.session.consecutive_losses  = 0
        daily_loss_pct = abs(min(self.session.session_pnl, 0)) / self.profile.account_size_usd * 100
        if daily_loss_pct >= self.profile.max_daily_loss_pct:
            self.session.daily_loss_hit = True
        self.save()

    def recent_win_rate(self, n: int = 20) -> float:
        recent = self.trades[-n:]
        if not recent:
            return 0.5
        wins = sum(1 for t in recent if t.pnl_usd > 0)
        return wins / len(recent)

    def performance_summary(self) -> dict:
        if not self.trades:
            return {"trades": 0, "win_rate": None, "avg_pnl": None, "total_pnl": 0}
        wins    = [t for t in self.trades if t.pnl_usd > 0]
        losses  = [t for t in self.trades if t.pnl_usd <= 0]
        total   = sum(t.pnl_usd for t in self.trades)
        g       = sum(t.pnl_usd for t in wins)
        l       = abs(sum(t.pnl_usd for t in losses))
        return {
            "trades":         len(self.trades),
            "win_rate":       round(len(wins) / len(self.trades) * 100, 1),
            "avg_win":        round(g / len(wins), 0) if wins else 0,
            "avg_loss":       round(l / len(losses), 0) if losses else 0,
            "profit_factor":  round(g / l, 2) if l > 0 else 999,
            "total_pnl":      round(total, 0),
            "consecutive_losses_today": self.session.consecutive_losses,
        }


# ══════════════════════════════════════════════════════════════════════════════
# SCM DECISION MODELS
# ══════════════════════════════════════════════════════════════════════════════

class SCMDecisionModels:
    """
    Supply Chain Management models adapted for capital allocation.

    EOQ → Position Sizing
    ─────────────────────
    In inventory theory, EOQ balances ordering cost vs holding cost.
    In trading: ordering cost = transaction cost + opportunity cost of being flat.
    Holding cost = risk per unit time × position size.
    Optimal position = sqrt(2 × edge × capital / risk_per_unit²)
    Simplified: position scales with edge strength and inversely with volatility.

    Newsvendor Model → SL/TP Optimisation
    ──────────────────────────────────────
    The newsvendor chooses a stock quantity to maximise expected profit under
    uncertain demand. Here: choose SL/TP to maximise expected P&L given the
    signal's historical win rate distribution.
    Optimal critical ratio = profit / (profit + loss) = tp / (tp + sl)
    If win_rate > critical_ratio → widen TP or tighten SL.
    If win_rate < critical_ratio → tighten TP or widen SL.

    Safety Stock → Account Buffer
    ─────────────────────────────
    Safety stock prevents stockouts under demand variability.
    Here: minimum account buffer before entering a trade = 3× max daily loss.
    If account drawdown > safety threshold → no new trades.

    Bullwhip Effect → News Amplification Warning
    ─────────────────────────────────────────────
    In supply chains, small demand changes cause wild upstream swings.
    In trading: breaking news causes overreaction. Agent flags if VIX spikes
    or news risk is high — the true "demand signal" is distorted.
    """

    @staticmethod
    def eoq_position_size(
        profile:    TraderProfile,
        signal:     SignalContext,
        market_ctx: MarketContext,
    ) -> dict:
        """
        EOQ-adapted position sizing.
        Returns recommended contracts and risk-adjusted sizing rationale.
        """
        account  = profile.account_size_usd
        risk_pct = profile.risk_per_trade_pct / 100

        # Base risk in USD
        base_risk_usd = account * risk_pct

        # Volatility adjustment: higher VIX → reduce size
        vix = market_ctx.vix_level or 20.0
        vol_adj = max(0.4, min(1.2, 20.0 / vix))  # VIX 20 = 1.0x, VIX 30 = 0.67x, VIX 15 = 1.2x

        # Confidence adjustment from signal score (0.6–1.0 range)
        conf_adj = 0.6 + (signal.score / 5) * 0.4

        # News risk adjustment
        news_adj = {"low": 1.0, "moderate": 0.7, "high": 0.4}.get(market_ctx.news_risk, 1.0)

        # Macro event near: halve size in the 30 minutes before a major release
        macro_adj = 0.5 if market_ctx.macro_event_near and market_ctx.macro_minutes_away < 30 else 1.0

        # Consecutive loss penalty (Gambler's ruin protection)
        loss_streak = 1.0
        if signal.score > 0:  # avoid division errors on dummy calls
            loss_streak = max(0.5, 1.0 - 0.15 * min(3, getattr(profile, '_consecutive_losses', 0)))

        adjusted_risk = base_risk_usd * vol_adj * conf_adj * news_adj * macro_adj * loss_streak

        # Contract size: risk_usd / (sl_pct × entry × point_value_per_point)
        sl_usd_per_contract = signal.sl_pct * signal.entry_price * NQ_POINT_VALUE if signal.entry_price > 0 else 1
        contracts = max(1, round(adjusted_risk / sl_usd_per_contract)) if sl_usd_per_contract > 0 else 1

        return {
            "recommended_contracts": contracts,
            "base_risk_usd":         round(base_risk_usd, 0),
            "adjusted_risk_usd":     round(adjusted_risk, 0),
            "vol_adjustment":        round(vol_adj, 2),
            "confidence_adjustment": round(conf_adj, 2),
            "news_adjustment":       round(news_adj, 2),
            "macro_adjustment":      round(macro_adj, 2),
            "sl_usd_per_contract":   round(sl_usd_per_contract, 0),
            "sizing_rationale":      (
                f"Base risk ${base_risk_usd:.0f} × "
                f"vol_adj {vol_adj:.2f} × "
                f"conf {conf_adj:.2f} × "
                f"news {news_adj:.2f} × "
                f"macro {macro_adj:.2f} = "
                f"${adjusted_risk:.0f} → {contracts} contract(s)"
            ),
        }

    @staticmethod
    def newsvendor_sl_tp(
        base_sl_pct:   float,
        base_tp_pct:   float,
        historical_wr: float,
        confidence:    float,
    ) -> dict:
        """
        Newsvendor-adapted SL/TP optimisation.
        Adjusts the base SL/TP ratios based on historical win rate.
        """
        # Critical ratio: the win rate at which the base R:R is exactly fair
        rr = base_tp_pct / base_sl_pct if base_sl_pct > 0 else 2.0
        critical_ratio = rr / (1 + rr)  # fair win rate for this R:R

        verdict = "balanced"
        adj_sl  = base_sl_pct
        adj_tp  = base_tp_pct

        if historical_wr > 0 and abs(historical_wr - critical_ratio) > 0.05:
            if historical_wr > critical_ratio:
                # We win more often than critical ratio → we can widen TP (take more per winner)
                adj_tp  = base_tp_pct * min(1.3, 1 + (historical_wr - critical_ratio))
                verdict = "widen_tp"
            else:
                # We win less often → tighten TP to improve probability
                adj_sl  = base_sl_pct * max(0.7, 1 - (critical_ratio - historical_wr))
                verdict = "tighten_sl"

        return {
            "adj_sl_pct":      round(adj_sl, 4),
            "adj_tp_pct":      round(adj_tp, 4),
            "critical_ratio":  round(critical_ratio, 3),
            "historical_wr":   round(historical_wr, 3),
            "verdict":         verdict,
            "note": (
                f"Historical win rate {historical_wr:.0%} vs critical ratio {critical_ratio:.0%} "
                f"→ {verdict.replace('_',' ')}"
            ),
        }

    @staticmethod
    def safety_stock_check(
        profile: TraderProfile,
        session: SessionState,
    ) -> dict:
        """
        Safety stock check: can the trader enter a new position today?
        Returns can_trade + reasons why not.
        """
        reasons  = []
        can_trade = True

        if session.daily_loss_hit:
            can_trade = False
            reasons.append(f"Daily loss limit reached ({profile.max_daily_loss_pct}% of account)")

        if session.trades_taken >= profile.max_daily_trades:
            can_trade = False
            reasons.append(f"Max daily trades reached ({profile.max_daily_trades})")

        if session.consecutive_losses >= 3:
            can_trade = False
            reasons.append("3 consecutive losses today — mandatory pause (protect capital)")

        if session.emotional_state in ("elevated", "fatigued", "overconfident"):
            reasons.append(f"Emotional state flagged as '{session.emotional_state}' — size down or skip")

        pnl_pct = session.session_pnl / profile.account_size_usd * 100
        if pnl_pct <= -(profile.max_daily_loss_pct * 0.7):
            reasons.append(f"Session P&L at {pnl_pct:.1f}% — approaching daily limit, reduce size")

        return {
            "can_trade": can_trade,
            "reasons":   reasons,
            "session_pnl_pct": round(pnl_pct, 2),
            "trades_today": session.trades_taken,
        }

    @staticmethod
    def bullwhip_warning(market_ctx: MarketContext) -> list[str]:
        """
        Detect bullwhip-effect conditions: amplified reactions to news.
        Returns a list of warnings (empty = no concern).
        """
        warnings = []
        vix = market_ctx.vix_level or 20.0

        if vix > 30:
            warnings.append(f"VIX {vix:.1f} — extreme fear regime. Signal edges shrink under panic conditions.")
        elif vix > 22:
            warnings.append(f"VIX {vix:.1f} — elevated. Wider spreads expected; tighten risk sizing.")

        if market_ctx.macro_event_near and market_ctx.macro_minutes_away < 60:
            warnings.append(
                f"{market_ctx.macro_event_name} in {market_ctx.macro_minutes_away}min. "
                f"Orders fill erratically around releases — consider waiting."
            )

        if market_ctx.news_risk == "high":
            warnings.append(
                f"Breaking news detected: '{market_ctx.news_summary[:80]}…' "
                "This is a classic bullwhip scenario — wait for the reaction to settle."
            )

        return warnings


# ══════════════════════════════════════════════════════════════════════════════
# INTELLIGENCE LAYER
# ══════════════════════════════════════════════════════════════════════════════

class IntelligenceLayer:
    """
    Gathers real-time external market context.
    Each method is independently replaceable with a live API call.
    """

    @staticmethod
    def get_vix_context() -> tuple[float | None, str]:
        """
        Fetch VIX level and classify the regime.
        Replace the stub with a real API call (Yahoo Finance, CBOE, etc.)
        """
        # ── STUB: replace with live call ──────────────────────────────────────
        # import yfinance as yf
        # vix_level = yf.Ticker("^VIX").fast_info["lastPrice"]
        vix_level = None   # None = unavailable

        if vix_level is None:
            return None, "unknown"
        if vix_level < 15:
            return vix_level, "low"
        elif vix_level < 22:
            return vix_level, "normal"
        elif vix_level < 30:
            return vix_level, "elevated"
        else:
            return vix_level, "fear"

    @staticmethod
    def get_put_call_ratio() -> float | None:
        """CBOE equity put/call ratio. >1.0 = bearish sentiment."""
        # ── STUB ──
        return None

    @staticmethod
    def get_macro_calendar() -> tuple[bool, str, int]:
        """
        Check if a high-impact macro event is coming within 60 minutes.
        Returns (is_near, event_name, minutes_away).
        Sources: ForexFactory, Investing.com, FRED calendar API.
        """
        # ── STUB ──
        return False, "", 9999

    @staticmethod
    def search_market_news(query: str = "NQ ES futures market news") -> tuple[str, str]:
        """
        Web search for breaking market intelligence.
        Returns (summary, risk_level).

        In production: use Anthropic web_search tool or a news API.
        This is called by the agent via Claude's tool use.
        """
        # ── STUB: the agent calls this via Claude tool_use ──
        return "", "low"

    @staticmethod
    def get_order_flow_note(symbol: str = "NQ") -> str:
        """
        Cumulative volume delta (CVD) trend and DOM imbalance.
        Sources: Sierra Chart, Bookmap, Tradovate depth API.
        """
        # ── STUB ──
        return ""

    @classmethod
    def build_context(cls) -> MarketContext:
        """Assemble a full MarketContext snapshot."""
        vix, vix_regime = cls.get_vix_context()
        pcr              = cls.get_put_call_ratio()
        macro, name, eta = cls.get_macro_calendar()
        news, news_risk  = cls.search_market_news()
        flow             = cls.get_order_flow_note()
        now              = datetime.now(timezone.utc)
        hour_utc = now.hour
        session_type = (
            "pre-market" if 12 <= hour_utc < 14 else
            "regular"    if 14 <= hour_utc < 21 else
            "overnight"
        )
        return MarketContext(
            timestamp         = now.isoformat(),
            vix_level         = vix,
            vix_regime        = vix_regime,
            put_call_ratio    = pcr,
            macro_event_near  = macro,
            macro_event_name  = name,
            macro_minutes_away= eta,
            news_summary      = news,
            news_risk         = news_risk,
            order_flow_note   = flow,
            session_type      = session_type,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TRADER AGENT
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a professional trading decision-support agent for an NQ/ES futures trader.

Your role is to bridge the gap between what the market is doing and what THIS specific trader should do RIGHT NOW, given who they are.

THEORETICAL FRAMEWORK YOU APPLY
────────────────────────────────
1. SCM Decision Models: You think about position sizing like EOQ (balance risk cost vs edge), SL/TP optimisation like the Newsvendor model (calibrate asymmetry to win rate), and daily risk limits like Safety Stock (protect the account buffer). Flag Bullwhip Effect when news is amplifying market reactions beyond their true signal.

2. Conversational memory: You remember this trader's profile, biases, session P&L, emotional state, and recent trade history. Every recommendation is personalised. You are not a generic signal bot.

3. External intelligence: You factor in VIX regime, macro calendar, breaking news, and order flow context before confirming any signal. You never say ENTER NOW into a known risk event.

COMMUNICATION STYLE
───────────────────
- ONE clear action: ENTER LONG / ENTER SHORT / WAIT / EXIT / REDUCE SIZE
- One plain-language sentence explaining WHY (not a list of indicators)
- One personalised note that references what you know about this trader
- Warnings are concise and actionable, never preachy
- In Socratic mode (when the trader questions a signal): challenge their thinking respectfully, don't just agree
- If the trader says they "feel like they should enter" with no signal: flag emotional bias by name

WHAT YOU NEVER DO
─────────────────
- Never give a generic answer that ignores the trader's memory
- Never recommend entering during high-impact news windows
- Never approve a trade when the safety stock check fails (daily loss, trade count, loss streak)
- Never claim certainty — always express decisions as probabilistic recommendations
- Never reproduce indicator lists — translate them into a single decision sentence

You have access to the full signal context, market intelligence, SCM analysis, and trader profile in the user message.
"""


class TraderAgent:
    """
    The conversational AI decision-support agent.
    Combines signal, SCM analysis, market context, and trader memory
    into a personalised, explainable decision.
    """

    def __init__(self, memory: TraderMemory | None = None):
        self.memory  = memory or TraderMemory()
        self.scm     = SCMDecisionModels()
        self.intel   = IntelligenceLayer()

    # ── Main decision entry point ─────────────────────────────────────────────

    def decide(
        self,
        signal:      SignalContext,
        market_ctx:  MarketContext | None = None,
    ) -> DecisionOutput:
        """
        Produce a full DecisionOutput from signal + context + memory.
        This is the function called by the Streamlit dashboard on every bar.
        """
        if market_ctx is None:
            market_ctx = self.intel.build_context()

        profile = self.memory.profile
        session = self.memory.session
        perf    = self.memory.performance_summary()

        # ── SCM checks ────────────────────────────────────────────────────────
        safety   = self.scm.safety_stock_check(profile, session)
        sizing   = self.scm.eoq_position_size(profile, signal, market_ctx)
        nv_opt   = self.scm.newsvendor_sl_tp(
            signal.sl_pct, signal.tp_pct,
            perf.get("win_rate", 50) / 100 if perf.get("win_rate") else 0.5,
            signal.confidence,
        )
        bullwhip = self.scm.bullwhip_warning(market_ctx)

        # ── Override signal if safety check fails ──────────────────────────────
        effective_signal = signal.signal
        warnings         = list(bullwhip)

        if not safety["can_trade"] and signal.signal != "NONE":
            effective_signal = "NONE"
            warnings = safety["reasons"] + warnings

        # ── Build prompt ──────────────────────────────────────────────────────
        prompt = self._build_decision_prompt(
            signal, effective_signal, market_ctx,
            safety, sizing, nv_opt, bullwhip, perf,
        )

        # ── Call LLM (Gemini free tier by default; Claude when key is set) ────
        try:
            messages   = self.memory.chat_history[-8:] + [{"role": "user", "content": prompt}]
            agent_text = chat_complete(SYSTEM_PROMPT, messages, max_tokens=600)
        except Exception as e:
            agent_text = f"Agent unavailable: {e}"

        # ── Parse agent response into structured output ───────────────────────
        decision = self._parse_agent_response(
            agent_text, effective_signal, signal, sizing, nv_opt, warnings,
        )

        # ── Update chat history ───────────────────────────────────────────────
        self.memory.chat_history.append({"role": "user",      "content": prompt})
        self.memory.chat_history.append({"role": "assistant", "content": agent_text})
        self.memory.save()

        return decision

    # ── Conversational Q&A ────────────────────────────────────────────────────

    def chat(
        self,
        user_message: str,
        signal:       SignalContext | None = None,
        market_ctx:   MarketContext | None = None,
    ) -> str:
        """
        Answer a free-form question from the trader.
        Maintains conversation history for context continuity.
        """
        ctx_block = ""
        if signal:
            ctx_block = (
                f"\n\nCurrent signal context: {signal.signal} on {signal.tf}, "
                f"score {signal.score}/5, entry {signal.entry_price:.2f}, "
                f"SL {signal.sl:.2f}, TP {signal.tp:.2f}."
            )
        if market_ctx and market_ctx.news_summary:
            ctx_block += f"\nNews context: {market_ctx.news_summary}"

        perf    = self.memory.performance_summary()
        profile = self.memory.profile
        memory_block = (
            f"\nTrader profile: {profile.name}, account ${profile.account_size_usd:,.0f}, "
            f"risk/trade {profile.risk_per_trade_pct}%, style: {profile.style_notes or 'general'}. "
            f"Today: {self.memory.session.trades_taken} trades, "
            f"P&L ${self.memory.session.session_pnl:+,.0f}, "
            f"state: {self.memory.session.emotional_state}. "
            f"All-time: {perf.get('trades',0)} trades, "
            f"win rate {perf.get('win_rate','?')}%, "
            f"total P&L ${perf.get('total_pnl',0):+,.0f}."
        )

        full_message = user_message + ctx_block + memory_block
        messages = self.memory.chat_history[-10:] + [{"role": "user", "content": full_message}]

        try:
            messages = self.memory.chat_history[-10:] + [{"role": "user", "content": full_message}]
            reply    = chat_complete(SYSTEM_PROMPT, messages, max_tokens=400)
        except Exception as e:
            reply = f"Agent unavailable: {e}"

        self.memory.chat_history.append({"role": "user",      "content": user_message})
        self.memory.chat_history.append({"role": "assistant", "content": reply})
        self.memory.save()
        return reply

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_decision_prompt(
        self, signal, effective, market_ctx,
        safety, sizing, nv_opt, bullwhip, perf,
    ) -> str:
        profile = self.memory.profile
        session = self.memory.session

        return f"""
DECISION REQUEST — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}

SIGNAL
  Raw signal      : {signal.signal} (effective after safety check: {effective})
  Timeframe       : {signal.tf}
  Score           : {signal.score}/5 conditions met
  Confidence      : {signal.confidence:.0%}
  Entry           : {signal.entry_price:,.2f}
  Stop-Loss       : {signal.sl:,.2f}  ({signal.sl_pct:.2%})
  Take-Profit     : {signal.tp:,.2f}  ({signal.tp_pct:.2%})
  R:R             : 1:{signal.rr_ratio:.1f}
  Pullback        : {signal.pullback_pct:.3%}
  NQ/ES Corr      : {signal.corr_20:.3f}

MARKET CONTEXT
  VIX             : {market_ctx.vix_level or 'unknown'} ({market_ctx.vix_regime})
  Put/Call ratio  : {market_ctx.put_call_ratio or 'unknown'}
  Macro event     : {'YES — ' + market_ctx.macro_event_name + ' in ' + str(market_ctx.macro_minutes_away) + 'min' if market_ctx.macro_event_near else 'None nearby'}
  News risk       : {market_ctx.news_risk}
  News summary    : {market_ctx.news_summary or 'None available'}
  Order flow      : {market_ctx.order_flow_note or 'Not available'}
  Session type    : {market_ctx.session_type}

SCM ANALYSIS
  Safety stock    : {'PASS — can trade' if safety['can_trade'] else 'FAIL — ' + '; '.join(safety['reasons'])}
  EOQ sizing      : {sizing['recommended_contracts']} contract(s) — {sizing['sizing_rationale']}
  Newsvendor opt  : {nv_opt['note']}
  Bullwhip flags  : {'; '.join(bullwhip) if bullwhip else 'None'}

TRADER MEMORY
  Name            : {profile.name}
  Account         : ${profile.account_size_usd:,.0f}
  Risk/trade      : {profile.risk_per_trade_pct}%
  Style           : {profile.style_notes or 'Not set'}
  Known biases    : {', '.join(profile.known_biases) or 'None recorded'}
  Session P&L     : ${session.session_pnl:+,.0f} ({session.trades_taken} trades today)
  Emotional state : {session.emotional_state}
  Consec. losses  : {session.consecutive_losses}

HISTORICAL PERFORMANCE
  All-time trades : {perf.get('trades', 0)}
  Win rate        : {perf.get('win_rate', 'N/A')}%
  Profit factor   : {perf.get('profit_factor', 'N/A')}
  Total P&L       : ${perf.get('total_pnl', 0):+,.0f}

Produce a decision. Be direct, personalised, and honest about uncertainty.
Format your response as:
ACTION: [ENTER_LONG | ENTER_SHORT | WAIT | EXIT | REDUCE_SIZE]
CONFIDENCE: [0-100]%
HEADLINE: [one bold sentence — max 12 words]
REASON: [2-3 sentences plain language, no indicator lists]
NOTE: [one personalised sentence referencing this trader's memory]
"""

    def _parse_agent_response(
        self, text, effective_signal, signal, sizing, nv_opt, warnings,
    ) -> DecisionOutput:
        lines = {
            k.strip(): v.strip()
            for line in text.splitlines()
            if ":" in line
            for k, v in [line.split(":", 1)]
        }

        action     = lines.get("ACTION", "WAIT").replace("_", " ")
        conf_str   = lines.get("CONFIDENCE", "0").replace("%", "").strip()
        try:    confidence = int(float(conf_str))
        except: confidence = 0

        headline   = lines.get("HEADLINE", "Stand by — no trade now")
        reason     = lines.get("REASON",   "Conditions not fully aligned.")
        note       = lines.get("NOTE",     "")

        ep   = signal.entry_price
        sl   = signal.sl
        tp   = signal.tp
        rpts = abs(ep - sl)
        wpts = abs(ep - tp)

        trade_plan = {}
        if effective_signal in ("LONG", "SHORT") and ep > 0:
            trade_plan = {
                "direction":       effective_signal,
                "entry":           ep,
                "sl":              sl,
                "tp":              tp,
                "risk_pts":        round(rpts, 2),
                "reward_pts":      round(wpts, 2),
                "risk_usd":        round(rpts * NQ_POINT_VALUE, 0),
                "reward_usd":      round(wpts * NQ_POINT_VALUE, 0),
                "rr_ratio":        round(wpts / rpts, 2) if rpts > 0 else 0,
                "contracts":       sizing["recommended_contracts"],
                "total_risk_usd":  round(rpts * NQ_POINT_VALUE * sizing["recommended_contracts"], 0),
                "total_reward_usd":round(wpts * NQ_POINT_VALUE * sizing["recommended_contracts"], 0),
                "max_hold_bars":   signal.hold_bars,
                "newsvendor_note": nv_opt["note"],
            }

        return DecisionOutput(
            action         = action,
            confidence_pct = confidence,
            headline       = headline,
            reason         = reason,
            trade_plan     = trade_plan,
            warnings       = warnings,
            agent_note     = note,
            scm_sizing     = sizing,
            timestamp      = datetime.now(timezone.utc).isoformat(),
        )

    # ── Profile management ────────────────────────────────────────────────────

    def update_profile(self, **kwargs):
        """Update trader profile fields and save."""
        for k, v in kwargs.items():
            if hasattr(self.memory.profile, k):
                setattr(self.memory.profile, k, v)
        self.memory.save()

    def set_emotional_state(self, state: str):
        """States: neutral / elevated / fatigued / overconfident"""
        self.memory.session.emotional_state = state
        self.memory.save()

    def record_completed_trade(self, trade: TradeRecord):
        """Call this when a trade closes to update memory and performance."""
        self.memory.record_trade(trade)

    def reset_session(self):
        """Call at the start of each trading day."""
        self.memory.session = SessionState(date=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
        self.memory.save()
