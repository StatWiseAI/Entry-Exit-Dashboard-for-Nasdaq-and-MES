"""
agent/intelligence_layer.py
============================
Real-time external market intelligence aggregator.

WHAT THIS MODULE DOES
─────────────────────
Pulls external signals that no price chart alone can provide:
  • VIX level and regime classification
  • CBOE put/call ratio (contrarian sentiment)
  • Economic calendar (Fed, CPI, NFP — the events that break signals)
  • Web-sourced news summary (via Anthropic web search or NewsAPI)
  • Order flow context (CVD trend, DOM imbalance)
  • Session classification (pre-market / regular / overnight)

DESIGN PRINCIPLE
────────────────
Every method is independently replaceable. The stub returns None/empty
when no live API is configured. The agent degrades gracefully — it simply
notes "VIX: unknown" instead of crashing.

ADDING A REAL DATA SOURCE
─────────────────────────
Each method has a LIVE DATA section with the exact API call to uncomment.
Install the relevant package, add the API key to Streamlit secrets,
and switch the stub for the real call. Nothing else changes.

EXTERNAL APIS USED (all have free tiers)
─────────────────────────────────────────
  VIX / market data : yfinance (pip install yfinance)
  Economic calendar : Investing.com via investpy, or TradingEconomics API
  News              : NewsAPI (newsapi.org) or Anthropic web_search tool
  Order flow        : Tradovate depth API, or Sierra Chart DTC
  Sentiment         : AAII survey, CNN Fear & Greed (scraped)
"""

from __future__ import annotations

import os
import json
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import NamedTuple

logger = logging.getLogger(__name__)

# ── High-impact macro events (UTC hours when they typically release) ───────────
# Used as a fallback calendar when live API is unavailable
MACRO_SCHEDULE_UTC = {
    "Mon": [],
    "Tue": [("ISM Manufacturing", 15, 0)],
    "Wed": [("ADP Employment", 13, 15), ("FOMC Minutes", 19, 0)],
    "Thu": [("Jobless Claims", 13, 30), ("GDP", 13, 30)],
    "Fri": [("NFP", 13, 30), ("CPI", 13, 30), ("Retail Sales", 13, 30)],
}

DAYS = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]


class NewsResult(NamedTuple):
    summary:    str
    risk_level: str   # low / moderate / high
    sources:    list


class VixResult(NamedTuple):
    level:  float | None
    regime: str          # low / normal / elevated / fear / unknown


class MacroResult(NamedTuple):
    is_near:     bool
    event_name:  str
    minutes_away:int


# ══════════════════════════════════════════════════════════════════════════════
# VIX  (CBOE Volatility Index)
# ══════════════════════════════════════════════════════════════════════════════

def get_vix() -> VixResult:
    """
    Fetch the current VIX level.

    LIVE DATA (uncomment after: pip install yfinance):
    ─────────────────────────────────────────────────
    import yfinance as yf
    try:
        ticker = yf.Ticker("^VIX")
        level  = ticker.fast_info.get("lastPrice") or ticker.history(period="1d")["Close"].iloc[-1]
        return _classify_vix(float(level))
    except Exception as e:
        logger.warning("VIX fetch failed: %s", e)
        return VixResult(None, "unknown")
    """
    # ── STUB ──
    return VixResult(None, "unknown")


def _classify_vix(level: float) -> VixResult:
    if level < 13:   return VixResult(level, "low")
    elif level < 20: return VixResult(level, "normal")
    elif level < 28: return VixResult(level, "elevated")
    else:            return VixResult(level, "fear")


# ══════════════════════════════════════════════════════════════════════════════
# PUT/CALL RATIO  (CBOE equity P/C)
# ══════════════════════════════════════════════════════════════════════════════

def get_put_call_ratio() -> float | None:
    """
    CBOE equity put/call ratio.
    Interpretation: >0.8 bearish sentiment; <0.6 complacent/bullish.

    LIVE DATA (uncomment after: pip install requests):
    ──────────────────────────────────────────────────
    import requests
    try:
        url  = "https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX?interval=1d&range=1d"
        resp = requests.get(url, timeout=5, headers={"User-Agent":"Mozilla/5.0"})
        # CBOE publishes daily P/C at https://www.cboe.com/market_statistics/daily/
        # Use their CSV endpoint or scrape the daily stat page.
        pass
    except Exception as e:
        logger.warning("P/C ratio fetch failed: %s", e)
    """
    # ── STUB ──
    return None


# ══════════════════════════════════════════════════════════════════════════════
# MACRO CALENDAR
# ══════════════════════════════════════════════════════════════════════════════

def get_macro_event() -> MacroResult:
    """
    Check if a high-impact macro event is within the next 60 minutes.

    LIVE DATA — Option A: TradingEconomics API (freemium):
    ────────────────────────────────────────────────────────
    import requests
    api_key = os.environ.get("TRADING_ECONOMICS_KEY", "")
    if api_key:
        url  = f"https://api.tradingeconomics.com/calendar/country/united states?c={api_key}&d1=today"
        resp = requests.get(url, timeout=8)
        events = resp.json()
        now_utc = datetime.now(timezone.utc)
        for ev in events:
            if ev.get("importance") != "High":
                continue
            ev_dt = datetime.fromisoformat(ev["date"].replace("Z","+00:00"))
            diff  = int((ev_dt - now_utc).total_seconds() / 60)
            if 0 <= diff <= 90:
                return MacroResult(True, ev["event"], diff)
        return MacroResult(False, "", 9999)

    LIVE DATA — Option B: ForexFactory JSON (unofficial, free):
    ─────────────────────────────────────────────────────────────
    # ForexFactory has an unofficial JSON endpoint at
    # https://nfs.faireconomy.media/ff_calendar_thisweek.json
    # Parse it for USD high-impact events near the current time.

    FALLBACK — static schedule:
    ────────────────────────────
    Uses the hardcoded MACRO_SCHEDULE_UTC dict above.
    """
    now_utc  = datetime.now(timezone.utc)
    day_name = DAYS[now_utc.weekday()]
    events   = MACRO_SCHEDULE_UTC.get(day_name, [])

    for name, hour, minute in events:
        ev_dt   = now_utc.replace(hour=hour, minute=minute, second=0, microsecond=0)
        diff    = int((ev_dt - now_utc).total_seconds() / 60)
        if 0 <= diff <= 90:
            return MacroResult(True, name, diff)
        if -15 <= diff < 0:      # within 15 min after release — still volatile
            return MacroResult(True, f"{name} (just released)", abs(diff))

    return MacroResult(False, "", 9999)


# ══════════════════════════════════════════════════════════════════════════════
# NEWS INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════

def get_news_context(query: str = "NQ ES S&P500 Nasdaq futures market") -> NewsResult:
    """
    Search for breaking market news and assess risk level.

    LIVE DATA — Option A: NewsAPI (newsapi.org — free 100 req/day):
    ────────────────────────────────────────────────────────────────
    import requests
    api_key = os.environ.get("NEWSAPI_KEY", "")
    if api_key:
        url  = "https://newsapi.org/v2/everything"
        params = {
            "q":        "stock market futures NQ ES",
            "sortBy":   "publishedAt",
            "pageSize": 5,
            "apiKey":   api_key,
            "language": "en",
        }
        resp    = requests.get(url, params=params, timeout=8)
        data    = resp.json()
        titles  = [a["title"] for a in data.get("articles",[])[:5]]
        summary = " | ".join(titles[:3])
        risk    = _classify_news_risk(summary)
        return NewsResult(summary, risk, titles)

    LIVE DATA — Option B: Anthropic web_search tool:
    ─────────────────────────────────────────────────
    This is handled inside the agent via Claude tool_use.
    When the agent runs, it can call web_search directly and
    the result feeds into the MarketContext.

    LIVE DATA — Option C: Alpha Vantage News Sentiment (free tier):
    ────────────────────────────────────────────────────────────────
    import requests
    api_key = os.environ.get("ALPHAVANTAGE_KEY", "")
    if api_key:
        url    = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=SPY,QQQ&apikey={api_key}"
        resp   = requests.get(url, timeout=10).json()
        feed   = resp.get("feed", [])[:5]
        titles = [a["title"] for a in feed]
        summary = " | ".join(titles[:3])
        return NewsResult(summary, _classify_news_risk(summary), titles)
    """
    # ── STUB ──
    return NewsResult("", "low", [])


def _classify_news_risk(summary: str) -> str:
    """Simple keyword-based news risk classifier."""
    if not summary:
        return "low"
    high_keywords = [
        "crash","collapse","emergency","bank run","default","downgrade",
        "rate hike surprise","fed intervention","war","sanctions","circuit breaker",
    ]
    mod_keywords  = [
        "fed","fomc","cpi","inflation","recession","layoffs","earnings miss",
        "downside","selloff","correction","uncertainty",
    ]
    s = summary.lower()
    if any(k in s for k in high_keywords):  return "high"
    if any(k in s for k in mod_keywords):   return "moderate"
    return "low"


# ══════════════════════════════════════════════════════════════════════════════
# ORDER FLOW
# ══════════════════════════════════════════════════════════════════════════════

def get_order_flow(symbol: str = "NQ") -> str:
    """
    Cumulative Volume Delta (CVD) trend and DOM imbalance note.

    LIVE DATA — Tradovate market data API:
    ──────────────────────────────────────
    from api.topstep_connector import TopstepConnector
    import os

    username = os.environ.get("TOPSTEP_USERNAME", "")
    password = os.environ.get("TOPSTEP_PASSWORD", "")
    if username and password:
        try:
            conn  = TopstepConnector(username, password, demo=True)
            bars  = conn.get_bars(symbol, "1min", n_bars=20)
            # CVD = cumulative sum of (up_volume - down_volume)
            if "upVolume" in bars.columns and "downVolume" in bars.columns:
                bars["delta"] = bars["upVolume"] - bars["downVolume"]
                cvd = bars["delta"].cumsum().iloc[-1]
                recent_delta = bars["delta"].iloc[-3:].mean()
                if recent_delta > 500:
                    return f"CVD rising +{cvd:.0f} — buying pressure"
                elif recent_delta < -500:
                    return f"CVD falling {cvd:.0f} — selling pressure"
                else:
                    return f"CVD neutral ({cvd:.0f})"
        except Exception as e:
            logger.warning("Order flow fetch failed: %s", e)

    LIVE DATA — Sierra Chart DTC server:
    ─────────────────────────────────────
    Sierra Chart can serve real-time DOM and volume delta via DTC protocol.
    A Python DTC client is available at https://github.com/dtc-protocol/dtc-python
    """
    # ── STUB ──
    return ""


# ══════════════════════════════════════════════════════════════════════════════
# SESSION CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

def get_session_type() -> str:
    """Classify the current CME trading session."""
    now_utc = datetime.now(timezone.utc)
    h = now_utc.hour
    # Pre-market: 06:00–08:30 CT = 12:00–14:30 UTC
    if 12 <= h < 14 or (h == 14 and now_utc.minute < 30):
        return "pre-market"
    # Regular: 08:30–15:00 CT = 14:30–21:00 UTC
    elif (h == 14 and now_utc.minute >= 30) or (15 <= h < 21):
        return "regular"
    # Lunch / post-close / overnight
    elif 21 <= h < 22:
        return "closing"
    else:
        return "overnight"


# ══════════════════════════════════════════════════════════════════════════════
# MASTER BUILD FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def build_market_context() -> dict:
    """
    Assemble all external intelligence into a single context dict.
    Called once per bar by the Streamlit app.
    All failures are caught — partial context is better than no context.
    """
    vix    = VixResult(None, "unknown")
    pcr    = None
    macro  = MacroResult(False, "", 9999)
    news   = NewsResult("", "low", [])
    flow   = ""
    session= "unknown"

    try: vix    = get_vix()
    except Exception as e: logger.warning("VIX error: %s", e)

    try: pcr    = get_put_call_ratio()
    except Exception as e: logger.warning("P/C error: %s", e)

    try: macro  = get_macro_event()
    except Exception as e: logger.warning("Macro error: %s", e)

    try: news   = get_news_context()
    except Exception as e: logger.warning("News error: %s", e)

    try: flow   = get_order_flow()
    except Exception as e: logger.warning("Flow error: %s", e)

    try: session = get_session_type()
    except Exception as e: logger.warning("Session error: %s", e)

    return {
        "timestamp":          datetime.now(timezone.utc).isoformat(),
        "vix_level":          vix.level,
        "vix_regime":         vix.regime,
        "put_call_ratio":     pcr,
        "macro_event_near":   macro.is_near,
        "macro_event_name":   macro.event_name,
        "macro_minutes_away": macro.minutes_away,
        "news_summary":       news.summary,
        "news_risk":          news.risk_level,
        "news_sources":       news.sources,
        "order_flow_note":    flow,
        "session_type":       session,
    }


# ── Quick self-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pprint
    print("Building market context (stubs active — shows structure)…\n")
    ctx = build_market_context()
    pprint.pprint(ctx)
    print("\nAll stubs ran successfully.")
    print("To activate live data: uncomment the LIVE DATA sections above")
    print("and set the relevant environment variables.")
