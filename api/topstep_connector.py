"""
api/topstep_connector.py
========================
Topstep uses Tradovate as its execution and data platform.
This file is a ready-to-complete scaffold. All the structure is here —
you only need to fill in your credentials and test it.

HOW TOPSTEP / TRADOVATE WORKS
──────────────────────────────
1. Topstep gives you a funded account on Tradovate.
2. Tradovate has a REST + WebSocket API.
3. You authenticate once → get a token → use it for all data requests.
4. Market data (bars/candles) comes from the /md/getchart endpoint.
5. Order placement uses /order/placeOrder (NOT wired here — manual trading only for now).

OFFICIAL DOCS
──────────────
  Tradovate API:    https://api.tradovate.com
  Tradovate GitHub: https://github.com/tradovate
  Topstep help:     https://help.topstep.com

SETUP STEPS
───────────
Step 1: Get your Tradovate credentials
  - Log into Topstep → go to the platform → Tradovate is your broker
  - Your Tradovate username = your Topstep login email
  - Your Tradovate password = your Topstep password
  - Enable API access at: https://trader.tradovate.com → Settings → API

Step 2: Install the dependency
  pip install requests websocket-client

Step 3: Add credentials to Streamlit secrets
  In share.streamlit.app → your app → Settings → Secrets, add:
    TOPSTEP_USERNAME = "your@email.com"
    TOPSTEP_PASSWORD = "yourpassword"
    TOPSTEP_DEMO = true          # true = demo/sim account, false = live

Step 4: In streamlit_app.py, change data source to "Topstep (live)"
  The load_from_topstep() function will call this connector automatically.

IMPORTANT: Never commit your password to GitHub.
Always use Streamlit secrets or environment variables.
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd

logger = logging.getLogger(__name__)

# ── Tradovate endpoints ───────────────────────────────────────────────────────
DEMO_BASE  = "https://demo.tradovateapi.com/v1"
LIVE_BASE  = "https://live.tradovateapi.com/v1"
MD_BASE    = "https://md.tradovateapi.com/v1"   # market data endpoint


# ── Timeframe mapping (our labels → Tradovate unitType + nUnits) ──────────────
TF_MAP = {
    "1min":  {"unitType": "MinuteBar", "nUnits": 1},
    "5min":  {"unitType": "MinuteBar", "nUnits": 5},
    "30min": {"unitType": "MinuteBar", "nUnits": 30},
    "1hour": {"unitType": "HourBar",   "nUnits": 1},
    "1day":  {"unitType": "DailyBar",  "nUnits": 1},
}

# ── Contract names (continuous front-month) ───────────────────────────────────
SYMBOL_MAP = {
    "NQ": "NQU4",   # ← Update to current front-month contract (e.g. NQZ4 for Dec)
    "ES": "ESU4",   # ← Update to current front-month contract
}


class TopstepConnector:
    """
    Authenticate with Tradovate and fetch OHLCV bar data.

    Usage
    -----
    connector = TopstepConnector(username="you@email.com", password="pass")
    nq_df = connector.get_bars("NQ", "30min", n_bars=200)
    es_df = connector.get_bars("ES", "30min", n_bars=200)
    """

    def __init__(self, username: str, password: str, demo: bool = True):
        self.username  = username
        self.password  = password
        self.demo      = demo
        self.base_url  = DEMO_BASE if demo else LIVE_BASE
        self.token     = None
        self.token_exp = None
        self._authenticate()

    # ── Authentication ────────────────────────────────────────────────────────

    def _authenticate(self):
        """
        POST /auth/accesstokenrequest
        Returns a bearer token valid for ~24 hours.
        """
        url = f"{self.base_url}/auth/accesstokenrequest"
        payload = {
            "name":       self.username,
            "password":   self.password,
            "appId":      "NQ_ES_Dashboard",
            "appVersion": "1.0",
            "cid":        0,
            "sec":        "",            # leave blank for personal use
        }
        try:
            resp = requests.post(url, json=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if "errorText" in data:
                raise ValueError(f"Tradovate auth error: {data['errorText']}")

            self.token     = data.get("accessToken")
            self.token_exp = time.time() + 82800  # ~23 hours
            logger.info("Tradovate authenticated successfully (demo=%s)", self.demo)

        except requests.exceptions.RequestException as e:
            raise ConnectionError(
                f"Cannot connect to Tradovate. Check your internet connection.\n"
                f"Error: {e}\n"
                f"URL tried: {url}"
            )

    def _ensure_token(self):
        """Re-authenticate if token is near expiry."""
        if self.token is None or time.time() > self.token_exp - 300:
            logger.info("Refreshing Tradovate token…")
            self._authenticate()

    def _headers(self) -> dict:
        self._ensure_token()
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type":  "application/json",
        }

    # ── Contract lookup ───────────────────────────────────────────────────────

    def get_contract_id(self, symbol: str) -> int:
        """
        Look up the numeric contract ID for a symbol string.
        e.g. "NQU4" → 123456
        """
        mapped = SYMBOL_MAP.get(symbol.upper(), symbol)
        url    = f"{self.base_url}/contract/find"
        resp   = requests.get(url, params={"name": mapped}, headers=self._headers(), timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            raise ValueError(
                f"Contract '{mapped}' not found. "
                f"Check SYMBOL_MAP in topstep_connector.py — "
                f"the front-month contract rolls quarterly (Mar=H, Jun=M, Sep=U, Dec=Z)."
            )
        return data[0]["id"]

    # ── Bar / candle data ─────────────────────────────────────────────────────

    def get_bars(
        self,
        symbol:    str,
        tf:        str,
        n_bars:    int  = 300,
        end_time:  datetime | None = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars from the Tradovate market-data endpoint.

        Parameters
        ----------
        symbol  : "NQ" or "ES"  (maps to front-month contract via SYMBOL_MAP)
        tf      : "1min" | "5min" | "30min" | "1hour" | "1day"
        n_bars  : how many bars to fetch (max ~5000 per request)
        end_time: fetch bars ending at this UTC datetime (default = now)

        Returns
        -------
        pd.DataFrame with columns: timestamp, open, high, low, close, volume
        Index is a DatetimeIndex.
        """
        if tf not in TF_MAP:
            raise ValueError(f"Unknown timeframe '{tf}'. Choose from: {list(TF_MAP.keys())}")

        tf_cfg      = TF_MAP[tf]
        contract_id = self.get_contract_id(symbol)
        end_dt      = end_time or datetime.now(timezone.utc)
        start_dt    = end_dt - timedelta(days=self._bars_to_days(tf, n_bars))

        url = f"{MD_BASE}/md/getchart"
        payload = {
            "symbol":          SYMBOL_MAP.get(symbol.upper(), symbol),
            "chartDescription": {
                "underlyingType": "MinuteBar" if "min" in tf_cfg["unitType"].lower() else tf_cfg["unitType"],
                "elementSize":    tf_cfg["nUnits"],
                "elementSizeUnit":"UnderlyingUnits",
                "withHistogram":  False,
            },
            "timeRange": {
                "asMuchAsElements": n_bars,
            }
        }

        try:
            resp = requests.post(url, json=payload, headers=self._headers(), timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Tradovate market data request failed: {e}")

        # ── Parse response bars ───────────────────────────────────────────────
        bars = data.get("bars", [])
        if not bars:
            logger.warning("No bars returned for %s %s. Market may be closed.", symbol, tf)
            return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])

        df = pd.DataFrame(bars)

        # Tradovate returns timestamps as milliseconds since epoch
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["timestamp"] = df["timestamp"].dt.tz_convert("America/Chicago")  # CME timezone
        df = df.rename(columns={"open":"open","high":"high","low":"low","close":"close","upVolume":"volume"})

        # Keep only the columns we need
        keep = [c for c in ["timestamp","open","high","low","close","volume"] if c in df.columns]
        df   = df[keep].copy()
        df   = df.set_index("timestamp").sort_index()
        df.index.name = "timestamp"

        logger.info("Fetched %d %s bars for %s", len(df), tf, symbol)
        return df.reset_index()

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def _bars_to_days(tf: str, n_bars: int) -> int:
        """Estimate calendar days needed to get n_bars of trading data."""
        mins_per_bar = {"1min":1, "5min":5, "30min":30, "1hour":60, "1day":1440}
        mins_needed  = mins_per_bar.get(tf, 30) * n_bars
        trading_mins_per_day = 23 * 60  # NQ trades ~23 hours/day
        days = max(5, int(mins_needed / trading_mins_per_day) + 3)
        return days

    def account_info(self) -> dict:
        """
        Fetch your funded account info (balance, buying power, etc.)
        Useful for a future position sizing calculator.
        """
        url  = f"{self.base_url}/account/list"
        resp = requests.get(url, headers=self._headers(), timeout=10)
        resp.raise_for_status()
        return resp.json()

    def is_market_open(self) -> bool:
        """
        Basic check: NQ/ES trade Sun 5pm – Fri 4pm CT with a 1-hour break.
        This is a simplified check; use the Tradovate clock endpoint for precision.
        """
        now = datetime.now(timezone.utc)
        # Rough CME Globex schedule: closed Fri 4pm–Sun 5pm CT
        # Simplified: check if it's a weekday between 22:00–23:00 UTC (daily break)
        weekday = now.weekday()  # 0=Mon … 6=Sun
        hour_utc = now.hour

        if weekday == 4 and hour_utc >= 21:   return False  # Fri close
        if weekday == 5:                        return False  # Sat closed
        if weekday == 6 and hour_utc < 22:     return False  # Sun pre-open
        if 22 <= hour_utc < 23:                return False  # Daily maintenance
        return True


# ══════════════════════════════════════════════════════════════════════════════
# QUICK TEST
# Run this file directly to test your credentials:
#   python api/topstep_connector.py
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys

    username = os.getenv("TOPSTEP_USERNAME") or input("Tradovate username (email): ")
    password = os.getenv("TOPSTEP_PASSWORD") or input("Tradovate password: ")
    demo     = input("Use demo/sim account? (y/n, default y): ").strip().lower() != "n"

    print(f"\nConnecting to Tradovate ({'DEMO' if demo else 'LIVE'})…")
    try:
        conn   = TopstepConnector(username, password, demo=demo)
        print("✓ Authenticated successfully\n")

        print("Fetching NQ 30-min bars…")
        nq_df = conn.get_bars("NQ", "30min", n_bars=10)
        print(nq_df.to_string())

        print("\nFetching ES 30-min bars…")
        es_df = conn.get_bars("ES", "30min", n_bars=10)
        print(es_df.to_string())

        print("\n✓ Connection working. You can now use this in streamlit_app.py")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Check your username/password")
        print("  2. Make sure API access is enabled in Tradovate Settings")
        print("  3. Try demo=True first")
        sys.exit(1)
