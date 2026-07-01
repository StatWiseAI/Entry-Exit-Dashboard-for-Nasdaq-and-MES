"""
api/tradingview_connector.py
=============================
TradingView cannot push data directly to Python — it sends webhooks via alerts.
This file has two parts:

  PART 1 — Pine Script (copy into TradingView)
    A template that fires an alert on every new bar, sending OHLCV data
    as a JSON payload to your webhook URL.

  PART 2 — Python webhook receiver (Flask, runs alongside Streamlit)
    Receives the POST requests from TradingView, stores bars in memory,
    and makes them available to streamlit_app.py.

HOW IT WORKS END-TO-END
────────────────────────
  TradingView chart
      │  (new bar closes)
      │  HTTP POST → your webhook URL
      ▼
  Flask receiver (this file, running on your server)
      │  stores bar in memory / file
      ▼
  streamlit_app.py
      │  reads latest bars → run_strategy() → render decision
      ▼
  Browser → ENTER LONG / ENTER SHORT / WAIT

SETUP STEPS (complete guide in DEPLOYMENT_GUIDE.md)
────────────────────────────────────────────────────
Step 1: Deploy your app publicly (Streamlit Cloud or Railway)
  Your app needs a public HTTPS URL for TradingView to reach.
  Localhost will NOT work.

Step 2: Note your webhook URL
  If your app is at https://yourapp.streamlit.app, the webhook URL is:
    https://yourapp.streamlit.app/webhook
  (requires the Flask receiver to run on a separate port — see below)

  *** Easiest approach: use a free webhook service ***
  Deploy the Flask receiver to Railway.app (free tier) separately.
  It runs 24/7 and saves bars to a JSON file your Streamlit app reads.

Step 3: Copy the Pine Script (Part 1 below) into TradingView
  Chart → Pine Script Editor → New → paste → Save → Add to chart
  Then: Alerts → Create alert → select your script → Webhook URL → paste URL

Step 4: Configure the symbols
  Add the script to NQ1! chart first, then ES1! chart.
  Each sends its own webhook payload with the symbol name included.

Step 5: Test
  python api/tradingview_connector.py --test
  Should show "Webhook receiver running on port 8765"

OFFICIAL DOCS
─────────────
  Webhook alerts:  https://www.tradingview.com/support/solutions/43000529348
  Alert variables: https://www.tradingview.com/support/solutions/43000531021
"""

# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — PINE SCRIPT  (copy this into TradingView Pine Script Editor)
# ══════════════════════════════════════════════════════════════════════════════

PINE_SCRIPT_TEMPLATE = r"""
//@version=5
// ─────────────────────────────────────────────────────────────
// NQ/ES Data Feed  —  sends each closed bar to your webhook
// Copy this script to BOTH NQ1! and ES1! charts.
// ─────────────────────────────────────────────────────────────
indicator("NQ/ES Webhook Feed", overlay=true)

// Alert fires on every bar close
alertcondition(
    barstate.isconfirmed,
    title   = "Bar closed — send to webhook",
    message = '{"symbol":"' + syminfo.ticker + '",' +
              '"tf":"' + timeframe.period + '",' +
              '"ts":"{{time}}",' +
              '"o":{{open}},' +
              '"h":{{high}},' +
              '"l":{{low}},' +
              '"c":{{close}},' +
              '"v":{{volume}}}'
)

// Optional: plot a label so you know the script is active
if barstate.islast
    label.new(bar_index, high, "📡 Webhook active", style=label.style_label_down, size=size.small)
"""

# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — PYTHON WEBHOOK RECEIVER
# ══════════════════════════════════════════════════════════════════════════════

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from collections import defaultdict, deque
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── In-memory bar store ───────────────────────────────────────────────────────
# Structure: _BAR_STORE[symbol][timeframe] = deque of bar dicts
_BAR_STORE: dict = defaultdict(lambda: defaultdict(lambda: deque(maxlen=500)))

# ── Persistence file (survives server restarts) ───────────────────────────────
BARS_FILE = Path(__file__).parent / "tv_bars.json"


def _load_persisted_bars():
    """Load bars from disk on startup."""
    if BARS_FILE.exists():
        try:
            data = json.loads(BARS_FILE.read_text())
            for symbol, tfs in data.items():
                for tf, bars in tfs.items():
                    _BAR_STORE[symbol][tf].extend(bars)
            logger.info("Loaded %d symbols from persisted bar store", len(data))
        except Exception as e:
            logger.warning("Could not load persisted bars: %s", e)


def _save_bars():
    """Persist bars to disk."""
    try:
        out = {s: {tf: list(bars) for tf, bars in tfs.items()}
               for s, tfs in _BAR_STORE.items()}
        BARS_FILE.write_text(json.dumps(out))
    except Exception as e:
        logger.warning("Could not persist bars: %s", e)


def receive_bar(payload: dict):
    """
    Process a single bar payload received from TradingView.
    Expected JSON format (from the Pine Script above):
      {
        "symbol": "NQ1!",
        "tf":     "30",         ← TradingView uses minutes as a string
        "ts":     "1712001600000",
        "o":      24100.25,
        "h":      24150.00,
        "l":      24080.50,
        "c":      24130.75,
        "v":      1250
      }
    """
    try:
        symbol = _normalize_symbol(payload["symbol"])
        tf     = _normalize_tf(payload["tf"])
        ts_raw = payload["ts"]

        # TradingView sends timestamp as ms since epoch (integer or string)
        ts = pd.to_datetime(int(ts_raw), unit="ms", utc=True)
        ts = ts.tz_convert("America/Chicago")

        bar = {
            "timestamp": ts.strftime("%Y-%m-%d %H:%M"),
            "open":      float(payload["o"]),
            "high":      float(payload["h"]),
            "low":       float(payload["l"]),
            "close":     float(payload["c"]),
            "volume":    float(payload.get("v", 0)),
        }

        store = _BAR_STORE[symbol][tf]

        # Avoid duplicates: if last bar has same timestamp, update it (bar update)
        if store and store[-1]["timestamp"] == bar["timestamp"]:
            store[-1] = bar
        else:
            store.append(bar)

        _save_bars()
        logger.info("Bar stored: %s %s @ %s close=%.2f", symbol, tf, bar["timestamp"], bar["close"])
        return bar

    except (KeyError, ValueError, TypeError) as e:
        logger.error("Invalid bar payload: %s — error: %s", payload, e)
        raise ValueError(f"Could not parse bar payload: {e}")


def get_latest_tv_bars(symbol: str, tf: str, min_bars: int = 50) -> Optional[pd.DataFrame]:
    """
    Return the latest bars for a symbol/timeframe as a DataFrame.
    Returns None if fewer than min_bars are available.

    Called by streamlit_app.py → load_from_tradingview()
    """
    _load_persisted_bars()
    sym    = _normalize_symbol(symbol)
    tf_key = _normalize_tf(tf)

    bars = list(_BAR_STORE[sym][tf_key])
    if len(bars) < min_bars:
        logger.warning(
            "Only %d bars available for %s %s (need %d). "
            "Wait for more bars to arrive via webhook.",
            len(bars), sym, tf_key, min_bars
        )
        return None

    df = pd.DataFrame(bars)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    return df


def get_bar_count(symbol: str, tf: str) -> int:
    sym    = _normalize_symbol(symbol)
    tf_key = _normalize_tf(tf)
    return len(_BAR_STORE[sym][tf_key])


def _normalize_symbol(raw: str) -> str:
    """NQ1! / NQ / NQU4 → NQ"""
    raw = raw.upper().strip()
    if raw.startswith("NQ"):   return "NQ"
    if raw.startswith("ES"):   return "ES"
    return raw


def _normalize_tf(raw: str) -> str:
    """TradingView sends timeframe as minutes string; normalise to our keys."""
    mapping = {"1":"1min","3":"3min","5":"5min","15":"15min","30":"30min",
               "60":"1hour","D":"1day","1D":"1day","W":"1week"}
    return mapping.get(str(raw).upper(), f"{raw}min")


# ══════════════════════════════════════════════════════════════════════════════
# FLASK WEBHOOK SERVER
# Run this in a separate terminal or deploy to Railway alongside Streamlit:
#   python api/tradingview_connector.py --serve
# ══════════════════════════════════════════════════════════════════════════════

def run_webhook_server(host: str = "0.0.0.0", port: int = 8765, secret: str = ""):
    """
    Lightweight Flask server that receives TradingView webhook POSTs.

    Deploy this to Railway.app (free):
      1. Create a new Railway project
      2. Add this file + requirements-webhook.txt
      3. Set start command: python api/tradingview_connector.py --serve
      4. Railway gives you a public URL like https://xyz.railway.app
      5. Your TradingView webhook URL = https://xyz.railway.app/webhook

    Optional security: set TV_WEBHOOK_SECRET in Railway env vars.
    Add &secret=YOUR_SECRET to the webhook URL in TradingView.
    """
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        print("Flask not installed. Run:  pip install flask")
        return

    _load_persisted_bars()
    app = Flask(__name__)
    logging.basicConfig(level=logging.INFO)

    @app.route("/", methods=["GET"])
    def health():
        total = sum(
            len(bars)
            for tfs in _BAR_STORE.values()
            for bars in tfs.values()
        )
        return jsonify({
            "status":     "running",
            "symbols":    list(_BAR_STORE.keys()),
            "total_bars": total,
            "time":       datetime.utcnow().isoformat(),
        })

    @app.route("/webhook", methods=["POST"])
    def webhook():
        # Optional secret check
        if secret:
            incoming = request.args.get("secret", "")
            if incoming != secret:
                return jsonify({"error": "Unauthorized"}), 403

        try:
            payload = request.get_json(force=True)
            if payload is None:
                return jsonify({"error": "No JSON body received"}), 400

            bar = receive_bar(payload)
            return jsonify({"status": "ok", "bar": bar}), 200

        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.exception("Unexpected webhook error")
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/bars/<symbol>/<tf>", methods=["GET"])
    def get_bars_endpoint(symbol, tf):
        """Debug endpoint: view stored bars for a symbol/tf."""
        bars = list(_BAR_STORE[_normalize_symbol(symbol)][_normalize_tf(tf)])
        return jsonify({"symbol": symbol, "tf": tf, "count": len(bars), "bars": bars[-20:]})

    print(f"\n{'='*55}")
    print(f"  TradingView Webhook Receiver")
    print(f"  Listening on http://{host}:{port}")
    print(f"  Webhook URL: http://YOUR_PUBLIC_HOST:{port}/webhook")
    print(f"  Debug bars:  http://YOUR_PUBLIC_HOST:{port}/bars/NQ/30min")
    print(f"{'='*55}\n")

    app.run(host=host, port=port, debug=False)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TradingView webhook receiver")
    parser.add_argument("--serve",  action="store_true", help="Start the webhook server")
    parser.add_argument("--host",   default="0.0.0.0",   help="Host to bind (default 0.0.0.0)")
    parser.add_argument("--port",   type=int, default=8765, help="Port (default 8765)")
    parser.add_argument("--secret", default="",          help="Optional webhook secret")
    parser.add_argument("--test",   action="store_true", help="Inject a test bar and print it")
    parser.add_argument("--show-pine", action="store_true", help="Print the Pine Script template")
    args = parser.parse_args()

    if args.show_pine:
        print(PINE_SCRIPT_TEMPLATE)

    elif args.test:
        print("Injecting test bar…")
        import time
        test_payload = {
            "symbol": "NQ1!",
            "tf":     "30",
            "ts":     str(int(time.time() * 1000)),
            "o":      24100.25,
            "h":      24150.00,
            "l":      24080.50,
            "c":      24130.75,
            "v":      1250,
        }
        bar = receive_bar(test_payload)
        print(f"✓ Stored: {bar}")
        count = get_bar_count("NQ", "30min")
        print(f"✓ NQ 30min bar count: {count}")

    elif args.serve:
        run_webhook_server(args.host, args.port, args.secret)

    else:
        parser.print_help()
