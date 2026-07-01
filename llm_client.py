"""
agent/llm_client.py
====================
Unified LLM client for the NQ/ES AI agent.

DEFAULT: Google Gemini (free tier)
SWITCH:  Set ANTHROPIC_API_KEY in Streamlit secrets → automatically upgrades to Claude

HOW TO SWITCH
─────────────
The active provider is chosen by checking which API key is available:

  1. If ANTHROPIC_API_KEY is set  → uses Claude claude-sonnet-4-6 (best reasoning)
  2. Else if GEMINI_API_KEY is set → uses Gemini 2.0 Flash (free tier, very capable)
  3. Else                          → raises a clear error with setup instructions

To get a free Gemini API key (takes 2 minutes):
  1. Go to https://aistudio.google.com/app/apikey
  2. Click "Create API key"
  3. Copy the key
  4. Add to Streamlit secrets:  GEMINI_API_KEY = "AIza..."
  5. Done — no credit card required

To upgrade to Anthropic Claude later:
  1. Go to https://console.anthropic.com
  2. Add to Streamlit secrets:  ANTHROPIC_API_KEY = "sk-ant-..."
  3. The system switches automatically — no code changes needed

FREE TIER LIMITS (Gemini 2.0 Flash as of June 2026)
─────────────────────────────────────────────────────
  Requests/min  : 15
  Requests/day  : 1,500
  Tokens/min    : 1,000,000
  This is more than enough for a trading dashboard refreshing every 30 seconds.

MODEL REFERENCE
───────────────
  Gemini free  : gemini-2.0-flash         (fast, free, great for structured output)
  Gemini paid  : gemini-1.5-pro           (deeper reasoning, paid)
  Anthropic    : claude-sonnet-4-6        (best reasoning, paid)
"""

from __future__ import annotations

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Model names ───────────────────────────────────────────────────────────────
GEMINI_MODEL    = "gemini-2.0-flash"
ANTHROPIC_MODEL = "claude-sonnet-4-6"

# ── Cached clients ────────────────────────────────────────────────────────────
_gemini_client    = None
_anthropic_client = None


def _get_api_key(env_name: str) -> str:
    """Read API key from env → Streamlit secrets → empty string."""
    val = os.environ.get(env_name, "")
    if val:
        return val
    try:
        import streamlit as st
        val = st.secrets.get(env_name, "")
        if val:
            os.environ[env_name] = val   # cache in env for subprocesses
    except Exception:
        pass
    return val


def active_provider() -> str:
    """Return which provider will be used: 'anthropic', 'gemini', or 'none'."""
    if _get_api_key("ANTHROPIC_API_KEY"):
        return "anthropic"
    if _get_api_key("GEMINI_API_KEY"):
        return "gemini"
    return "none"


def active_model() -> str:
    """Return the model name that will be used."""
    return ANTHROPIC_MODEL if active_provider() == "anthropic" else GEMINI_MODEL


def chat_complete(
    system_prompt: str,
    messages: list[dict],
    max_tokens: int = 600,
) -> str:
    """
    Send a conversation to the active LLM and return the text response.

    Parameters
    ----------
    system_prompt : str
        The system/persona prompt (same format for both providers).
    messages : list[dict]
        Conversation history: [{"role": "user"|"assistant", "content": "..."}]
    max_tokens : int
        Maximum tokens in the response.

    Returns
    -------
    str
        The assistant's text reply.

    Raises
    ------
    RuntimeError
        If no API key is configured.
    """
    provider = active_provider()

    if provider == "anthropic":
        return _call_anthropic(system_prompt, messages, max_tokens)
    elif provider == "gemini":
        return _call_gemini(system_prompt, messages, max_tokens)
    else:
        raise RuntimeError(
            "No LLM API key found.\n\n"
            "Option 1 — Gemini (free, takes 2 min):\n"
            "  1. Visit https://aistudio.google.com/app/apikey\n"
            "  2. Create an API key\n"
            "  3. Add to Streamlit secrets:  GEMINI_API_KEY = \"AIza...\"\n\n"
            "Option 2 — Anthropic Claude (paid, best quality):\n"
            "  1. Visit https://console.anthropic.com\n"
            "  2. Add to Streamlit secrets:  ANTHROPIC_API_KEY = \"sk-ant-...\"\n"
        )


# ══════════════════════════════════════════════════════════════════════════════
# GEMINI IMPLEMENTATION
# ══════════════════════════════════════════════════════════════════════════════

def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        api_key = _get_api_key("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set.")
        import google.genai as genai
        _gemini_client = genai.Client(api_key=api_key)
        logger.info("Gemini client initialised (model: %s)", GEMINI_MODEL)
    return _gemini_client


def _call_gemini(system_prompt: str, messages: list[dict], max_tokens: int) -> str:
    """
    Call Gemini via the google-genai SDK.
    Gemini uses a flat content list; system prompt is prepended as a user turn.
    """
    import google.genai as genai
    from google.genai import types as gtypes

    client = _get_gemini_client()

    # Build contents list: Gemini uses role="user"/"model" (not "assistant")
    contents = []
    for msg in messages:
        role    = "model" if msg["role"] == "assistant" else "user"
        contents.append(
            gtypes.Content(
                role=role,
                parts=[gtypes.Part(text=msg["content"])]
            )
        )

    config = gtypes.GenerateContentConfig(
        system_instruction=system_prompt,
        max_output_tokens=max_tokens,
        temperature=0.3,          # low temp for consistent structured output
        candidate_count=1,
    )

    try:
        response = client.models.generate_content(
            model    = GEMINI_MODEL,
            contents = contents,
            config   = config,
        )
        return response.text.strip()
    except Exception as e:
        logger.error("Gemini API error: %s", e)
        raise


# ══════════════════════════════════════════════════════════════════════════════
# ANTHROPIC IMPLEMENTATION
# ══════════════════════════════════════════════════════════════════════════════

def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        api_key = _get_api_key("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set.")
        import anthropic
        _anthropic_client = anthropic.Anthropic(api_key=api_key)
        logger.info("Anthropic client initialised (model: %s)", ANTHROPIC_MODEL)
    return _anthropic_client


def _call_anthropic(system_prompt: str, messages: list[dict], max_tokens: int) -> str:
    client = _get_anthropic_client()
    response = client.messages.create(
        model      = ANTHROPIC_MODEL,
        max_tokens = max_tokens,
        system     = system_prompt,
        messages   = messages,
    )
    return response.content[0].text.strip()


# ══════════════════════════════════════════════════════════════════════════════
# STATUS HELPER  (used by Streamlit sidebar)
# ══════════════════════════════════════════════════════════════════════════════

def provider_status() -> dict:
    """
    Return a dict describing the current provider status.
    Used to render the status badge in the Streamlit sidebar.
    """
    provider = active_provider()
    if provider == "anthropic":
        return {
            "provider":  "Anthropic Claude",
            "model":     ANTHROPIC_MODEL,
            "status":    "active",
            "badge":     "🟣 Claude claude-sonnet-4-6",
            "color":     "#a78bfa",
            "free":      False,
        }
    elif provider == "gemini":
        return {
            "provider":  "Google Gemini",
            "model":     GEMINI_MODEL,
            "status":    "active",
            "badge":     "🔵 Gemini 2.0 Flash (free)",
            "color":     "#60a5fa",
            "free":      True,
        }
    else:
        return {
            "provider":  "None",
            "model":     "—",
            "status":    "unconfigured",
            "badge":     "⚪ No AI key set",
            "color":     "#52525b",
            "free":      False,
        }


# ══════════════════════════════════════════════════════════════════════════════
# QUICK TEST
# python agent/llm_client.py
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys

    status = provider_status()
    print(f"\nProvider : {status['provider']}")
    print(f"Model    : {status['model']}")
    print(f"Status   : {status['status']}")
    print(f"Free tier: {status['free']}")

    if status["status"] != "active":
        print("\nNo API key configured. See instructions above.")
        sys.exit(1)

    print("\nSending test message…")
    try:
        reply = chat_complete(
            system_prompt = "You are a concise trading assistant. Reply in one sentence.",
            messages      = [{"role": "user", "content": "What is the VIX and why does it matter for NQ futures?"}],
            max_tokens    = 100,
        )
        print(f"Response : {reply}")
        print("\n✓ LLM client working correctly.")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
