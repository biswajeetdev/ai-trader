"""merger_arb.py — Merger arbitrage signal from news_catalyst output.
Detects M&A targets and surfaces the spread. Alert-only — no auto-execution.
Deal price not available from free APIs, so flags for review at +65 confidence.
"""

import re

_MA_KEYWORDS = re.compile(
    r"\b(acqui|merger|takeover|buyout|tender offer|go.private|going private|deal|bid for)\w*\b",
    re.IGNORECASE,
)


def detect_merger_arb(news_signals: list[dict]) -> list[dict]:
    """Scan news_catalyst output for M&A signals. Returns list of arb candidates."""
    arb = []
    seen = set()
    for sig in (news_signals or []):
        text   = sig.get("headline", "") + " " + sig.get("summary", "")
        symbol = sig.get("symbol", "")
        if not symbol or not _MA_KEYWORDS.search(text):
            continue
        if symbol in seen:
            continue
        seen.add(symbol)
        arb.append({
            "symbol":      symbol,
            "signal":      "MERGER_ARB",
            "confidence":  65,
            "reason":      f"M&A signal detected: {text[:120]}",
        })
    return arb


def get_arb_signal(symbol: str, news_signals: list[dict]) -> dict | None:
    """Return merger arb signal for a specific symbol, or None."""
    for arb in detect_merger_arb(news_signals):
        if arb["symbol"] == symbol:
            return {
                "action":     "BUY",
                "confidence": arb["confidence"],
                "reason":     arb["reason"],
                "strategy":   "MERGER_ARB",
            }
    return None
