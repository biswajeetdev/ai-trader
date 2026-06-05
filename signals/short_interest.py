"""
short_interest.py — FINRA short interest screen (free bimonthly data)
Source: FINRA OTC Transparency (https://www.finra.org/finra-data/browse-catalog/short-sale-volume-data)

Logic:
- Short Interest Ratio (SIR) = shares_short / avg_daily_volume
- SIR > 15 days → AVOID (high squeeze risk OR strong consensus short, avoid buying)
- Days-to-cover > 10 → flag as potential short squeeze candidate on strong catalyst
"""

import json, time, requests
from pathlib import Path
from datetime import datetime

CACHE_FILE = Path(__file__).parent.parent / "signals_cache.json"
CACHE_TTL  = 86400 * 2  # bimonthly data, 48h cache


def _cache(key, val=None):
    try:
        data = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
        if val is None:
            e = data.get(key)
            return e["data"] if e and time.time() - e.get("ts", 0) < CACHE_TTL else None
        data[key] = {"data": val, "ts": time.time()}
        CACHE_FILE.write_text(json.dumps(data, indent=2))
    except Exception:
        return None


def _fetch_finra_short(symbol: str) -> dict | None:
    """Try yfinance for short interest data (free, no API key)."""
    try:
        import yfinance as yf
        tk = yf.Ticker(symbol)
        info = tk.info
        shares_short = info.get("sharesShort", 0) or 0
        float_shares = info.get("floatShares", 0) or 0
        avg_volume = info.get("averageVolume", 0) or 0
        short_pct_float = info.get("shortPercentOfFloat", 0) or 0

        if avg_volume > 0 and shares_short > 0:
            sir = round(shares_short / avg_volume, 1)
        else:
            sir = 0.0

        return {
            "shares_short":      shares_short,
            "short_pct_float":   round(short_pct_float * 100, 1) if short_pct_float else 0,
            "sir_days":          sir,
            "avg_volume":        avg_volume,
        }
    except Exception:
        return None


def get_short_interest_signal(symbol: str, market: str = "us-stock") -> dict:
    """Returns short interest signal for a symbol."""
    if market != "us-stock":
        return {"signal": "NEUTRAL", "confidence_delta": 0, "sir_days": 0, "summary": "N/A (not US stock)"}

    cache_key = f"si_{symbol}"
    cached = _cache(cache_key)
    if cached:
        return cached

    data = _fetch_finra_short(symbol)
    if not data:
        return {"signal": "NEUTRAL", "confidence_delta": 0, "sir_days": 0, "summary": "SI data unavailable"}

    sir = data["sir_days"]
    spf = data["short_pct_float"]

    if sir > 15 or spf > 25:
        signal = "AVOID"
        delta = -15
        summary = f"SHORT_INTEREST: SIR={sir:.1f}d, {spf:.0f}% float — HIGH short, avoid long"
    elif sir > 8 or spf > 15:
        signal = "CAUTION"
        delta = -5
        summary = f"SHORT_INTEREST: SIR={sir:.1f}d, {spf:.0f}% float — elevated, caution"
    elif sir > 5 and spf > 8:
        signal = "SQUEEZE_WATCH"
        delta = 5
        summary = f"SHORT_INTEREST: SIR={sir:.1f}d, {spf:.0f}% float — potential squeeze on catalyst"
    else:
        signal = "CLEAN"
        delta = 3
        summary = f"SHORT_INTEREST: SIR={sir:.1f}d, {spf:.0f}% float — clean, no short overhang"

    result = {
        "signal":            signal,
        "confidence_delta":  delta,
        "sir_days":          sir,
        "short_pct_float":   spf,
        "summary":           summary,
    }
    _cache(cache_key, result)
    return result


if __name__ == "__main__":
    for sym in ["NVDA", "AAPL", "GME", "MSFT"]:
        r = get_short_interest_signal(sym)
        print(f"{sym}: {r['summary']}")
