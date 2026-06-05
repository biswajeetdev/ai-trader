"""
polymarket_signals.py — Prediction market signals (Tier 1: all three features)

1. get_prediction_signals()  — scan Polymarket for markets tied to our holdings
2. get_rate_arbitrage_signal() — bond market vs Polymarket Fed rate divergence
3. scan_macro_markets()      — BTC/ETH direction + recession + S&P500 outlook

API: gamma-api.polymarket.com (public REST, no auth, no key needed)
Cache: 30 min — prediction market odds move slowly
"""

import json
import time
import requests
from pathlib import Path

GAMMA_API    = "https://gamma-api.polymarket.com"
CACHE_FILE   = Path(__file__).parent.parent / "signals_cache.json"
CACHE_TTL    = 1800   # 30 min

# Map watchlist symbols → Polymarket search keywords
SYMBOL_KEYWORDS: dict[str, list[str]] = {
    "BTC":          ["bitcoin price", "bitcoin above", "BTC"],
    "ETH":          ["ethereum price", "ethereum", "ETH"],
    "NVDA":         ["nvidia", "NVDA"],
    "AAPL":         ["apple stock", "AAPL"],
    "MSFT":         ["microsoft", "MSFT"],
    "RELIANCE.NS":  ["india stock", "nifty"],
    "TCS.NS":       ["india IT", "nifty"],
    "HDFCBANK.NS":  ["india bank", "nifty"],
}

# Macro markets always fetched regardless of watchlist
MACRO_KEYWORDS = [
    "federal reserve rate cut",
    "recession 2026",
    "S&P 500",
    "stock market crash",
]


# ── Cache helpers ──────────────────────────────────────────────────────────────

def _cache_get(key: str):
    try:
        data = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
        entry = data.get(key)
        if entry and time.time() - entry.get("ts", 0) < CACHE_TTL:
            return entry["data"]
    except Exception:
        pass
    return None


def _cache_set(key: str, val) -> None:
    try:
        data = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
        data[key] = {"data": val, "ts": time.time()}
        CACHE_FILE.write_text(json.dumps(data, indent=2))
    except Exception:
        pass


# ── Polymarket API ─────────────────────────────────────────────────────────────

def _fetch_markets(keyword: str, limit: int = 3) -> list[dict]:
    """Search active Polymarket markets by keyword. Returns list of market dicts."""
    try:
        r = requests.get(
            f"{GAMMA_API}/markets",
            params={"keyword": keyword, "limit": limit, "active": "true", "closed": "false"},
            timeout=8,
        )
        if not r.ok:
            return []
        return r.json() if isinstance(r.json(), list) else r.json().get("markets", [])
    except Exception:
        return []


def _parse_market(m: dict) -> dict | None:
    """Extract question, yes_prob, volume, end_date from a raw market dict."""
    try:
        question = m.get("question", "")
        # outcomePrices is a JSON-encoded string like '["0.43","0.57"]'
        raw_prices = m.get("outcomePrices", "[]")
        prices = json.loads(raw_prices) if isinstance(raw_prices, str) else raw_prices
        outcomes = json.loads(m.get("outcomes", '["Yes","No"]'))
        yes_idx  = outcomes.index("Yes") if "Yes" in outcomes else 0
        yes_prob = round(float(prices[yes_idx]) * 100, 1)
        volume   = round(float(m.get("volume", 0) or 0) / 1000, 1)   # in $k
        end_date = (m.get("endDate", "") or "")[:10]
        return {
            "question": question,
            "yes_prob": yes_prob,
            "volume_k": volume,
            "end_date": end_date,
        }
    except Exception:
        return None


# ── Public API ─────────────────────────────────────────────────────────────────

def get_prediction_signals(watchlist_tickers: list[str] | None = None) -> list[dict]:
    """
    Fetch Polymarket markets relevant to our watchlist symbols + macro markets.
    Returns list of signal dicts: {symbol, question, yes_prob, volume_k, end_date}
    """
    cached = _cache_get("polymarket_signals")
    if cached:
        return cached

    signals = []
    tickers = watchlist_tickers or list(SYMBOL_KEYWORDS.keys())

    # Per-symbol markets
    for sym in tickers:
        keywords = SYMBOL_KEYWORDS.get(sym) or SYMBOL_KEYWORDS.get(sym.split(".")[0], [sym])
        for kw in keywords[:1]:   # one keyword per symbol to stay fast
            for m in _fetch_markets(kw, limit=2):
                parsed = _parse_market(m)
                if parsed and parsed["volume_k"] > 1:   # skip dust markets
                    signals.append({"symbol": sym, **parsed})
            break

    # Macro markets (Fed rate, recession, S&P)
    for kw in MACRO_KEYWORDS:
        for m in _fetch_markets(kw, limit=1):
            parsed = _parse_market(m)
            if parsed and parsed["volume_k"] > 10:
                signals.append({"symbol": "_macro", **parsed})

    _cache_set("polymarket_signals", signals)
    return signals


def get_rate_arbitrage_signal() -> dict | None:
    """
    Bond market vs Polymarket Fed rate divergence signal.

    Compares:
      - Polymarket "Fed rate cut" probability (prediction market crowd)
      - FRED fed_rate from openbb_macro (current rate level as proxy)

    Returns dict with divergence assessment, or None if data unavailable.
    """
    cached = _cache_get("rate_arb_signal")
    if cached:
        return cached

    # Get Polymarket Fed rate cut probability
    fed_markets = _fetch_markets("federal reserve rate cut", limit=3)
    poly_prob = None
    for m in fed_markets:
        parsed = _parse_market(m)
        if parsed and parsed["volume_k"] > 5:
            poly_prob = parsed["yes_prob"]
            poly_question = parsed["question"]
            break

    if poly_prob is None:
        return None

    # Get bond market rate signal via openbb_macro (already cached there)
    fed_rate = None
    try:
        from signals.openbb_macro import get_macro_enrichment
        macro = get_macro_enrichment()
        fed_rate = macro.get("fed_rate")
    except Exception:
        pass

    # Divergence: crowd expects cut (>60%) but rate is still high (>4%)
    # or crowd expects hold (<30%) but rate is low (<2%)
    result = {
        "poly_cut_prob":  poly_prob,
        "poly_question":  poly_question,
        "fed_rate":       fed_rate,
        "signal":         "NEUTRAL",
        "summary":        f"Polymarket Fed cut: {poly_prob}%",
    }
    if poly_prob is not None and fed_rate is not None:
        if poly_prob > 60 and fed_rate > 4.0:
            result["signal"]  = "DOVISH_EXPECTED"
            result["summary"] += f" — market expects cut despite rate={fed_rate}% (bullish equities)"
        elif poly_prob < 30 and fed_rate < 2.0:
            result["signal"]  = "HAWKISH_EXPECTED"
            result["summary"] += f" — no cut expected despite low rate={fed_rate}% (neutral)"
        else:
            result["summary"] += f" (rate={fed_rate}%, no arb)"

    _cache_set("rate_arb_signal", result)
    return result


def format_for_llm(signals: list[dict], symbol: str) -> str:
    """Format Polymarket signals for a specific symbol into LLM context string."""
    sym_sigs = [s for s in signals if s.get("symbol") in (symbol, "_macro")]
    if not sym_sigs:
        return ""

    lines = []
    for s in sym_sigs[:4]:   # cap at 4 per symbol
        tag = "[MACRO]" if s["symbol"] == "_macro" else "[POLY]"
        lines.append(
            f"{tag} {s['question'][:80]} → {s['yes_prob']}% Yes "
            f"(${s['volume_k']:.0f}k vol, ends {s['end_date']})"
        )

    # Append rate arb signal if relevant (crypto + equities care about Fed)
    if symbol in ("BTC", "ETH", "NVDA", "AAPL", "MSFT"):
        try:
            arb = get_rate_arbitrage_signal()
            if arb and arb["signal"] != "NEUTRAL":
                lines.append(f"[RATE-ARB] {arb['summary']}")
        except Exception:
            pass

    return "\n".join(lines)
