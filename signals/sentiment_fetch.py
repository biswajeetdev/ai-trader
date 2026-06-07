"""signals/sentiment_fetch.py — Free social sentiment for any ticker.

Sources (no API key needed):
  StockTwits: public symbol stream
  Reddit WSB: search JSON endpoint (no auth)

Returns a short formatted string ready to inject into debate context.
"""

import time
from functools import lru_cache

try:
    import requests as _req
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

_CACHE: dict = {}
_CACHE_TTL   = 300  # 5 min — don't hammer free endpoints


def _stocktwits(symbol: str) -> tuple[float, list[str]]:
    url  = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
    resp = _req.get(url, timeout=6, headers={"User-Agent": "ai-trader/1.0"})
    if resp.status_code != 200:
        return 0.0, []
    msgs     = resp.json().get("messages", [])[:20]
    bulls    = sum(1 for m in msgs if (m.get("entities", {}).get("sentiment") or {}).get("basic") == "Bullish")
    bears    = sum(1 for m in msgs if (m.get("entities", {}).get("sentiment") or {}).get("basic") == "Bearish")
    total    = bulls + bears
    score    = (bulls - bears) / total if total else 0.0
    snippets = [m["body"][:60] for m in msgs[:3]]
    return score, snippets


def _reddit_wsb(symbol: str) -> tuple[float, list[str]]:
    url    = (f"https://www.reddit.com/r/wallstreetbets/search.json"
              f"?q={symbol}&sort=hot&limit=8&t=day")
    resp   = _req.get(url, timeout=6, headers={"User-Agent": "ai-trader/1.0"})
    if resp.status_code != 200:
        return 0.0, []
    posts  = resp.json().get("data", {}).get("children", [])
    scores = [p["data"].get("score", 0) for p in posts]
    upvote = [p["data"].get("upvote_ratio", 0.5) for p in posts]
    avg_uv = sum(upvote) / len(upvote) if upvote else 0.5
    score  = (avg_uv - 0.5) * 2  # -1..+1
    titles = [p["data"]["title"][:60] for p in posts[:3]]
    return score, titles


def get_sentiment(symbol: str) -> str:
    """Return formatted sentiment context string for injection into debate."""
    if not _HAS_REQUESTS:
        return ""
    now = time.time()
    if symbol in _CACHE and now - _CACHE[symbol]["ts"] < _CACHE_TTL:
        return _CACHE[symbol]["text"]

    st_score, st_snips = 0.0, []
    rd_score, rd_snips = 0.0, []
    try:
        st_score, st_snips = _stocktwits(symbol)
    except Exception:
        pass
    try:
        rd_score, rd_snips = _reddit_wsb(symbol)
    except Exception:
        pass

    if not st_snips and not rd_snips:
        return ""

    combined = (st_score + rd_score) / 2
    label    = "BULLISH" if combined > 0.15 else "BEARISH" if combined < -0.15 else "NEUTRAL"
    parts    = [f"LIVE SOCIAL SENTIMENT ({symbol}): {label} (score {combined:+.2f})"]
    if st_snips:
        parts.append(f"StockTwits: {' | '.join(st_snips)}")
    if rd_snips:
        parts.append(f"WSB: {' | '.join(rd_snips)}")

    text = "\n".join(parts)
    _CACHE[symbol] = {"ts": now, "text": text}
    return text
