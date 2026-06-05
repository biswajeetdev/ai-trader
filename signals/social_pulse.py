"""
social_pulse.py — Celebrity & influencer signal monitor
Watches Google News RSS for stock-moving statements from key figures.
No API keys needed. Completely free.

Signals returned:
  {"source": "Elon Musk", "text": "...", "tickers": ["TSLA"], "sentiment": 0.8, "urgency": "HIGH"}
"""

import re
import time
import json
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from pathlib import Path

CACHE_FILE = Path(__file__).parent.parent / "signals_cache.json"
CACHE_TTL  = 30 * 60  # 30 minutes — don't hammer RSS feeds

# ── Key figures to monitor and their known ticker associations ────────────────
INFLUENCERS = {
    "Elon Musk": {
        "query":   "Elon Musk stock buy invest tweet",
        "tickers": ["TSLA", "DOGE", "BTC", "X", "TWTR", "SPACEX"],
        "weight":  1.0,   # 1.0 = highest impact
    },
    "Donald Trump": {
        "query":   "Trump tweet stock tariff trade deal",
        "tickers": ["DJT", "DWAC", "META", "AAPL", "MSFT"],
        "weight":  0.9,
    },
    "Cathie Wood": {
        "query":   "Cathie Wood ARK invest buy",
        "tickers": ["TSLA", "COIN", "ROKU", "NVDA", "BTC"],
        "weight":  0.7,
    },
    "Warren Buffett": {
        "query":   "Buffett Berkshire buy stake acquire",
        "tickers": ["AAPL", "KO", "BAC", "OXY", "BRK"],
        "weight":  0.9,
    },
    "Michael Saylor": {
        "query":   "Saylor MicroStrategy bitcoin buy",
        "tickers": ["BTC", "MSTR"],
        "weight":  0.8,
    },
    "Chamath Palihapitiya": {
        "query":   "Chamath invest spac stock buy",
        "tickers": ["IPOE", "IPOF", "BTC"],
        "weight":  0.7,
    },
    "Jim Cramer": {
        "query":   "Cramer CNBC buy recommend stock",
        "tickers": [],   # dynamically parsed
        "weight":  -0.3, # NEGATIVE — Cramer is a famous inverse indicator
    },
}

# ── Sentiment keywords ────────────────────────────────────────────────────────
BULLISH_KW = ["buy", "invest", "bullish", "long", "opportunity", "recommend",
              "purchase", "acquire", "stake", "love", "great", "backing"]
BEARISH_KW = ["sell", "short", "bearish", "dump", "avoid", "warning",
              "crash", "drop", "concern", "problem", "issue"]

# ── Ticker extraction ─────────────────────────────────────────────────────────
KNOWN_TICKERS = {
    "apple": "AAPL", "microsoft": "MSFT", "nvidia": "NVDA", "tesla": "TSLA",
    "bitcoin": "BTC", "ethereum": "ETH", "google": "GOOGL", "amazon": "AMZN",
    "meta": "META", "netflix": "NFLX", "dell": "DELL", "intel": "INTC",
    "amd": "AMD", "palantir": "PLTR", "coinbase": "COIN", "twitter": "X",
    "spacex": "SPACEX", "dogecoin": "DOGE", "solana": "SOL",
}


def _fetch_rss(query, max_age_hours=6):
    """Fetch Google News RSS for a query, return items from last N hours."""
    url = (f"https://news.google.com/rss/search"
           f"?q={requests.utils.quote(query)}&hl=en-US&gl=US&ceid=US:en")
    try:
        resp = requests.get(url, timeout=10,
                            headers={"User-Agent": "Mozilla/5.0"})
        if not resp.ok:
            return []
        root  = ET.fromstring(resp.content)
        items = root.findall(".//item")
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        results = []
        for item in items[:10]:
            title = item.findtext("title", "")
            desc  = item.findtext("description", "")
            pub   = item.findtext("pubDate", "")
            link  = item.findtext("link", "")
            try:
                from email.utils import parsedate_to_datetime
                pub_dt = parsedate_to_datetime(pub).astimezone(timezone.utc)
                if pub_dt < cutoff:
                    continue
            except Exception:
                pass
            results.append({"title": title, "desc": desc,
                            "pub": pub, "link": link})
        return results
    except Exception:
        return []


def _score_sentiment(text):
    """Returns sentiment score: +1 = fully bullish, -1 = fully bearish."""
    text_lower = text.lower()
    bull = sum(1 for w in BULLISH_KW if w in text_lower)
    bear = sum(1 for w in BEARISH_KW if w in text_lower)
    total = bull + bear
    if total == 0:
        return 0.0
    return round((bull - bear) / total, 2)


def _extract_tickers(text, known_tickers):
    """Extract stock tickers from text."""
    found = set(known_tickers)
    text_lower = text.lower()
    for name, ticker in KNOWN_TICKERS.items():
        if name in text_lower:
            found.add(ticker)
    # Uppercase $TICKER pattern
    found.update(re.findall(r'\$([A-Z]{2,5})\b', text))
    return list(found)


def _load_cache():
    if not CACHE_FILE.exists():
        return {}
    try:
        data = json.loads(CACHE_FILE.read_text())
        # Expire cache
        if time.time() - data.get("_ts", 0) > CACHE_TTL:
            return {}
        return data
    except Exception:
        return {}


def _save_cache(data):
    data["_ts"] = time.time()
    CACHE_FILE.write_text(json.dumps(data, indent=2))


def get_social_signals(watchlist_tickers=None, use_cache=True):
    """
    Fetch social signals for all influencers.
    Returns list of signal dicts sorted by urgency.
    watchlist_tickers: if set, only return signals matching these tickers.
    """
    cache = _load_cache() if use_cache else {}
    if cache.get("social_signals"):
        return cache["social_signals"]

    all_signals = []

    for person, cfg in INFLUENCERS.items():
        items = _fetch_rss(cfg["query"], max_age_hours=6)
        for item in items:
            text      = f"{item['title']} {item['desc']}"
            sentiment = _score_sentiment(text) * cfg["weight"]
            tickers   = _extract_tickers(text, cfg["tickers"])

            if watchlist_tickers:
                tickers = [t for t in tickers if t in watchlist_tickers]
                if not tickers:
                    continue

            if abs(sentiment) < 0.1:
                continue  # skip neutral noise

            urgency = "HIGH" if abs(sentiment) >= 0.5 else "MEDIUM"

            all_signals.append({
                "source":    person,
                "headline":  item["title"][:200],
                "tickers":   tickers,
                "sentiment": round(sentiment, 2),
                "direction": "BULLISH" if sentiment > 0 else "BEARISH",
                "urgency":   urgency,
                "pub":       item["pub"],
                "link":      item["link"],
            })
        time.sleep(0.3)  # polite delay between RSS calls

    # Sort: HIGH urgency first, then by abs(sentiment)
    all_signals.sort(key=lambda x: (x["urgency"]=="HIGH", abs(x["sentiment"])), reverse=True)

    cache["social_signals"] = all_signals
    _save_cache(cache)
    return all_signals


def format_for_llm(signals, symbol):
    """Format relevant signals for a specific symbol as LLM context."""
    relevant = [s for s in signals if symbol in s.get("tickers", [])]
    if not relevant:
        return "No social signals for this asset in last 6 hours."
    lines = []
    for s in relevant[:3]:
        lines.append(f"[{s['urgency']}] {s['source']} → {s['direction']} "
                     f"(score {s['sentiment']}): \"{s['headline'][:120]}\"")
    return "\n".join(lines)


if __name__ == "__main__":
    print("Fetching social signals...")
    sigs = get_social_signals(use_cache=False)
    for s in sigs[:10]:
        print(f"  {s['urgency']:6} | {s['source']:20} | {s['direction']:7} "
              f"| {s['tickers']} | {s['headline'][:80]}")
