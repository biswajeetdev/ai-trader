"""
wsb_social.py — Reddit WallStreetBets + Truth Social + StockTwits signals
All free, no API keys needed.
"""

import re, time, requests, json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from pathlib import Path

CACHE_FILE = Path(__file__).parent.parent / "signals_cache.json"

WATCHLIST_MAP = {
    "NVDA": ["nvda","nvidia","jensen","blackwell","hopper"],
    "AAPL": ["aapl","apple","iphone","tim cook","vision pro"],
    "MSFT": ["msft","microsoft","azure","copilot","satya"],
    "BTC":  ["btc","bitcoin","btc-usd","sats","satoshi","spot etf"],
    "ETH":  ["eth","ethereum","vitalik","merge","staking","dencun"],
}

BULL_KW = ["moon","rocket","buy","calls","bullish","long","yolo","squeeze",
           "ath","breakout","pump","rip","send it","🚀","💎","🔥"]
BEAR_KW = ["puts","short","crash","dump","sell","bearish","bubble",
           "overvalued","rug","collapse","bagholders","🩳"]


def _cache_get(key, ttl=900):  # 15 min for wsb
    try:
        data  = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
        entry = data.get(key)
        if entry and time.time() - entry.get("ts", 0) < ttl:
            return entry.get("data")
    except Exception:
        pass
    return None


def _cache_set(key, value):
    try:
        data = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
        data[key] = {"data": value, "ts": time.time()}
        CACHE_FILE.write_text(json.dumps(data, indent=2))
    except Exception:
        pass


def _rss_items(url, max_age_hours=6):
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if not r.ok:
            return []
        root   = ET.fromstring(r.content)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        out    = []
        for item in root.findall(".//item")[:20]:
            title = item.findtext("title", "")
            desc  = item.findtext("description", "") or ""
            pub   = item.findtext("pubDate", "")
            try:
                from email.utils import parsedate_to_datetime
                if parsedate_to_datetime(pub).astimezone(timezone.utc) < cutoff:
                    continue
            except Exception:
                pass
            out.append({"title": title, "desc": desc[:300], "pub": pub})
        return out
    except Exception:
        return []


def _score(text, ticker):
    text_l = text.lower()
    mentions = sum(1 for kw in WATCHLIST_MAP.get(ticker, []) if kw in text_l)
    if mentions == 0:
        return 0.0
    bull = sum(1 for w in BULL_KW if w in text_l)
    bear = sum(1 for w in BEAR_KW if w in text_l)
    total = bull + bear
    return round((bull - bear) / max(total, 1), 2)


# ── Reddit WSB ────────────────────────────────────────────────────────────────

def get_wsb_signals(watchlist_tickers=None):
    cached = _cache_get("wsb")
    if cached is not None:
        return cached

    signals = []
    # WSB via old.reddit RSS (no auth, always works)
    feeds = [
        "https://old.reddit.com/r/wallstreetbets/hot/.rss",
        "https://old.reddit.com/r/stocks/hot/.rss",
        "https://old.reddit.com/r/investing/hot/.rss",
        "https://old.reddit.com/r/options/hot/.rss",
    ]

    for feed_url in feeds:
        for item in _rss_items(feed_url, max_age_hours=12):
            text = f"{item['title']} {item['desc']}"
            for ticker in (watchlist_tickers or list(WATCHLIST_MAP.keys())):
                score = _score(text, ticker)
                if abs(score) >= 0.2:
                    signals.append({
                        "source":    "Reddit WSB",
                        "ticker":    ticker,
                        "headline":  item["title"][:150],
                        "sentiment": score,
                        "direction": "BULLISH" if score > 0 else "BEARISH",
                        "urgency":   "HIGH" if abs(score) >= 0.6 else "MEDIUM",
                        "pub":       item["pub"],
                    })
        time.sleep(0.5)

    signals.sort(key=lambda x: abs(x["sentiment"]), reverse=True)
    _cache_set("wsb", signals)
    return signals


# ── Truth Social (Trump) via RSS mirror ───────────────────────────────────────

TRUMP_RSS_MIRRORS = [
    "https://trumpstruth.org/feed",
    "https://truthsocial.com/@realDonaldTrump/feed.rss",
    "https://news.google.com/rss/search?q=Trump+Truth+Social+stock+buy+sell+tariff&hl=en-US",
]

TRUMP_STOCK_PATTERNS = [
    (r'\b(BUY|BUYING|INVEST)\b', "BULLISH"),
    (r'\b(SELL|SELLING|DUMP)\b', "BEARISH"),
    (r'tariff[s]? on', "BEARISH"),   # tariffs = bad for stocks
    (r'trade deal', "BULLISH"),
    (r'great company|great stock', "BULLISH"),
    (r'LIBERATE|freedom|drill|energy', "BULLISH"),  # energy/oil stocks
]

def get_trump_signals(watchlist_tickers=None):
    cached = _cache_get("trump_social")
    if cached is not None:
        return cached

    signals = []
    for url in TRUMP_RSS_MIRRORS:
        items = _rss_items(url, max_age_hours=24)
        if items:
            for item in items:
                text = f"{item['title']} {item['desc']}"
                # Check for stock mentions
                for ticker in (watchlist_tickers or list(WATCHLIST_MAP.keys())):
                    if any(kw in text.lower() for kw in WATCHLIST_MAP.get(ticker, [])):
                        for pat, direction in TRUMP_STOCK_PATTERNS:
                            if re.search(pat, text, re.IGNORECASE):
                                signals.append({
                                    "source":    "Trump/Truth Social",
                                    "ticker":    ticker,
                                    "headline":  item["title"][:150],
                                    "sentiment": 0.9 if direction=="BULLISH" else -0.9,
                                    "direction": direction,
                                    "urgency":   "HIGH",
                                    "pub":       item["pub"],
                                })
                                break
            break  # use first working mirror
        time.sleep(0.3)

    _cache_set("trump_social", signals)
    return signals


# ── StockTwits sentiment (public API) ─────────────────────────────────────────

def get_stocktwits_sentiment(ticker):
    """StockTwits public trending API — no auth for basic sentiment."""
    cached = _cache_get(f"st_{ticker}")
    if cached is not None:
        return cached

    try:
        r = requests.get(
            f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json",
            timeout=8, headers={"User-Agent": "Mozilla/5.0"}
        )
        if not r.ok:
            return None
        data = r.json()
        msgs = data.get("messages", [])
        if not msgs:
            return None

        bull = sum(1 for m in msgs if m.get("entities",{}).get("sentiment",{}).get("basic")=="Bullish")
        bear = sum(1 for m in msgs if m.get("entities",{}).get("sentiment",{}).get("basic")=="Bearish")
        total = bull + bear
        if total == 0:
            return None

        score = (bull - bear) / total
        result = {
            "source":    "StockTwits",
            "ticker":    ticker,
            "bull_pct":  round(bull/total*100, 1),
            "bear_pct":  round(bear/total*100, 1),
            "sentiment": round(score, 2),
            "direction": "BULLISH" if score > 0 else "BEARISH",
            "messages":  len(msgs),
        }
        _cache_set(f"st_{ticker}", result)
        return result
    except Exception:
        return None


# ── Combined social signal ────────────────────────────────────────────────────

def get_all_social(watchlist_tickers=None):
    wsb    = get_wsb_signals(watchlist_tickers)
    trump  = get_trump_signals(watchlist_tickers)
    twits  = []
    for t in (watchlist_tickers or list(WATCHLIST_MAP.keys())):
        if t in ("BTC","ETH"):
            continue   # StockTwits is US stocks focused
        s = get_stocktwits_sentiment(t)
        if s:
            twits.append({**s, "urgency": "HIGH" if abs(s["sentiment"]) > 0.6 else "MEDIUM",
                          "headline": f"StockTwits: {s['bull_pct']}% bull / {s['bear_pct']}% bear ({s['messages']} msgs)"})

    all_ = wsb + trump + twits
    all_.sort(key=lambda x: (x["urgency"]=="HIGH", abs(x["sentiment"])), reverse=True)
    return all_


def format_for_llm(signals, symbol):
    rel = [s for s in signals if s.get("ticker","").upper() == symbol.upper()]
    if not rel:
        return "No WSB/social signals."
    return "\n".join(
        f"[{s['source']}] {s['direction']} {s['urgency']} "
        f"(score {s['sentiment']}): {s.get('headline','')[:100]}"
        for s in rel[:3]
    )


if __name__ == "__main__":
    print("=== Reddit WSB ===")
    wsb = get_wsb_signals()
    print(f"Found {len(wsb)} WSB signals")
    for s in wsb[:5]:
        print(f"  {s['urgency']:6} {s['ticker']:5} {s['direction']:8} {s['headline'][:60]}")

    print("\n=== Trump/Truth Social ===")
    tr = get_trump_signals()
    print(f"Found {len(tr)} Trump signals")
    for s in tr[:3]:
        print(f"  {s['ticker']:5} {s['direction']:8} {s['headline'][:60]}")

    print("\n=== StockTwits ===")
    for t in ["NVDA","AAPL","MSFT"]:
        s = get_stocktwits_sentiment(t)
        if s:
            print(f"  {t}: {s['bull_pct']}% bull / {s['bear_pct']}% bear ({s['messages']} msgs)")
