"""
insider_trades.py — Government official & insider trade tracker

Sources:
  US Congress : Google News RSS (congressional trading stories) +
                Quiver Quantitative API (free tier, optional API key)
  India NSE   : NSE bulk/block deals endpoint (official public data)
  UK/Global   : Google News RSS (MP financial interest stories)

No mandatory API keys — works out of the box.
Optional: set QUIVER_API_KEY env var for richer US Congress data.
"""

import json, os, re, time, requests
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from pathlib import Path

CACHE_FILE = Path(__file__).parent.parent / "signals_cache.json"
CACHE_TTL  = 60 * 60   # 1 hour

QUIVER_CONGRESS_URL = "https://api.quiverquant.com/beta/live/congresstrading"

KNOWN_TICKERS = {
    "apple":"AAPL","microsoft":"MSFT","nvidia":"NVDA","tesla":"TSLA",
    "bitcoin":"BTC","ethereum":"ETH","google":"GOOGL","alphabet":"GOOGL",
    "amazon":"AMZN","meta":"META","netflix":"NFLX","dell":"DELL",
    "intel":"INTC","amd":"AMD","palantir":"PLTR","coinbase":"COIN",
    "dogecoin":"DOGE","solana":"SOL","nvidia":"NVDA",
}


# ── cache helpers ─────────────────────────────────────────────────────────────

def _load_cache():
    try:
        return json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
    except Exception:
        return {}

def _save_cache(key, value):
    data = _load_cache()
    data[key] = {"data": value, "ts": time.time()}
    CACHE_FILE.write_text(json.dumps(data, indent=2))

def _cache_fresh(key, ttl=CACHE_TTL):
    entry = _load_cache().get(key)
    if entry and time.time() - entry.get("ts", 0) < ttl:
        return entry.get("data")
    return None

def _filter_by_ticker(trades, tickers):
    if not tickers:
        return trades
    return [t for t in trades if t.get("ticker","").upper() in tickers]


# ── RSS news fetch ─────────────────────────────────────────────────────────────

def _fetch_rss(query, max_age_hours=48):
    url = (f"https://news.google.com/rss/search"
           f"?q={requests.utils.quote(query)}&hl=en-US&gl=US&ceid=US:en")
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent":"Mozilla/5.0"})
        if not r.ok:
            return []
        root  = ET.fromstring(r.content)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        results = []
        for item in root.findall(".//item")[:15]:
            title = item.findtext("title","")
            pub   = item.findtext("pubDate","")
            link  = item.findtext("link","")
            try:
                from email.utils import parsedate_to_datetime
                pub_dt = parsedate_to_datetime(pub).astimezone(timezone.utc)
                if pub_dt < cutoff:
                    continue
            except Exception:
                pass
            results.append({"title": title, "pub": pub, "link": link})
        return results
    except Exception:
        return []

def _extract_tickers(text):
    found = set()
    text_lower = text.lower()
    for name, ticker in KNOWN_TICKERS.items():
        if name in text_lower:
            found.add(ticker)
    found.update(re.findall(r'\$([A-Z]{2,5})\b', text))
    found.update(re.findall(r'\b([A-Z]{2,4})\b', text))  # bare uppercase
    # Remove common false positives
    found -= {"US","UK","SEC","ETF","IPO","CEO","CFO","AI","IT","GDP","GDP","API","RSS"}
    return list(found)

def _sentiment(text):
    bull = sum(1 for w in ["buy","purchase","acquired","long","bullish","bought"] if w in text.lower())
    bear = sum(1 for w in ["sell","sold","short","bearish","dumped","divest"] if w in text.lower())
    if bull > bear: return "BULLISH", "buy"
    if bear > bull: return "BEARISH", "sell"
    return "NEUTRAL", "hold"


# ── US CONGRESS ───────────────────────────────────────────────────────────────

def get_congress_trades(watchlist_tickers=None, days_back=60):
    """
    Fetch US Congress STOCK Act trades via Quiver Quantitative (free, no key).
    1000 most recent trades including Representative, Ticker, Transaction,
    Amount, Party, ExcessReturn, PriceChange.
    """
    cached = _cache_fresh("congress_trades")
    if cached:
        return _filter_by_ticker(cached, watchlist_tickers)

    results = []
    cutoff  = datetime.now(timezone.utc) - timedelta(days=days_back)

    try:
        r = requests.get(QUIVER_CONGRESS_URL, timeout=20,
                         headers={"User-Agent": "Mozilla/5.0",
                                  "Accept":     "application/json"})
        if r.ok:
            trades = r.json()
            for t in trades:
                ticker     = (t.get("Ticker") or "").strip().upper()
                action_raw = (t.get("Transaction") or "").lower()
                name       = t.get("Representative", "Unknown")
                party      = t.get("Party", "")
                date_str   = t.get("TransactionDate") or t.get("ReportDate", "")
                amount     = t.get("Range") or t.get("Amount", "N/A")
                house      = t.get("House", "")
                excess_ret = t.get("ExcessReturn")   # positive = outperformed SPY after trade
                price_chg  = t.get("PriceChange")

                if not ticker or ticker in ("--","N/A",""):
                    continue

                # Parse date
                try:
                    tx_dt = datetime.strptime(date_str[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    if tx_dt < cutoff:
                        continue
                except Exception:
                    pass

                action = ("buy"  if "purchase" in action_raw else
                          "sell" if "sale"     in action_raw else action_raw)

                results.append({
                    "source":      f"US Congress ({house})",
                    "country":     "US",
                    "person":      name,
                    "party":       party,
                    "ticker":      ticker,
                    "action":      action,
                    "amount":      str(amount),
                    "date":        date_str,
                    "excess_ret":  excess_ret,   # useful: +ve = insiders were right
                    "price_chg":   price_chg,
                    "signal":      ("BULLISH" if action == "buy"  else
                                    "BEARISH" if action == "sell" else "NEUTRAL"),
                })
    except Exception as e:
        print(f"  [warn] Congress API failed: {e}")

    # Sort newest first
    results.sort(key=lambda x: x.get("date",""), reverse=True)
    _save_cache("congress_trades", results)
    return _filter_by_ticker(results, watchlist_tickers)


# ── INDIA NSE INSIDER TRADES ──────────────────────────────────────────────────

def get_india_insider_trades(watchlist_tickers=None, days_back=14):
    """
    Fetch NSE India bulk/block deals + insider trading disclosures.
    Official public data — no auth required.
    """
    cached = _cache_fresh("india_insider")
    if cached:
        return _filter_by_ticker(cached, watchlist_tickers)

    results = []
    session = requests.Session()
    session.headers.update({
        "User-Agent":      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept":          "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer":         "https://www.nseindia.com/",
        "X-Requested-With":"XMLHttpRequest",
    })

    # Prime session cookies (NSE requires this)
    try:
        session.get("https://www.nseindia.com/market-data/bulk-and-block-deals",
                    timeout=8)
        time.sleep(1)
    except Exception:
        pass

    endpoints = [
        ("bulk",  "https://www.nseindia.com/api/bulk-deal-data?isView=true"),
        ("block", "https://www.nseindia.com/api/block-deal-data?isView=true"),
    ]

    for deal_type, url in endpoints:
        try:
            r = session.get(url, timeout=12)
            if not r.ok:
                continue
            data  = r.json()
            deals = data.get("data", data) if isinstance(data,dict) else data
            for d in deals[:40]:
                symbol    = (d.get("symbol") or d.get("SYMBOL","")).upper().strip()
                client    = d.get("clientName") or d.get("CLIENT_NAME","Unknown")
                buy_sell  = (d.get("buySell") or d.get("BUY_SELL","")).upper()
                qty       = d.get("quantity") or d.get("QTY_TRADED","")
                price     = d.get("tradePrice") or d.get("TRADE_PRICE","")
                date_str  = d.get("date") or d.get("DATE","")
                if not symbol:
                    continue
                results.append({
                    "source":  f"NSE India ({deal_type})",
                    "country": "India",
                    "person":  str(client)[:40],
                    "ticker":  symbol,
                    "action":  "buy" if "B" in buy_sell else "sell",
                    "qty":     str(qty),
                    "price":   str(price),
                    "date":    str(date_str)[:10],
                    "signal":  "BULLISH" if "B" in buy_sell else "BEARISH",
                })
        except Exception:
            continue

    # Fallback: BSE India insider trading news
    if not results:
        for item in _fetch_rss("NSE BSE insider trading promoter bought sold India", max_age_hours=72):
            signal, action = _sentiment(item["title"])
            results.append({
                "source":  "India (News)",
                "country": "India",
                "person":  "Indian Insider",
                "ticker":  "",
                "action":  action,
                "headline": item["title"][:150],
                "date":    item["pub"][:16],
                "signal":  signal,
            })

    _save_cache("india_insider", results)
    return _filter_by_ticker(results, watchlist_tickers)


# ── GLOBAL OFFICIALS (UK, EU, others via News) ────────────────────────────────

def get_global_official_trades(watchlist_tickers=None):
    """Google News RSS for international government official trading stories."""
    cached = _cache_fresh("global_trades", ttl=6*3600)
    if cached:
        return cached

    queries = [
        "UK MP parliament shares financial interest bought",
        "EU parliament official stock bought sold",
        "government minister insider trading stock",
    ]
    results = []
    for q in queries:
        for item in _fetch_rss(q, max_age_hours=96):
            signal, action = _sentiment(item["title"])
            tickers = _extract_tickers(item["title"])
            if watchlist_tickers:
                tickers = [t for t in tickers if t in watchlist_tickers]
            results.append({
                "source":  "Global Official (News)",
                "country": "Intl",
                "person":  "Government Official",
                "ticker":  tickers[0] if tickers else "",
                "action":  action,
                "headline": item["title"][:150],
                "date":    item["pub"][:16],
                "signal":  signal,
            })
        time.sleep(0.2)

    _save_cache("global_trades", results)
    return results


# ── COMBINED FEED ─────────────────────────────────────────────────────────────

def get_all_insider_signals(watchlist_tickers=None):
    us     = get_congress_trades(watchlist_tickers)
    india  = get_india_insider_trades(watchlist_tickers)
    global_ = get_global_official_trades(watchlist_tickers)
    all_   = us + india + global_
    all_.sort(key=lambda x: (x["signal"]=="BULLISH", x.get("date","")), reverse=True)
    return all_


def format_for_llm(trades, symbol):
    relevant = [t for t in trades
                if t.get("ticker","").upper() == symbol.upper() or not t.get("ticker")]
    if not relevant:
        return "No recent govt/insider trades found."
    lines = []
    for t in relevant[:3]:
        headline = t.get("headline","")
        person   = t.get("person","Unknown")
        lines.append(f"[{t['country']}] {person} — {t['signal']} "
                     f"({t['action'].upper()} {t['ticker']} {t.get('date','')[:10]})"
                     + (f": {headline[:100]}" if headline else ""))
    return "\n".join(lines)


if __name__ == "__main__":
    print("=== US Congress ===")
    ct = get_congress_trades()
    print(f"Found {len(ct)} records")
    for t in ct[:5]:
        print(f"  {t['signal']:8} | {t['person'][:25]:25} | {t['action']:4} | {t.get('headline',t.get('ticker',''))[:70]}")

    print("\n=== India NSE ===")
    it = get_india_insider_trades()
    print(f"Found {len(it)} records")
    for t in it[:5]:
        print(f"  {t['signal']:8} | {t['person'][:25]:25} | {t['action']:4} {t['ticker']}")

    print("\n=== Global ===")
    gt = get_global_official_trades()
    print(f"Found {len(gt)} records")
    for t in gt[:3]:
        print(f"  {t['signal']:8} | {t['country']:6} | {t.get('headline','')[:70]}")
