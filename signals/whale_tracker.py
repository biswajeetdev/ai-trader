"""
whale_tracker.py — Hedge fund & billionaire position tracker
Sources: SEC EDGAR free API (no key needed)

Tracks 13D/13G filings (activist investors taking >5% stake) and
recent 13F quarterly holdings for major funds.

When a mega-fund files a new 13D on your watchlist stock = massive alpha signal.
These filings often precede 10-30% moves.

Whales tracked: Berkshire, Pershing Square, Tiger Global, Appaloosa,
                Third Point, Soros, Druckenmiller, D.E. Shaw, Two Sigma,
                Citadel, Renaissance, Elliott Management, ValueAct
"""

import json, re, time, requests
from datetime import datetime, timezone, timedelta
from pathlib import Path

CACHE_FILE = Path(__file__).parent.parent / "signals_cache.json"
CACHE_TTL  = 4 * 3600   # 4h — 13F/13D filings not that frequent

EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index"
EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"

# CIK → fund name for major whales
WHALES = {
    "0001067983": "Berkshire Hathaway (Buffett)",
    "0001336528": "Pershing Square (Ackman)",
    "0001336652": "Tiger Global (Chase Coleman)",
    "0000315066": "Soros Fund Management",
    "0001029160": "Appaloosa Management (Tepper)",
    "0001168164": "Third Point (Loeb)",
    "0001079114": "D.E. Shaw",
    "0001037389": "Two Sigma",
    "0001423298": "Elliott Management (Singer)",
    "0001582202": "ValueAct Capital",
    "0000102909": "Vanguard Group",
    "0000804328": "BlackRock",
}

# Filing types that signal new/increased positions
BULLISH_FORMS = {"13D", "13G", "SC 13D", "SC 13G", "SC 13D/A", "SC 13G/A"}
QUARTERLY_FORMS = {"13F-HR", "13F-HR/A"}

HEADERS = {"User-Agent": "ai-trader research@example.com",
           "Accept-Encoding": "gzip, deflate"}


def _load_cache():
    if not CACHE_FILE.exists():
        return {}
    try:
        data = json.loads(CACHE_FILE.read_text())
        if time.time() - data.get("_ts", 0) > CACHE_TTL:
            return {}
        return data
    except Exception:
        return {}


def _save_cache(data):
    data["_ts"] = time.time()
    try:
        CACHE_FILE.write_text(json.dumps(data, indent=2))
    except Exception:
        pass


def _search_edgar_filings(ticker, days_back=30, as_of=None):
    """
    Search EDGAR full-text for recent filings mentioning the ticker.
    Returns 13D/G (activist) and 13F (quarterly) filings.

    `as_of` (YYYY-MM-DD) pins the search window end for point-in-time backtests;
    default None = today (live behaviour, unchanged for existing callers).
    """
    filings = []
    end_dt = datetime.strptime(as_of, "%Y-%m-%d") if as_of else datetime.now()
    start = (end_dt - timedelta(days=days_back)).strftime("%Y-%m-%d")
    end   = end_dt.strftime("%Y-%m-%d")
    try:
        resp = requests.get(
            EDGAR_SEARCH,
            params={"q": f'"{ticker}"',
                    "forms": "SC 13D,SC 13G,13F-HR",
                    "dateRange": "custom",
                    "startdt": start, "enddt": end},
            headers=HEADERS, timeout=12,
        )
        if not resp.ok:
            return filings
        hits = resp.json().get("hits", {}).get("hits", [])
        for h in hits[:10]:
            src   = h.get("_source", {})
            form  = src.get("form_type", "") or src.get("file_type", "")
            entity = src.get("entity_name", src.get("display_names", ["?"])[0]
                             if isinstance(src.get("display_names"), list) else "?")
            filed = src.get("file_date", "")
            filings.append({
                "form":   form,
                "entity": str(entity)[:80],
                "filed":  filed,
                "ticker": ticker,
            })
    except Exception:
        pass
    return filings


def _recent_whale_filings(cik, whale_name, days_back=45):
    """
    Fetch recent filings for a specific whale CIK via EDGAR submissions API.
    Returns list of recent 13D/G/F filings.
    """
    results = []
    try:
        resp = requests.get(
            EDGAR_SUBMISSIONS.format(cik=cik.lstrip("0")),
            headers=HEADERS, timeout=10,
        )
        if not resp.ok:
            return results
        data   = resp.json()
        recent = data.get("filings", {}).get("recent", {})
        forms  = recent.get("form", [])
        dates  = recent.get("filingDate", [])
        descs  = recent.get("primaryDocument", [])
        cutoff = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        for form, date, doc in zip(forms, dates, descs):
            if date < cutoff:
                continue
            if form in BULLISH_FORMS or form in QUARTERLY_FORMS:
                results.append({
                    "whale":  whale_name,
                    "cik":    cik,
                    "form":   form,
                    "filed":  date,
                    "doc":    doc,
                    "is_activist": form in BULLISH_FORMS,
                })
    except Exception:
        pass
    return results


def get_whale_signals(watchlist_tickers=None, use_cache=True):
    """
    Returns list of signal dicts when major funds file on watchlist tickers.
    Also returns summary of recent whale filing activity.
    """
    cache = _load_cache() if use_cache else {}
    if cache.get("whale_signals"):
        return cache["whale_signals"]

    tickers   = watchlist_tickers or []
    all_sigs  = []

    # ── Track ticker-specific filings from any filer ──────────────────────────
    for sym in tickers:
        filings = _search_edgar_filings(sym, days_back=21)
        for f in filings:
            is_activist = f["form"] in BULLISH_FORMS
            score    = 0.85 if is_activist else 0.40
            urgency  = "HIGH" if is_activist else "MEDIUM"
            headline = (f"13D/G ACTIVIST: {f['entity']} filed {f['form']} on {sym} ({f['filed']})"
                        if is_activist
                        else f"13F: {f['entity']} reported {sym} position ({f['filed']})")
            all_sigs.append({
                "source":     "SEC-EDGAR-13F/13D",
                "tickers":    [sym],
                "event_type": "ACTIVIST_STAKE" if is_activist else "FUND_POSITION",
                "direction":  "BULLISH",
                "sentiment":  score,
                "urgency":    urgency,
                "headline":   headline[:200],
                "filer":      f["entity"],
                "form":       f["form"],
                "pub":        f["filed"],
            })
        time.sleep(0.3)

    # ── Scan known whales for any recent activist filings ─────────────────────
    for cik, name in list(WHALES.items())[:6]:   # limit to top 6 to stay fast
        filings = _recent_whale_filings(cik, name, days_back=21)
        for f in filings:
            if not f["is_activist"]:
                continue
            all_sigs.append({
                "source":     "SEC-EDGAR-WHALE",
                "tickers":    [],     # we don't know which ticker without parsing the doc
                "event_type": "WHALE_ACTIVIST",
                "direction":  "BULLISH",
                "sentiment":  0.75,
                "urgency":    "HIGH",
                "headline":   f"{name} filed {f['form']} ({f['filed']}) — check dashboard",
                "filer":      name,
                "form":       f["form"],
                "pub":        f["filed"],
            })
        time.sleep(0.2)

    all_sigs.sort(key=lambda x: (x["urgency"] == "HIGH", x["sentiment"]), reverse=True)
    cache["whale_signals"] = all_sigs
    _save_cache(cache)
    return all_sigs


def format_for_llm(signals, symbol):
    relevant = [s for s in signals
                if symbol in s.get("tickers", []) or not s.get("tickers")]
    if not relevant:
        return "No recent institutional 13F/13D filings."
    lines = []
    for s in relevant[:3]:
        lines.append(f"[{s['urgency']}] {s['event_type']} → {s['direction']}: "
                     f"{s['headline'][:140]}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    tickers = sys.argv[1:] or ["NVDA", "AAPL", "MSFT", "BTC", "ETH"]
    print(f"Whale tracker for {tickers}...\n")
    sigs = get_whale_signals(watchlist_tickers=tickers, use_cache=False)
    if not sigs:
        print("No recent institutional filings found.")
    for s in sigs[:15]:
        t = ",".join(s["tickers"]) or "general"
        print(f"  {s['urgency']:6} | {s['event_type']:18} | {t:8} | {s['headline'][:70]}")
