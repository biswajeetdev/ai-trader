"""
news_catalyst.py — Corporate event & news signal monitor
Sources (all free, zero API keys):
  1. yfinance .news   — recent articles per ticker (JSON, always fresh)
  2. Google News RSS  — merger/acquisition keyword searches
  3. SEC EDGAR EFTS   — 8-K filings (material events: M&A, earnings, regulatory)

Signal format matches social_pulse.py for drop-in LLM context.
"""

import re, time, json, math, requests
import xml.etree.ElementTree as ET
import yfinance as yf
from datetime import datetime, timezone, timedelta
from pathlib import Path

CACHE_FILE = Path(__file__).parent.parent / "signals_cache.json"
CACHE_TTL  = 20 * 60   # 20 min — news is time-sensitive

# ── Event keyword → (sentiment score, event_type) ────────────────────────────
BULLISH_KW = {
    "merger":            (0.80, "MERGER"),
    "acquisition":       (0.75, "MERGER"),
    "acquired by":       (0.85, "MERGER"),
    "takeover":          (0.75, "MERGER"),
    "buyout":            (0.80, "MERGER"),
    "to be acquired":    (0.90, "MERGER"),
    "deal worth":        (0.60, "DEAL"),
    "partnership":       (0.40, "DEAL"),
    "joint venture":     (0.45, "DEAL"),
    "strategic alliance":(0.45, "DEAL"),
    "beats estimates":   (0.70, "EARNINGS_BEAT"),
    "earnings beat":     (0.70, "EARNINGS_BEAT"),
    "tops expectations": (0.65, "EARNINGS_BEAT"),
    "record revenue":    (0.65, "EARNINGS_BEAT"),
    "raised guidance":   (0.60, "EARNINGS_BEAT"),
    "upgrade":           (0.55, "ANALYST_UPGRADE"),
    "outperform":        (0.50, "ANALYST_UPGRADE"),
    "buy rating":        (0.55, "ANALYST_UPGRADE"),
    "price target raised":(0.50,"ANALYST_UPGRADE"),
    "fda approved":      (0.85, "REGULATORY"),
    "approved":          (0.40, "REGULATORY"),
    "buyback":           (0.50, "BUYBACK"),
    "dividend increase": (0.45, "DIVIDEND"),
    "stock split":       (0.35, "CORPORATE"),
    "investment":        (0.30, "DEAL"),
    # ── AI / contract catalysts (the "got an AI contract → spiked" class) ──
    # NOTE: a bullish hit here is annotated with chase-risk downstream so the
    # brain treats an already-spiked name as priced-in, not a fresh long.
    "ai contract":       (0.70, "AI_CONTRACT"),
    "ai partnership":    (0.60, "AI_CONTRACT"),
    "ai deal":           (0.60, "AI_CONTRACT"),
    "ai infrastructure": (0.55, "AI_CONTRACT"),
    "data center deal":  (0.55, "AI_CONTRACT"),
    "chip order":        (0.55, "AI_CONTRACT"),
    "awarded contract":  (0.65, "CONTRACT_WIN"),
    "wins contract":     (0.65, "CONTRACT_WIN"),
    "secures contract":  (0.65, "CONTRACT_WIN"),
    "government contract":(0.60, "CONTRACT_WIN"),
    "defense contract":  (0.60, "CONTRACT_WIN"),
    "multi-year deal":   (0.55, "CONTRACT_WIN"),
    "supply agreement":  (0.50, "CONTRACT_WIN"),
    "selected by":       (0.45, "CONTRACT_WIN"),
}
BEARISH_KW = {
    "miss estimates":    (-0.75, "EARNINGS_MISS"),
    "earnings miss":     (-0.80, "EARNINGS_MISS"),
    "below expectations":(-0.70, "EARNINGS_MISS"),
    "guidance cut":      (-0.70, "EARNINGS_MISS"),
    "lowered guidance":  (-0.65, "EARNINGS_MISS"),
    "downgrade":         (-0.60, "ANALYST_DOWNGRADE"),
    "sell rating":       (-0.65, "ANALYST_DOWNGRADE"),
    "price target cut":  (-0.55, "ANALYST_DOWNGRADE"),
    "underperform":      (-0.50, "ANALYST_DOWNGRADE"),
    "investigation":     (-0.70, "REGULATORY_RISK"),
    "antitrust":         (-0.75, "REGULATORY_RISK"),
    "regulatory block":  (-0.80, "REGULATORY_RISK"),
    "sec probe":         (-0.75, "REGULATORY_RISK"),
    "lawsuit":           (-0.55, "LEGAL"),
    "fine":              (-0.50, "LEGAL"),
    "penalty":           (-0.50, "LEGAL"),
    "data breach":       (-0.65, "LEGAL"),
    "recall":            (-0.60, "PRODUCT_RISK"),
    "layoffs":           (-0.40, "RESTRUCTURING"),
    "bankruptcy":        (-0.90, "BANKRUPTCY"),
    "insolvency":        (-0.90, "BANKRUPTCY"),
}

# Company name → ticker map for text extraction
COMPANY_MAP = {
    "nvidia": "NVDA", "apple": "AAPL", "microsoft": "MSFT",
    "bitcoin": "BTC", "ethereum": "ETH", "google": "GOOGL",
    "amazon": "AMZN", "meta": "META", "tesla": "TSLA",
    "netflix": "NFLX", "amd": "AMD", "intel": "INTC",
}

# Google News RSS search templates for M&A events
MA_QUERIES = [
    "{symbol} merger acquisition deal",
    "{symbol} acquired buyout takeover",
]

EDGAR_URL = "https://efts.sec.gov/LATEST/search-index"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _score_text(text):
    """Returns (sentiment, event_type, matched_kw) for article text."""
    t = text.lower()
    best_score = 0.0
    best_type  = "NEWS"
    best_kw    = ""
    for kw, (score, etype) in {**BULLISH_KW, **BEARISH_KW}.items():
        if kw in t and abs(score) > abs(best_score):
            best_score = score
            best_type  = etype
            best_kw    = kw
    return round(best_score, 2), best_type, best_kw


# ── Anti-chasing guard ────────────────────────────────────────────────────────
# A catalyst the market has already absorbed is not an opportunity — buying a
# name that's already up 300% makes you the exit liquidity. We only treat a
# bullish catalyst as actionable if the move hasn't already happened.
CHASE_WINDOW_PCT = 15.0   # already up >15% over the last week → likely priced in
CHASE_GAP_PCT    = 8.0    # or gapped >8% in the latest session → chasing

def _recent_move_pct(symbol, days=5):
    """How far a name has already run. Returns (pct_over_window, latest_gap_pct)
    or (None, None) if price data is unavailable."""
    try:
        tk   = f"{symbol}-USD" if symbol in ("BTC", "ETH") else symbol
        hist = yf.Ticker(tk).history(period=f"{days + 2}d")
        closes = hist["Close"].dropna().tolist()
        if len(closes) < 2:
            return None, None
        pct_window = (closes[-1] - closes[0])  / closes[0]  * 100
        gap_latest = (closes[-1] - closes[-2]) / closes[-2] * 100
        return round(pct_window, 1), round(gap_latest, 1)
    except Exception:
        return None, None


def _annotate_chase_risk(signal):
    """Tag a bullish catalyst with chase-risk so the LLM never initiates a long
    on a name that has already made its move. Additive — never drops the signal."""
    signal.setdefault("chase_risk", False)
    signal.setdefault("already_moved_pct", None)
    signal.setdefault("actionable", True)
    if signal["direction"] != "BULLISH" or signal["sentiment"] < 0.5:
        return signal
    tickers = signal.get("tickers", [])
    if len(tickers) != 1:        # ambiguous subject — skip the price check
        return signal
    mv_window, gap = _recent_move_pct(tickers[0])
    if mv_window is None:
        return signal
    signal["already_moved_pct"] = mv_window
    if mv_window >= CHASE_WINDOW_PCT or (gap is not None and gap >= CHASE_GAP_PCT):
        signal["chase_risk"] = True
        signal["actionable"] = False
    return signal


def _extract_tickers(text, base_tickers):
    """Pull tickers from text, merge with base list."""
    found = set(base_tickers)
    t = text.lower()
    for name, tick in COMPANY_MAP.items():
        if name in t:
            found.add(tick)
    found.update(re.findall(r'\$([A-Z]{2,5})\b', text))
    return list(found)


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


# ── Source 1: yfinance news ───────────────────────────────────────────────────

def _yf_news(symbol, max_age_hours=12):
    """Fetch recent news via yfinance (no API key, always free)."""
    signals = []
    try:
        ticker = f"{symbol}-USD" if symbol in ("BTC", "ETH") else symbol
        news   = yf.Ticker(ticker).news or []
        cutoff = time.time() - max_age_hours * 3600
        for item in news[:15]:
            if item.get("providerPublishTime", 0) < cutoff:
                continue
            title  = item.get("title", "")
            summ   = item.get("summary", "") or item.get("description", "")
            text   = f"{title} {summ}"
            score, etype, kw = _score_text(text)
            if abs(score) < 0.25:
                continue
            tickers = _extract_tickers(text, [symbol])
            signals.append({
                "source":     "yfinance-news",
                "headline":   title[:200],
                "event_type": etype,
                "tickers":    tickers,
                "sentiment":  score,
                "direction":  "BULLISH" if score > 0 else "BEARISH",
                "urgency":    "HIGH" if abs(score) >= 0.6 else "MEDIUM",
                "kw":         kw,
                "pub":        datetime.fromtimestamp(
                    item.get("providerPublishTime", 0), tz=timezone.utc
                ).strftime("%Y-%m-%d %H:%M"),
            })
    except Exception:
        pass
    return signals


# ── Source 2: Google News RSS ─────────────────────────────────────────────────

def _rss_news(symbol, max_age_hours=8):
    """Search Google News RSS for M&A + catalyst news."""
    signals = []
    for tpl in MA_QUERIES:
        query = tpl.format(symbol=symbol)
        url   = (f"https://news.google.com/rss/search"
                 f"?q={requests.utils.quote(query)}&hl=en-US&gl=US&ceid=US:en")
        try:
            resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
            if not resp.ok:
                continue
            root   = ET.fromstring(resp.content)
            cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
            for item in root.findall(".//item")[:8]:
                title  = item.findtext("title", "")
                desc   = item.findtext("description", "")
                pub    = item.findtext("pubDate", "")
                text   = f"{title} {desc}"
                try:
                    from email.utils import parsedate_to_datetime
                    if parsedate_to_datetime(pub).astimezone(timezone.utc) < cutoff:
                        continue
                except Exception:
                    pass
                score, etype, kw = _score_text(text)
                if abs(score) < 0.30:
                    continue
                tickers = _extract_tickers(text, [symbol])
                if symbol not in tickers:
                    continue
                signals.append({
                    "source":     "google-news",
                    "headline":   title[:200],
                    "event_type": etype,
                    "tickers":    tickers,
                    "sentiment":  score,
                    "direction":  "BULLISH" if score > 0 else "BEARISH",
                    "urgency":    "HIGH" if abs(score) >= 0.65 else "MEDIUM",
                    "kw":         kw,
                    "pub":        pub[:30],
                })
            time.sleep(0.3)
        except Exception:
            pass
    return signals


# ── Source 3: SEC EDGAR 8-K filings ──────────────────────────────────────────

def _edgar_8k(symbol, lookback_days=7, as_of=None):
    """
    Search SEC EDGAR full-text for recent 8-K filings mentioning the ticker.
    8-K = material events: M&A, earnings, exec changes, FDA, etc.
    Free — no API key. Rate limit ~10 req/s.

    `as_of` (YYYY-MM-DD) pins the window end for point-in-time backtests;
    default None = today (live behaviour, unchanged for existing callers).
    """
    signals = []
    try:
        end_dt = datetime.strptime(as_of, "%Y-%m-%d") if as_of else datetime.now()
        today = end_dt.strftime("%Y-%m-%d")
        start = (end_dt - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        resp  = requests.get(
            EDGAR_URL,
            params={"q": f'"{symbol}"', "forms": "8-K",
                    "dateRange": "custom", "startdt": start, "enddt": today},
            headers={"User-Agent": "ai-trader biswajeet@example.com"},
            timeout=10,
        )
        if not resp.ok:
            return signals
        hits = resp.json().get("hits", {}).get("hits", [])
        for h in hits[:5]:
            src    = h.get("_source", {})
            entity = src.get("entity_name", "Unknown")
            date   = src.get("file_date", "")
            desc   = f"SEC 8-K filing by {entity}"
            score, etype, kw = _score_text(desc)
            # 8-K is always material — treat as MEDIUM even without keyword match
            if score == 0.0:
                score = 0.3; etype = "SEC_FILING"
            signals.append({
                "source":     "SEC-EDGAR-8K",
                "headline":   f"8-K: {entity} ({date})"[:200],
                "event_type": etype,
                "tickers":    [symbol],
                "sentiment":  score,
                "direction":  "BULLISH" if score > 0 else "BEARISH",
                "urgency":    "HIGH" if etype in ("MERGER", "EARNINGS_BEAT", "EARNINGS_MISS") else "MEDIUM",
                "kw":         kw or "8-K",
                "pub":        date,
            })
    except Exception:
        pass
    return signals


# ── Public API ────────────────────────────────────────────────────────────────

def get_news_signals(watchlist_tickers=None, use_cache=True):
    """
    Fetch corporate event & news signals for all tickers.
    Returns list of signal dicts sorted by urgency + sentiment strength.
    """
    cache = _load_cache() if use_cache else {}
    if cache.get("news_signals"):
        return cache["news_signals"]

    tickers   = watchlist_tickers or []
    all_sigs  = []

    for sym in tickers:
        sigs = _yf_news(sym) + _rss_news(sym) + _edgar_8k(sym)
        all_sigs.extend(sigs)
        time.sleep(0.2)

    # Deduplicate by (symbol, event_type, direction) keeping highest sentiment
    seen   = {}
    unique = []
    for s in all_sigs:
        key = (tuple(sorted(s["tickers"])), s["event_type"], s["direction"])
        if key not in seen or abs(s["sentiment"]) > abs(seen[key]["sentiment"]):
            seen[key] = s
    unique = sorted(seen.values(),
                    key=lambda x: (x["urgency"] == "HIGH", abs(x["sentiment"])),
                    reverse=True)

    # Flag bullish catalysts the market has already priced in (anti-chasing).
    for s in unique:
        _annotate_chase_risk(s)

    cache["news_signals"] = unique
    _save_cache(cache)
    return unique


def format_for_llm(signals, symbol):
    """Format relevant signals for one symbol as LLM context string."""
    relevant = [s for s in signals if symbol in s.get("tickers", [])]
    if not relevant:
        return "No material news or corporate events in last 8–12h."
    lines = []
    for s in relevant[:4]:
        note = ""
        moved = s.get("already_moved_pct")
        if s.get("chase_risk"):
            note = (f"  ⚠ CHASE RISK: already {moved:+.1f}% over ~1wk — catalyst "
                    f"likely priced in; do NOT open a new long on this alone.")
        elif moved is not None:
            note = f"  (only {moved:+.1f}% over ~1wk — catalyst may still be early)"
        lines.append(
            f"[{s['urgency']}] {s['event_type']} via {s['source']} → "
            f"{s['direction']} (score {s['sentiment']:+.2f}): "
            f"\"{s['headline'][:130]}\" ({s['pub']}){note}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    tickers = sys.argv[1:] or ["NVDA", "AAPL", "MSFT", "BTC", "ETH"]
    print(f"Fetching news signals for {tickers}...\n")
    sigs = get_news_signals(watchlist_tickers=tickers, use_cache=False)
    if not sigs:
        print("No material events found.")
    for s in sigs[:20]:
        print(f"  {s['urgency']:6} | {s['event_type']:20} | {s['direction']:7} "
              f"| {','.join(s['tickers']):12} | {s['headline'][:80]}")
