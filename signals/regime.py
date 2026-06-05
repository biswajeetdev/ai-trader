"""
regime.py — Market Regime Detection + Economic Calendar Blackouts

Regime tiers (VIX-based):
  CALM     VIX < 15   → full size, trend-follow
  NORMAL   VIX 15–25  → standard strategy
  ELEVATED VIX 25–35  → reduce size 50%, prefer defensive
  CRISIS   VIX > 35   → cash only, no new longs

Economic calendar blackouts (±24h around):
  FOMC rate decisions (8×/year)
  CPI/PPI releases (monthly)
  NFP Jobs report (first Friday/month)
  GDP releases (quarterly)
"""

import json, time, requests
import yfinance as yf
from datetime import datetime, timezone, timedelta
from pathlib import Path

CACHE_FILE = Path(__file__).parent.parent / "signals_cache.json"

# Fed meeting dates 2026 (approximate — update annually)
FOMC_DATES_2026 = [
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-10",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
]

BLACKOUT_HOURS = 24   # hours before/after event to avoid trading


def _cache(key, val=None, ttl=1800):
    try:
        data = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
        if val is None:
            e = data.get(key)
            return e["data"] if e and time.time()-e.get("ts",0)<ttl else None
        data[key] = {"data": val, "ts": time.time()}
        CACHE_FILE.write_text(json.dumps(data, indent=2))
    except Exception:
        return None


# ── VIX Regime ────────────────────────────────────────────────────────────────

def get_regime():
    """Returns regime dict with tier, vix, size_multiplier, description."""
    cached = _cache("regime", ttl=900)
    if cached:
        return cached

    try:
        from datetime import timedelta as _td
        _end   = (datetime.now(timezone.utc) + _td(days=1)).strftime("%Y-%m-%d")
        _start = (datetime.now(timezone.utc) - _td(days=10)).strftime("%Y-%m-%d")
        df  = yf.download("^VIX", start=_start, end=_end,
                          interval="1d", progress=False, auto_adjust=True)
        vix = float(df["Close"].squeeze().iloc[-1]) if not df.empty else None
    except Exception:
        vix = None

    if vix is None:
        result = {"tier":"UNKNOWN","vix":None,"size_mult":0.75,
                  "description":"VIX unavailable — reducing size as precaution",
                  "allow_new_longs":True}
    elif vix < 15:
        result = {"tier":"CALM","vix":round(vix,2),"size_mult":1.0,
                  "description":"Low fear — full position size, trend-following favoured",
                  "allow_new_longs":True}
    elif vix < 25:
        result = {"tier":"NORMAL","vix":round(vix,2),"size_mult":1.0,
                  "description":"Normal volatility — standard strategy",
                  "allow_new_longs":True}
    elif vix < 35:
        result = {"tier":"ELEVATED","vix":round(vix,2),"size_mult":0.5,
                  "description":"Elevated fear — position size halved, avoid new longs",
                  "allow_new_longs":False}
    else:
        result = {"tier":"CRISIS","vix":round(vix,2),"size_mult":0.0,
                  "description":"CRISIS — cash only, no new positions",
                  "allow_new_longs":False}

    _cache("regime", result)
    return result


# ── BotScore Regime Classifier ───────────────────────────────────────────────

def get_market_regime_score() -> dict:
    """
    Classifies current market as Human-driven or Bot-driven.

    Bot-dominated markets (high-freq, algo-driven) show:
      - VIX < 15 (low fear = algos in control)
      - SPY 5d return > 0 (momentum-following bots)
      - Low intraday range (tight spreads)

    Human-dominated markets show:
      - VIX > 20 (fear = human panic selling)
      - High intraday volatility
      - Reversal patterns (mean reversion works better)

    Returns: {
        "regime": "BOT_DRIVEN" | "HUMAN_DRIVEN" | "MIXED",
        "score": 0-100,   # higher = more bot-driven
        "strategy_hint": "momentum" | "mean_reversion" | "neutral",
        "summary": "one-line string"
    }
    """
    _FALLBACK = {
        "regime": "MIXED",
        "score": 50,
        "strategy_hint": "neutral",
        "summary": "Regime data unavailable",
    }

    cached = _cache("botscore", ttl=900)
    if cached:
        return cached

    try:
        from datetime import timedelta as _td
        _end   = (datetime.now(timezone.utc) + _td(days=1)).strftime("%Y-%m-%d")
        _start = (datetime.now(timezone.utc) - _td(days=15)).strftime("%Y-%m-%d")

        vix_df = yf.download("^VIX", start=_start, end=_end,
                             interval="1d", progress=False, auto_adjust=True)
        spy_df = yf.download("SPY", start=_start, end=_end,
                             interval="1d", progress=False, auto_adjust=True)

        if vix_df.empty or spy_df.empty:
            return _FALLBACK

        vix = float(vix_df["Close"].squeeze().iloc[-1])

        # SPY 5-day return
        spy_close = spy_df["Close"].squeeze()
        if len(spy_close) < 6:
            return _FALLBACK
        spy_5d_ret = (spy_close.iloc[-1] / spy_close.iloc[-6] - 1) * 100

        # SPY average intraday range over last 5 days (high-low / close)
        spy_high  = spy_df["High"].squeeze()
        spy_low   = spy_df["Low"].squeeze()
        intraday_range_pct = ((spy_high - spy_low) / spy_close).iloc[-5:].mean() * 100

    except Exception:
        return _FALLBACK

    # Score each signal (higher = more bot-driven)
    score = 50  # neutral baseline

    # VIX contribution (±25 points)
    if vix < 13:
        score += 25
    elif vix < 15:
        score += 15
    elif vix < 20:
        score += 5
    elif vix < 25:
        score -= 10
    else:
        score -= 25

    # SPY momentum contribution (±15 points)
    if spy_5d_ret > 1.0:
        score += 15
    elif spy_5d_ret > 0:
        score += 8
    elif spy_5d_ret > -1.0:
        score -= 5
    else:
        score -= 15

    # Intraday range contribution (±10 points)
    # Low range (<0.6%) = algos running tight spreads = more bot-driven
    if intraday_range_pct < 0.6:
        score += 10
    elif intraday_range_pct < 1.0:
        score += 3
    elif intraday_range_pct > 1.5:
        score -= 10

    score = max(0, min(100, score))

    if score >= 65:
        regime        = "BOT_DRIVEN"
        strategy_hint = "momentum"
        summary       = (f"Bot-driven market (VIX {vix:.1f}, SPY 5d {spy_5d_ret:+.1f}%) "
                         f"— algos in control, momentum favoured")
    elif score <= 35:
        regime        = "HUMAN_DRIVEN"
        strategy_hint = "mean_reversion"
        summary       = (f"Human-driven market (VIX {vix:.1f}, SPY 5d {spy_5d_ret:+.1f}%) "
                         f"— fear/panic dynamics, mean-reversion favoured")
    else:
        regime        = "MIXED"
        strategy_hint = "neutral"
        summary       = (f"Mixed regime (VIX {vix:.1f}, SPY 5d {spy_5d_ret:+.1f}%) "
                         f"— no dominant driver")

    result = {
        "regime":        regime,
        "score":         round(score),
        "strategy_hint": strategy_hint,
        "summary":       summary,
    }
    _cache("botscore", result)
    return result


# ── Economic Calendar ─────────────────────────────────────────────────────────

def _is_blackout(event_dates, hours=BLACKOUT_HOURS):
    """True if now is within ±hours of any event date."""
    now    = datetime.now(timezone.utc)
    window = timedelta(hours=hours)
    for ds in event_dates:
        try:
            dt = datetime.strptime(ds, "%Y-%m-%d").replace(
                hour=18, minute=0, tzinfo=timezone.utc)   # events typically ~2 PM ET = 18:00 UTC
            if abs(now - dt) <= window:
                return True, ds
        except Exception:
            pass
    return False, None


def get_fomc_blackout():
    """True if within 24h of an FOMC meeting."""
    hit, date = _is_blackout(FOMC_DATES_2026)
    return hit, date


def get_economic_events_today():
    """
    Fetch today's high-impact economic events via Trading Economics RSS.
    Returns list of event names happening today ± 24h.
    Falls back to FOMC hardcoded calendar.
    """
    cached = _cache("econ_events", ttl=3600)
    if cached is not None:
        return cached

    events = []

    # Check hardcoded FOMC calendar
    fomc_hit, fomc_date = get_fomc_blackout()
    if fomc_hit:
        events.append({"name":"FOMC Rate Decision","date":fomc_date,"impact":"CRITICAL"})

    # Try economiccalendar.com or TradingEconomics RSS
    try:
        import xml.etree.ElementTree as ET
        url = "https://tradingeconomics.com/rss/news.aspx"
        r   = requests.get(url, timeout=8, headers={"User-Agent":"Mozilla/5.0"})
        if r.ok:
            root = ET.fromstring(r.content)
            now  = datetime.now(timezone.utc)
            for item in root.findall(".//item")[:20]:
                title = item.findtext("title","").lower()
                if any(kw in title for kw in ["fed","fomc","cpi","inflation","jobs","nfp","gdp","rate decision"]):
                    events.append({"name":item.findtext("title",""),"date":str(now.date()),"impact":"HIGH"})
    except Exception:
        pass

    _cache("econ_events", events)
    return events


def is_event_blackout():
    """
    Returns (True, event_name) if trading is risky due to macro event.
    Returns (False, None) otherwise.
    """
    fomc, date = get_fomc_blackout()
    if fomc:
        return True, f"FOMC Rate Decision ({date})"
    events = get_economic_events_today()
    critical = [e for e in events if e.get("impact") in ("CRITICAL","HIGH")]
    if critical:
        return True, critical[0]["name"]
    return False, None


# ── Combined regime context for LLM ──────────────────────────────────────────

def get_full_regime_context():
    """Returns regime dict enriched with economic calendar + FRED macro data."""
    regime = get_regime()
    blackout, event = is_event_blackout()

    regime["event_blackout"] = blackout
    regime["event_name"]     = event
    regime["trade_allowed"]  = regime["allow_new_longs"] and not blackout

    if blackout:
        regime["description"] += f" | EVENT BLACKOUT: {event} — no new positions"
    if not regime["allow_new_longs"] and not blackout:
        regime["description"] += " | Elevated/Crisis VIX — no new longs"

    # Enrich with FRED macro (yield curve, CPI, unemployment)
    try:
        from signals.openbb_macro import get_macro_enrichment
        macro_data = get_macro_enrichment()
        regime["macro"] = macro_data
        # Compound size multiplier: VIX-based × macro-based
        regime["size_mult"] = round(regime["size_mult"] * macro_data.get("macro_size_adj", 1.0), 2)
        if macro_data.get("recession_risk") in ("HIGH", "ELEVATED"):
            regime["description"] += f" | MACRO: {macro_data['macro_summary']}"
    except Exception:
        regime["macro"] = {}

    return regime


def format_for_llm(regime):
    tier   = regime.get("tier","?")
    vix    = regime.get("vix","?")
    mult   = regime.get("size_mult",1.0)
    event  = regime.get("event_name","")
    desc   = regime.get("description","")
    macro  = regime.get("macro",{})
    macro_str = f" | {macro['macro_summary']}" if macro.get("macro_summary") else ""
    return (f"REGIME: {tier} (VIX {vix}) | Size multiplier: {mult}× | "
            + (f"EVENT BLACKOUT: {event} | " if event else "")
            + desc[:120] + macro_str)


if __name__ == "__main__":
    r = get_full_regime_context()
    print(f"Regime  : {r['tier']} (VIX {r['vix']})")
    print(f"Size ×  : {r['size_mult']}")
    print(f"Trade?  : {r['trade_allowed']}")
    print(f"Blackout: {r['event_blackout']} — {r.get('event_name','')}")
    print(f"Desc    : {r['description']}")
