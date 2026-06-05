"""
earnings_calendar.py — Earnings date monitor + surprise tracker
Sources: yfinance (free) + Yahoo Finance RSS
Flags: upcoming earnings, recent surprise beats/misses
"""

import json, time, requests
import yfinance as yf
from datetime import datetime, timezone, timedelta
from pathlib import Path

CACHE_FILE = Path(__file__).parent.parent / "signals_cache.json"
BLACKOUT_DAYS = 5   # HOLD within this many days of earnings


def _cache(key, value=None, ttl=3600):
    try:
        data = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
        if value is None:
            entry = data.get(key)
            return entry["data"] if entry and time.time()-entry.get("ts",0)<ttl else None
        data[key] = {"data": value, "ts": time.time()}
        CACHE_FILE.write_text(json.dumps(data, indent=2))
    except Exception:
        return None


def get_earnings_info(symbol, market):
    """Returns earnings data for a stock symbol."""
    if market != "us-stock":
        return {"symbol": symbol, "market": market, "days_to_earnings": None}

    cached = _cache(f"earn_{symbol}")
    if cached:
        return cached

    try:
        tk   = yf.Ticker(symbol)
        info = tk.info

        # Earnings date
        ts  = info.get("earningsTimestamp") or info.get("earningsDate")
        dte = None
        if ts:
            dt  = datetime.fromtimestamp(ts, tz=timezone.utc) if isinstance(ts,(int,float)) else ts
            dte = (dt - datetime.now(tz=timezone.utc)).days

        # EPS surprise from last quarter
        cal  = tk.calendar
        eps_est   = None
        eps_actual= None
        surprise_pct = None
        if cal is not None and not cal.empty:
            try:
                eps_est    = float(cal.get("Earnings Estimate", [None])[0] or 0)
                eps_actual = float(info.get("trailingEps", 0) or 0)
                if eps_est and eps_est != 0:
                    surprise_pct = round((eps_actual - eps_est) / abs(eps_est) * 100, 1)
            except Exception:
                pass

        earn_str = (f"in {dte}d" if dte and dte > 0 else
                    f"{abs(dte)}d ago" if dte is not None else "N/A")
        result = {
            "symbol":           symbol,
            "days_to_earnings": dte,
            "blackout":         dte is not None and abs(dte) <= BLACKOUT_DAYS,
            "eps_estimate":     eps_est,
            "eps_actual":       eps_actual,
            "eps_surprise_pct": surprise_pct,
            "revenue_growth":   info.get("revenueGrowth"),
            "next_earnings_str": earn_str,
        }
        _cache(f"earn_{symbol}", result)
        return result

    except Exception:
        return {"symbol": symbol, "days_to_earnings": None, "blackout": False}


def earnings_signal(info):
    """
    Returns (signal_str, confidence_delta) based on earnings context.
    Positive delta = boost confidence. Negative = reduce.
    """
    dte     = info.get("days_to_earnings")
    surp    = info.get("eps_surprise_pct")
    signal  = []
    delta   = 0

    if dte is not None:
        if abs(dte) <= BLACKOUT_DAYS:
            signal.append(f"BLACKOUT (earnings {'in' if dte>0 else ''} {abs(dte)}d)")
            delta -= 50   # force HOLD
        elif 0 < dte <= 14:
            signal.append(f"Earnings in {dte}d — caution")
            delta -= 15

    if surp is not None:
        if surp > 10:
            signal.append(f"Beat EPS by {surp}% last Q → post-earnings drift UP")
            delta += 10
        elif surp > 0:
            signal.append(f"Slight EPS beat {surp}%")
            delta += 5
        elif surp < -10:
            signal.append(f"Missed EPS by {abs(surp)}% last Q → drift DOWN risk")
            delta -= 15
        elif surp < 0:
            signal.append(f"Slight EPS miss {surp}%")
            delta -= 5

    return " | ".join(signal) if signal else "No earnings signal", delta


def get_all_earnings(watchlist_symbols, markets):
    results = {}
    for sym, mkt in zip(watchlist_symbols, markets):
        results[sym] = get_earnings_info(sym, mkt)
    return results


def format_for_llm(info):
    signal_str, delta = earnings_signal(info)
    return (f"Earnings: {info.get('next_earnings_str','N/A')} | "
            f"Last EPS surprise: {info.get('eps_surprise_pct','N/A')}% | "
            f"Signal: {signal_str}")


if __name__ == "__main__":
    symbols = [("NVDA","us-stock"),("AAPL","us-stock"),("MSFT","us-stock")]
    for sym, mkt in symbols:
        info = get_earnings_info(sym, mkt)
        sig, delta = earnings_signal(info)
        print(f"{sym}: {info['next_earnings_str']:12} | EPS surprise: {info.get('eps_surprise_pct','N/A')}% | {sig} (Δ{delta:+d})")
