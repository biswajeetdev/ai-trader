"""dividend_capture.py — Buy before ex-dividend date, sell after.
Only trades when dividend yield > 0.5% of stock price (worth the trade).
Ex-div dates from yfinance tk.calendar (confirmed free API).
"""

import json
import time
import yfinance as yf
from datetime import datetime, timezone
from pathlib import Path

CACHE_FILE = Path(__file__).parent.parent / "signals_cache.json"
CACHE_TTL  = 6 * 3600   # 6 hours in seconds
MIN_YIELD  = 0.5         # minimum yield % to trigger a signal


def _cache(key, val=None, ttl=CACHE_TTL):
    try:
        data = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
        if val is None:
            entry = data.get(key)
            return entry["data"] if entry and time.time() - entry.get("ts", 0) < ttl else None
        data[key] = {"data": val, "ts": time.time()}
        CACHE_FILE.write_text(json.dumps(data, indent=2))
    except Exception:
        return None


def get_dividend_signal(symbol: str, market: str) -> dict:
    # Returns signal dict; only acts on us-stock market
    _neutral = {
        "signal": "NEUTRAL", "ex_div_date": None, "dividend_per_share": 0.0,
        "yield_pct": 0.0, "days_to_exdiv": None, "summary": "Not applicable",
    }

    if market != "us-stock":
        _neutral["summary"] = "Dividend capture only for us-stock"
        return _neutral

    cache_key = f"dividend_{symbol}"
    cached = _cache(cache_key)
    if cached:
        return cached

    try:
        tk   = yf.Ticker(symbol)
        cal  = tk.calendar or {}
        info = tk.info or {}

        # Ex-dividend date: prefer calendar, fall back to info timestamp
        ex_div_raw    = cal.get("Ex-Dividend Date") or info.get("exDividendDate")
        div_per_share = float(info.get("dividendRate") or info.get("lastDividendValue") or 0)
        current_price = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)

        if not ex_div_raw or current_price == 0:
            _neutral["summary"] = f"No ex-dividend data available for {symbol}"
            return _neutral

        # Normalise to date object — yfinance may return Timestamp, int (epoch), or str
        if hasattr(ex_div_raw, "date"):
            ex_div_dt = ex_div_raw.date()
        elif isinstance(ex_div_raw, (int, float)):
            # epoch seconds from info["exDividendDate"]
            ex_div_dt = datetime.utcfromtimestamp(ex_div_raw).date()
        else:
            ex_div_dt = datetime.strptime(str(ex_div_raw)[:10], "%Y-%m-%d").date()

        today      = datetime.now(timezone.utc).date()
        days_to    = (ex_div_dt - today).days
        yield_pct  = round((div_per_share / current_price) * 100, 3) if current_price else 0.0

        if 1 <= days_to <= 3 and yield_pct >= MIN_YIELD:
            result = {
                "signal":            "BUY_BEFORE_EXDIV",
                "ex_div_date":       str(ex_div_dt),
                "dividend_per_share": div_per_share,
                "yield_pct":         yield_pct,
                "days_to_exdiv":     days_to,
                "summary":           f"{symbol} ex-div in {days_to}d (${div_per_share:.2f}/sh, {yield_pct:.2f}%)",
            }
        elif days_to == -1:
            result = {
                "signal":            "SELL_AFTER_EXDIV",
                "ex_div_date":       str(ex_div_dt),
                "dividend_per_share": div_per_share,
                "yield_pct":         yield_pct,
                "days_to_exdiv":     days_to,
                "summary":           f"{symbol} ex-div was yesterday — capture complete, sell now",
            }
        else:
            result = {
                "signal":            "NEUTRAL",
                "ex_div_date":       str(ex_div_dt),
                "dividend_per_share": div_per_share,
                "yield_pct":         yield_pct,
                "days_to_exdiv":     days_to,
                "summary":           f"{symbol} ex-div in {days_to}d — outside 1-3d capture window",
            }

        _cache(cache_key, result)
        return result

    except Exception as e:
        _neutral["summary"] = f"Error fetching dividend data for {symbol}: {e}"
        return _neutral
