"""signals/marketstack.py — Marketstack OHLCV fallback when yfinance rate-limits.

Free tier: 100 req/month (no key registration needed for basic endpoint).
Set MARKETSTACK_API_KEY env var for higher limits.

Usage:
    from signals.marketstack import get_ohlcv_df
    df = get_ohlcv_df("AAPL", days=30)  # same columns as yfinance

Returns None if unavailable so callers can fall back gracefully.
"""

import os
from datetime import datetime, timedelta, timezone
from functools import lru_cache

try:
    import requests as _req
    import pandas as _pd
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

_BASE    = "http://api.marketstack.com/v1"
_API_KEY = os.environ.get("MARKETSTACK_API_KEY", "")  # empty = limited free tier


def get_ohlcv_df(symbol: str, days: int = 30):
    """Return OHLCV DataFrame with columns matching yfinance (Open/High/Low/Close/Volume).
    Returns None on failure.
    """
    if not _AVAILABLE:
        return None
    if not _API_KEY:
        return None  # skip if no key — don't burn free quota silently
    try:
        end   = datetime.now(timezone.utc).date()
        start = end - timedelta(days=days + 10)  # buffer for weekends
        params = {
            "access_key": _API_KEY,
            "symbols":    symbol,
            "date_from":  start.isoformat(),
            "date_to":    end.isoformat(),
            "limit":      days + 10,
        }
        resp = _req.get(f"{_BASE}/eod", params=params, timeout=8)
        resp.raise_for_status()
        rows = resp.json().get("data", [])
        if not rows:
            return None
        df = _pd.DataFrame(rows)
        df["date"] = _pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df = df.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume",
        })
        return df[["Open", "High", "Low", "Close", "Volume"]].tail(days)
    except Exception:
        return None


def is_available() -> bool:
    return _AVAILABLE and bool(_API_KEY)
