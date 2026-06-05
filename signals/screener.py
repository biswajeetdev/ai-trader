"""
screener.py — Dynamic universe scanner for BB-squeeze + ADX breakout setups.

Called once per trader run. Scans US megacaps + Nifty50 leaders for:
  - BB squeeze releasing (bearish/bullish compression ending)
  - ADX > 20 (trending, not choppy)
  - Price above SMA50 (don't fight the trend)
  - Volume above 20-day average

Returns top N candidates as watchlist-format dicts:
  [{"symbol": "TSLA", "market": "us-stock"}, ...]
"""

import time
import logging
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor

log = logging.getLogger(__name__)

US_UNIVERSE = [
    "TSLA", "META", "GOOGL", "AMZN", "AMD", "CRM", "SHOP",
    "PLTR", "SMCI", "ARM", "MSTR", "COIN", "SOFI", "RBLX",
    "UBER", "LYFT", "SNOW", "NET", "DDOG", "ZS",
]

INDIA_UNIVERSE = [
    "INFY.NS", "WIPRO.NS", "BAJFINANCE.NS", "ICICIBANK.NS",
    "KOTAKBANK.NS", "LT.NS", "MARUTI.NS", "SUNPHARMA.NS",
    "ADANIPORTS.NS", "TITAN.NS",
]

_CACHE: dict = {}          # {"ts": float, "results": list[dict]}
_CACHE_TTL   = 30 * 60    # 30 minutes


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Compute ADX without external dependencies."""
    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)

    up   = high.diff()
    down = -low.diff()
    dm_pos = up.where((up > down) & (up > 0), 0.0)
    dm_neg = down.where((down > up) & (down > 0), 0.0)

    atr    = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    di_pos = 100 * dm_pos.ewm(span=period, min_periods=period, adjust=False).mean() / atr
    di_neg = 100 * dm_neg.ewm(span=period, min_periods=period, adjust=False).mean() / atr

    dx = (100 * (di_pos - di_neg).abs() / (di_pos + di_neg).replace(0, float("nan")))
    return dx.ewm(span=period, min_periods=period, adjust=False).mean()


def _score_symbol(symbol: str) -> float | None:
    """
    Returns squeeze score 0-100 or None if data unavailable.
    Higher = stronger squeeze-release + trend setup.
    """
    try:
        df = yf.download(symbol, period="60d", interval="1d", progress=False, auto_adjust=True)
        if df is None or len(df) < 55:
            return None

        close  = df["Close"].squeeze()
        high   = df["High"].squeeze()
        low    = df["Low"].squeeze()
        volume = df["Volume"].squeeze()

        # ── Bollinger Bands (25, 2σ) ──────────────────────────────────────────
        bb_mid = close.rolling(25).mean()
        bb_std = close.rolling(25).std()
        bb_up  = bb_mid + 2.0 * bb_std
        bb_lo  = bb_mid - 2.0 * bb_std

        # ── Keltner Channel (20, 1.5×ATR) ────────────────────────────────────
        tr     = pd.concat(
            [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
            axis=1,
        ).max(axis=1)
        kc_atr = tr.rolling(20).mean()
        kc_mid = close.rolling(20).mean()
        kc_up  = kc_mid + 1.5 * kc_atr
        kc_lo  = kc_mid - 1.5 * kc_atr

        sq_now  = bb_up.iloc[-1] < kc_up.iloc[-1] and bb_lo.iloc[-1] > kc_lo.iloc[-1]
        sq_prev = bb_up.iloc[-2] < kc_up.iloc[-2] and bb_lo.iloc[-2] > kc_lo.iloc[-2]
        squeeze_released = sq_prev and not sq_now

        # ── ADX ───────────────────────────────────────────────────────────────
        adx_series = _adx(high, low, close)
        adx_val    = adx_series.iloc[-1]

        # ── SMA50 ─────────────────────────────────────────────────────────────
        sma50         = close.rolling(50).mean().iloc[-1]
        price_above   = close.iloc[-1] > sma50

        # ── Liquidity check (20-day avg volume) ──────────────────────────────
        avg_volume = volume.rolling(20).mean().iloc[-1]
        if avg_volume < 500_000:
            return None   # illiquid — wide spreads hurt fills

        # ── Volume spike ─────────────────────────────────────────────────────
        vol_spike  = volume.iloc[-1] > 1.2 * avg_volume

        # ── Score ─────────────────────────────────────────────────────────────
        score = 0.0
        if squeeze_released:       score += 40
        if adx_val > 20:           score += 30
        if price_above:            score += 20
        if vol_spike:              score += 10
        if avg_volume >= 2_000_000: score += 10   # liquid — tighter spreads

        return score

    except Exception as exc:
        log.debug("screener skip %s: %s", symbol, exc)
        return None


def get_screener_candidates(
    top_n: int = 3,
    exclude: list[str] | None = None,
) -> list[dict]:
    """
    Scan universe, return top_n squeeze candidates as watchlist dicts.
    exclude: symbols already in the fixed watchlist (skip them).

    Returns list like:
        [{"symbol": "TSLA", "market": "us-stock"}, {"symbol": "INFY.NS", "market": "in-stock"}]
    """
    global _CACHE

    now = time.monotonic()
    if _CACHE and (now - _CACHE["ts"]) < _CACHE_TTL:
        log.debug("screener: returning cached results")
        return _CACHE["results"]

    exclude_set = {s.upper() for s in (exclude or [])}

    candidates = [
        (sym, "us-stock") for sym in US_UNIVERSE
        if sym.upper() not in exclude_set
    ] + [
        (sym, "in-stock") for sym in INDIA_UNIVERSE
        if sym.upper() not in exclude_set
    ]

    def _worker(args: tuple[str, str]) -> tuple[str, str, float] | None:
        sym, market = args
        score = _score_symbol(sym)
        if score is None or score == 0:
            return None
        return (sym, market, score)

    results: list[tuple[str, str, float]] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for item in pool.map(_worker, candidates):
            if item is not None:
                results.append(item)

    results.sort(key=lambda x: x[2], reverse=True)
    top = [{"symbol": sym, "market": market} for sym, market, _ in results[:top_n]]

    _CACHE = {"ts": now, "results": top}
    log.info("screener: top %d candidates — %s", len(top), [r["symbol"] for r in top])
    return top
