"""
stock_ranker.py — Cross-sectional ranking for the watchlist.

Synthesized from:
  - Auquan "Long-Short Strategies using Ranking": composite score ranks all stocks,
    go long top quintile, short bottom quintile.
  - Packt Ch4 "Absolute & Relative Series": relative performance vs SPY is the
    real signal — a stock can be technically bullish but a short candidate if it
    lags the market.

Score = equal-weight blend of:
  1. Relative momentum vs SPY (30d)        ← Packt Ch4
  2. Absolute momentum (20d price change)  ← Auquan
  3. RSI signal (40-60 = neutral best for short-put, <40 = short candidate)
  4. SMA50 trend score (+1 above, -1 below)

Returns a dict: {symbol: score} ranked highest to lowest.
Higher score = relative strength → prefer for LONG / short-put entries.
Lower score  = relative weakness  → flag as SHORT candidate.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


SPY_SYMBOL = "SPY"
_spy_cache: dict = {}        # {date_str: Series of closes}
_score_cache: dict = {}      # {date_str: {symbol: score}}
CACHE_MINUTES = 30


def _today_key() -> str:
    return datetime.now().strftime("%Y-%m-%d-%H")   # hourly cache key


def _get_spy_returns(period_days: int = 30) -> float | None:
    """SPY return over period_days."""
    try:
        df = yf.download(SPY_SYMBOL, period="3mo", interval="1d", progress=False)
        if df.empty or len(df) < period_days:
            return None
        close = df["Close"].squeeze()
        return float(close.iloc[-1] / close.iloc[-period_days] - 1)
    except Exception:
        return None


def _score_symbol(symbol: str, spy_return_30d: float | None) -> float | None:
    """
    Compute composite score for one symbol. Returns float or None on failure.
    """
    try:
        df = yf.download(symbol, period="3mo", interval="1d", progress=False)
        if df.empty or len(df) < 50:
            return None

        close = df["Close"].squeeze()
        price = float(close.iloc[-1])

        # ── Factor 1: Relative momentum vs SPY (Packt Ch4) ───────────────────
        abs_30d = float(close.iloc[-1] / close.iloc[-30] - 1)
        if spy_return_30d is not None:
            rel_mom = abs_30d - spy_return_30d   # excess return over SPY
        else:
            rel_mom = abs_30d

        # ── Factor 2: Absolute 20d momentum (Auquan) ─────────────────────────
        mom_20d = float(close.iloc[-1] / close.iloc[-20] - 1)

        # ── Factor 3: RSI score (neutral 40-60 = best for cash-secured puts) ──
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        ll    = float(loss.iloc[-1])
        rsi   = 100.0 if ll == 0 else float(100 - 100 / (1 + gain.iloc[-1] / ll))
        # Map RSI to [-1, +1]: 50 = 0, 70+ = +1 (strong), 30- = -1 (weak)
        rsi_score = (rsi - 50) / 50

        # ── Factor 4: Trend (above SMA50 = +1, below = -1) ───────────────────
        sma50      = float(close.rolling(50).mean().iloc[-1])
        trend_score = 1.0 if price > sma50 else -1.0

        # ── Composite (equal weight) ──────────────────────────────────────────
        score = 0.35 * rel_mom + 0.25 * mom_20d + 0.20 * rsi_score + 0.20 * trend_score
        return round(score, 4)

    except Exception:
        return None


def rank_watchlist(symbols_markets: list[dict], use_cache: bool = True) -> dict:
    """
    Rank all US stocks in the watchlist by composite score.

    Args:
        symbols_markets: list of {"symbol": ..., "market": ...} from config watchlist
        use_cache: skip recomputing within the same hour

    Returns:
        dict: {symbol: {"score": float, "rank": int, "signal": "LONG"|"SHORT"|"NEUTRAL"}}
        Sorted highest score first.
    """
    key = _today_key()
    if use_cache and key in _score_cache:
        return _score_cache[key]

    us_stocks = [i["symbol"] for i in symbols_markets if i.get("market") == "us-stock"]
    if not us_stocks:
        return {}

    spy_ret = _get_spy_returns(30)

    raw_scores: dict[str, float] = {}
    for sym in us_stocks:
        s = _score_symbol(sym, spy_ret)
        if s is not None:
            raw_scores[sym] = s

    if not raw_scores:
        return {}

    # Cross-sectional z-score normalization (Auquan approach)
    scores = pd.Series(raw_scores)
    mu, sigma = scores.mean(), scores.std()
    if sigma > 0:
        z = (scores - mu) / sigma
    else:
        z = scores

    # Assign quintile signal
    q80 = z.quantile(0.80)
    q20 = z.quantile(0.20)

    ranked = {}
    for rank, (sym, zval) in enumerate(z.sort_values(ascending=False).items(), 1):
        signal = "LONG" if zval >= q80 else ("SHORT" if zval <= q20 else "NEUTRAL")
        ranked[sym] = {
            "score":     round(float(z[sym]), 3),
            "raw_score": round(float(raw_scores[sym]), 4),
            "rank":      rank,
            "signal":    signal,
        }

    _score_cache[key] = ranked
    return ranked


def get_rank_context(symbol: str, ranked: dict) -> str:
    """Format rank info as a string for LLM context injection."""
    if not ranked or symbol not in ranked:
        return ""
    r = ranked[symbol]
    total = len(ranked)
    return (f"RANK: #{r['rank']}/{total} (z={r['score']:+.2f}) "
            f"signal={r['signal']} | "
            f"top quintile=LONG candidate, bottom quintile=SHORT candidate")
