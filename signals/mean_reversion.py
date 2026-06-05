"""
mean_reversion.py — Two mean reversion strategies to complement momentum.

1. RSI reversion: When ADX < 20 (sideways market), buy oversold RSI, sell overbought.
   In trending markets (ADX > 25), momentum wins. In choppy markets, reversion wins.

2. Pairs spread: Track spread between correlated pairs.
   When Z-score > 2 → pair has diverged → trade to close the gap.
"""

import json
import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import date

DIR = Path(__file__).parent.parent
_SPREAD_FILE = DIR / "pairs_spread.json"

# Pairs to track: (symbol_A, symbol_B, yf_ticker_A, yf_ticker_B)
PAIRS = [
    ("RELIANCE", "HDFCBANK", "RELIANCE.NS", "HDFCBANK.NS"),
    ("TCS",      "INFY",     "TCS.NS",      "INFY.NS"),
    ("NVDA",     "MSFT",     "NVDA",        "MSFT"),
]

ZSCORE_ENTRY  = 2.0   # enter when spread is 2 std devs from mean
ZSCORE_EXIT   = 0.5   # exit when spread reverts to 0.5 std devs
ADX_CHOP_MAX  = 20    # RSI reversion only valid when ADX < this
RSI_OVERSOLD  = 35
RSI_OVERBOUGHT = 65


# ── RSI Mean Reversion Signal ─────────────────────────────────────────────────

def rsi_reversion_signal(ind: dict) -> dict | None:
    """
    Returns a reversion signal when market is choppy (ADX < 20) and RSI extreme.
    Only fires when momentum strategy would NOT fire (i.e. no squeeze release, no strong trend).

    Returns: {"action": "BUY"|"SELL", "reason": str, "strength": float} or None
    """
    adx = ind.get("adx", 25)
    rsi = ind.get("rsi14", 50)
    squeeze_released = ind.get("squeeze_released", False)

    # Don't fire if market is trending or squeeze just released (momentum takes priority)
    if adx >= ADX_CHOP_MAX or squeeze_released:
        return None

    if rsi <= RSI_OVERSOLD:
        strength = (RSI_OVERSOLD - rsi) / RSI_OVERSOLD  # 0-1, higher = more oversold
        return {
            "action": "BUY",
            "reason": f"RSI reversion: ADX={adx:.1f} (choppy) RSI={rsi:.1f} (oversold)",
            "strength": round(strength, 2),
        }

    if rsi >= RSI_OVERBOUGHT:
        strength = (rsi - RSI_OVERBOUGHT) / (100 - RSI_OVERBOUGHT)
        return {
            "action": "SELL",
            "reason": f"RSI reversion: ADX={adx:.1f} (choppy) RSI={rsi:.1f} (overbought)",
            "strength": round(strength, 2),
        }

    return None


# ── Pairs Spread ──────────────────────────────────────────────────────────────

def _load_spread_history() -> dict:
    if _SPREAD_FILE.exists():
        try:
            return json.loads(_SPREAD_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_spread_history(data: dict):
    _SPREAD_FILE.write_text(json.dumps(data, indent=2))


def compute_pair_zscore(ticker_a: str, ticker_b: str, lookback: int = 60) -> dict | None:
    """
    Compute the log-price spread Z-score between two assets.
    Z > 2: A is expensive vs B → expect convergence
    Z < -2: A is cheap vs B → expect convergence
    """
    try:
        data = yf.download([ticker_a, ticker_b], period=f"{lookback}d",
                           auto_adjust=True, progress=False)["Close"]
        if isinstance(data, pd.Series) or data.shape[1] < 2:
            return None
        data = data.dropna()
        if len(data) < 20:
            return None

        import numpy as np
        log_spread = (data[ticker_a] / data[ticker_b]).apply(lambda x: x if x > 0 else float('nan')).dropna()

        mean = float(log_spread.rolling(20).mean().iloc[-1])
        std  = float(log_spread.rolling(20).std().iloc[-1])
        current = float(log_spread.iloc[-1])

        if std == 0:
            return None

        zscore = (current - mean) / std
        return {
            "zscore":  round(zscore, 2),
            "spread":  round(current, 4),
            "mean":    round(mean, 4),
            "std":     round(std, 4),
            "date":    date.today().isoformat(),
        }
    except Exception:
        return None


def get_pairs_signals() -> list[dict]:
    """
    Check all pairs for divergence. Returns list of actionable signals.
    """
    signals = []
    history = _load_spread_history()

    for sym_a, sym_b, tick_a, tick_b in PAIRS:
        result = compute_pair_zscore(tick_a, tick_b)
        if result is None:
            continue

        z = result["zscore"]
        pair_key = f"{sym_a}/{sym_b}"
        history[pair_key] = result

        if z > ZSCORE_ENTRY:
            signals.append({
                "pair": pair_key,
                "sym_a": sym_a, "sym_b": sym_b,
                "action": f"SELL {sym_a} / BUY {sym_b}",
                "zscore": z,
                "reason": f"Spread Z={z:.1f} — {sym_a} expensive vs {sym_b}, expect reversion",
            })
        elif z < -ZSCORE_ENTRY:
            signals.append({
                "pair": pair_key,
                "sym_a": sym_a, "sym_b": sym_b,
                "action": f"BUY {sym_a} / SELL {sym_b}",
                "zscore": z,
                "reason": f"Spread Z={z:.1f} — {sym_a} cheap vs {sym_b}, expect reversion",
            })

    _save_spread_history(history)
    return signals


def format_pairs_for_llm(signals: list[dict]) -> str:
    if not signals:
        return "PAIRS: No divergence detected."
    lines = ["PAIRS REVERSION:"]
    for s in signals:
        lines.append(f"  {s['pair']}: Z={s['zscore']:.1f} → {s['action']}")
    return "\n".join(lines)
