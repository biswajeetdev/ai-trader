"""Short put screener — finds cash-secured put opportunities on US stocks.

Entry: VIX<25, RSI 40-65, above SMA50, no earnings within 8 days,
       OTM 4-9%, DTE 21-45, bid >= $0.25.
Exit:  50% profit or DTE <= 7 (gamma risk).
"""

import json
import yfinance as yf
from datetime import date
from pathlib import Path

POSITIONS_FILE = Path(__file__).parent.parent / "short_put_positions.json"

MAX_VIX          = 25
MIN_RSI          = 40
MAX_RSI          = 65
OTM_MIN_PCT      = 4
OTM_MAX_PCT      = 9
MIN_DTE          = 21
MAX_DTE          = 45
MIN_PREMIUM      = 0.25   # $25/contract minimum credit
MAX_OPEN         = 3      # max concurrent positions
PROFIT_CLOSE_PCT = 0.50   # buy back when 50% of premium decayed
GAMMA_DTE        = 7      # close at ≤7 DTE to avoid gamma risk


def load_positions() -> dict:
    if POSITIONS_FILE.exists():
        try:
            return json.loads(POSITIONS_FILE.read_text())
        except Exception:
            pass
    return {}


def save_positions(data: dict) -> None:
    POSITIONS_FILE.write_text(json.dumps(data, indent=2))


def find_short_put_opportunity(symbol: str, ind: dict, fund: dict,
                               vix: float | None) -> dict | None:
    """
    Screen symbol for short put entry.
    ind: output of compute_indicators(). fund: output of get_fundamentals().
    Returns opportunity dict or None.
    """
    if vix and vix > MAX_VIX:
        return None

    positions = load_positions()
    if symbol in positions:
        return None
    if len(positions) >= MAX_OPEN:
        return None

    rsi   = ind.get("rsi14", 50)
    price = ind.get("price", 0)
    sma50 = ind.get("sma50", price)

    if not (MIN_RSI <= rsi <= MAX_RSI):
        return None
    if price < sma50:
        return None

    days_earn = fund.get("days_to_earnings")
    if days_earn is not None and 0 < days_earn < 8:
        return None

    try:
        tk   = yf.Ticker(symbol)
        exps = tk.options
        if not exps:
            return None

        today = date.today()
        valid = [
            (e, (date.fromisoformat(e) - today).days)
            for e in exps
            if MIN_DTE <= (date.fromisoformat(e) - today).days <= MAX_DTE
        ]
        if not valid:
            return None

        exp_str, dte = min(valid, key=lambda x: abs(x[1] - 30))

        puts = tk.option_chain(exp_str).puts
        puts = puts[puts["bid"] > 0].copy()
        if puts.empty:
            return None

        lo         = price * (1 - OTM_MAX_PCT / 100)
        hi         = price * (1 - OTM_MIN_PCT / 100)
        candidates = puts[(puts["strike"] >= lo) & (puts["strike"] <= hi)]
        if candidates.empty:
            return None

        best    = candidates.loc[candidates["bid"].idxmax()]
        strike  = float(best["strike"])
        premium = round(float(best["bid"]), 2)

        if premium < MIN_PREMIUM:
            return None

        return {
            "symbol":   symbol,
            "price":    round(price, 2),
            "strike":   strike,
            "expiry":   exp_str,
            "dte":      dte,
            "premium":  premium,
            "otm_pct":  round((price - strike) / price * 100, 1),
            "credit":   round(premium * 100, 2),
            "max_risk": round((strike - premium) * 100, 2),
            "rsi":      round(rsi, 1),
        }
    except Exception:
        return None


def check_exits() -> list[dict]:
    """Return list of open positions that hit exit conditions."""
    positions = load_positions()
    to_close  = []
    today     = date.today()

    for symbol, pos in list(positions.items()):
        try:
            dte = (date.fromisoformat(pos["expiry"]) - today).days
            if dte <= GAMMA_DTE:
                to_close.append({**pos, "close_reason": f"DTE={dte}", "current_premium": None})
                continue

            puts = yf.Ticker(symbol).option_chain(pos["expiry"]).puts
            row  = puts[abs(puts["strike"] - pos["strike"]) < 0.01]
            if row.empty:
                continue

            ask     = float(row["ask"].iloc[0])
            current = ask if ask > 0 else float(row["lastPrice"].iloc[0])
            entry   = pos["entry_premium"]

            if current <= entry * (1 - PROFIT_CLOSE_PCT):
                pnl = round((entry - current) * pos.get("qty", 1) * 100, 2)
                to_close.append({**pos, "close_reason": f"50% profit +${pnl}",
                                 "current_premium": current})
        except Exception:
            pass

    return to_close
