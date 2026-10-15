"""Short put screener — finds cash-secured put opportunities on US stocks.

Entry: VIX<25, RSI 40-65, above SMA50, no earnings within 8 days,
       OTM 4-9%, DTE 21-45, bid >= $0.25.
Exit:  50% profit or DTE <= 7 (gamma risk).
"""

import json
import yfinance as yf
from datetime import date, timedelta
from pathlib import Path

from broker.state_io import atomic_write_json

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
    atomic_write_json(POSITIONS_FILE, data)


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
            if dte < 0:
                # Already expired — the contract no longer trades, so a close
                # order can never fill. Flag for settlement instead, or it is
                # retried forever and permanently occupies a MAX_OPEN slot.
                to_close.append({**pos, "close_reason": f"EXPIRED {-dte}d ago",
                                 "current_premium": None, "expired": True})
                continue
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


def settle_expired(pos: dict) -> dict:
    """
    Settle a short put whose expiry has already passed.

    An expired contract cannot be closed by order, so the position is resolved
    from the underlying's close on the expiry date and removed from tracking:
      close >= strike -> expired worthless, keep the full credit
      close <  strike -> assigned: the credit is still kept, and shares are
                         acquired at the strike

    `pnl` is REALIZED P&L on the option only, and is the credit in both cases —
    assignment does not realise a loss, it converts the position into stock at
    a known basis. The paper loss on assignment lives in that stock position and
    is reported separately as `shares_acquired` / `cost_basis`, so the learning
    layer (update_from_trade_history, bootstrap_from_trade_history) is never
    taught that a short put "lost" money it did not lose.

    If the settlement price cannot be fetched the position is still released
    (the contract is gone either way) and flagged UNKNOWN for manual review —
    leaving it in place would occupy a MAX_OPEN slot forever.
    """
    symbol = pos["symbol"]
    strike = float(pos["strike"])
    qty    = int(pos.get("qty", 1))
    credit = float(pos.get("credit", pos.get("entry_premium", 0) * 100 * qty))

    outcome, settle_px, pnl = "UNKNOWN", None, None
    shares_acquired, cost_basis, paper_gap = 0, None, None
    try:
        exp = date.fromisoformat(pos["expiry"])
        df  = yf.download(symbol, start=exp.isoformat(),
                          end=(exp + timedelta(days=5)).isoformat(),
                          progress=False, auto_adjust=True)
        if not df.empty:
            settle_px = float(df["Close"].squeeze().iloc[0])
            if settle_px >= strike:
                outcome, pnl = "EXPIRED_WORTHLESS", round(credit, 2)
            else:
                # Assigned: credit is kept, shares acquired at the strike.
                # The mark-to-expiry shortfall is stock P&L, not option P&L.
                outcome, pnl = "ASSIGNED", round(credit, 2)
                shares_acquired = 100 * qty
                cost_basis = round(strike - credit / shares_acquired, 4)
                paper_gap = round((settle_px - strike) * shares_acquired, 2)
    except Exception:
        pass

    positions = load_positions()
    positions.pop(symbol, None)
    save_positions(positions)

    return {"symbol": symbol, "strike": strike, "expiry": pos["expiry"],
            "outcome": outcome, "settle_price": settle_px, "pnl": pnl,
            "credit": credit, "qty": qty, "shares_acquired": shares_acquired,
            "cost_basis": cost_basis, "paper_gap": paper_gap}
