"""
risk.py — Position sizing, stop-loss, profit targets, circuit breaker, trade history.
All stateful risk logic lives here so trader.py stays orchestration-only.
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from broker.ai4trade import get_positions_api

DIR        = Path(__file__).parent.parent
POSITIONS  = DIR / "positions.json"
TRADE_HIST = DIR / "trade_history.json"
HWM_FILE   = DIR / "portfolio_hwm.json"

# ── Tuned parameters (vbt walk-forward validated) ────────────────────────────
STOP_LOSS_ATR      = 3.0
PROFIT_TARGET_ATR  = 7.5
PARTIAL_TARGET_ATR = PROFIT_TARGET_ATR / 2   # 3.75×ATR → take 50% off, move stop to breakeven
RECENT_TRADES      = 5
DRAWDOWN_HALT_PCT  = 0.10   # halt new BUYs if portfolio drops >10% from peak
MIN_CONFIDENCE     = 70
MAX_TRADE_USD      = 5_000


# ── Position store ────────────────────────────────────────────────────────────

def load_positions() -> dict:
    return json.loads(POSITIONS.read_text()) if POSITIONS.exists() else {}


def save_positions(pos: dict) -> None:
    POSITIONS.write_text(json.dumps(pos, indent=2))


def record_open(symbol: str, price: float, qty: float, atr: float, market: str) -> None:
    pos = load_positions()
    pos[symbol] = {
        "action":       "BUY",
        "entry_price":  price,
        "quantity":     qty,
        "atr":          atr,
        "atr_at_entry": atr,
        "market":       market,
        "entry_date":   datetime.now().strftime("%Y-%m-%d"),
        "stop_price":   round(price - STOP_LOSS_ATR  * atr, 4),
        "target_price": round(price + PROFIT_TARGET_ATR * atr, 4),
        "trail_stop":   round(price - STOP_LOSS_ATR  * atr, 4),
        "partial_done": False,
    }
    save_positions(pos)


def record_close(symbol: str, exit_price: float) -> None:
    pos   = load_positions()
    entry = pos.pop(symbol, {})
    save_positions(pos)

    if not entry:
        return
    hist = json.loads(TRADE_HIST.read_text()) if TRADE_HIST.exists() else []
    qty  = entry.get("quantity", 0)
    pnl  = round((exit_price - entry["entry_price"]) * qty, 2)
    hist.append({
        "symbol":     symbol,
        "market":     entry.get("market", ""),
        "entry":      entry["entry_price"],
        "exit":       exit_price,
        "qty":        qty,
        "pnl":        pnl,
        "pnl_pct":    round((exit_price / entry["entry_price"] - 1) * 100, 2),
        "outcome":    "WIN" if pnl > 0 else "LOSS",
        "entry_date": entry.get("entry_date", ""),
        "exit_date":  datetime.now().strftime("%Y-%m-%d"),
    })
    if len(hist) > 50:
        hist = hist[-50:]
    TRADE_HIST.write_text(json.dumps(hist, indent=2))


def check_stops(current_price: float, symbol: str) -> str | None:
    """Returns 'STOP_LOSS', 'PROFIT_TARGET', 'PARTIAL_PROFIT', 'TRAIL_STOP', or None."""
    positions = load_positions()
    pos = positions.get(symbol)
    if not pos or pos["action"] != "BUY":
        return None
    if current_price <= pos["stop_price"]:
        return "STOP_LOSS"
    if current_price >= pos["target_price"]:
        return "PROFIT_TARGET"
    # Partial exit: 50% off at halfway to target, then move stop to breakeven
    entry = pos["entry_price"]
    atr   = pos.get("atr", 0)
    if not pos.get("partial_done") and atr > 0:
        if current_price >= entry + PARTIAL_TARGET_ATR * atr:
            return "PARTIAL_PROFIT"
    # Chandelier trailing stop: init if unset, then ratchet upward only
    if pos.get("trail_stop") is None:
        pos["trail_stop"] = round(chandelier_exit([current_price], atr), 4)
    new_trail = chandelier_exit([current_price], atr)
    if new_trail > pos["trail_stop"]:
        pos["trail_stop"] = round(new_trail, 4)
    positions[symbol] = pos
    save_positions(positions)
    if current_price <= pos["trail_stop"]:
        return "TRAIL_STOP"
    return None


def mark_partial_done(symbol: str) -> None:
    """After partial exit: flag as done, move stop_price to entry (breakeven)."""
    positions = load_positions()
    pos = positions.get(symbol)
    if not pos:
        return
    pos["partial_done"] = True
    pos["stop_price"]   = pos["entry_price"]   # ride the rest risk-free
    positions[symbol]   = pos
    save_positions(positions)


def recent_trade_context() -> str:
    """Last N completed trades formatted for LLM feedback."""
    if not TRADE_HIST.exists():
        return "No completed trades yet."
    hist = json.loads(TRADE_HIST.read_text())[-RECENT_TRADES:]
    if not hist:
        return "No completed trades yet."
    return " | ".join(
        f"{t['symbol']} {t['outcome']} {t['pnl_pct']:+.1f}% ({t['entry_date']}→{t['exit_date']})"
        for t in hist
    )


# ── Drawdown circuit breaker ──────────────────────────────────────────────────

def load_hwm() -> dict:
    if HWM_FILE.exists():
        try:
            return json.loads(HWM_FILE.read_text())
        except Exception:
            pass
    return {"peak": 0.0, "halted": False, "halted_since": None}


def save_hwm(hwm: dict) -> None:
    HWM_FILE.write_text(json.dumps(hwm, indent=2))


def _portfolio_value(token: str, cash: float) -> float:
    """Cash + mark-to-market of open positions from ai4trade.ai."""
    try:
        data      = get_positions_api(token)
        positions = data.get("positions", [])
        holdings  = sum(
            float(p.get("quantity", 0)) * float(p.get("current_price") or p.get("price", 0))
            for p in positions
        )
        return cash + holdings
    except Exception:
        return cash   # conservative fallback


def check_drawdown_circuit(token: str, cash: float) -> tuple[bool, float, float, float]:
    """
    Update the high-water mark and check the 10% drawdown circuit breaker.
    Persists state across launchd runs via portfolio_hwm.json.

    Returns: (halted, current_value, drawdown_pct, peak)
    """
    hwm     = load_hwm()
    current = _portfolio_value(token, cash)

    if current > hwm["peak"]:
        hwm["peak"] = current

    peak         = hwm["peak"]
    drawdown_pct = (current - peak) / peak if peak > 0 else 0.0

    # Rolling equity history for dynamic position sizing (Packt Ch8)
    eq_hist = hwm.get("equity_history", [])
    eq_hist.append(round(current, 2))
    hwm["equity_history"] = eq_hist[-30:]   # keep last 30 runs (~15h at 30-min cron)

    if drawdown_pct <= -DRAWDOWN_HALT_PCT:
        if not hwm["halted"]:
            hwm["halted"]       = True
            hwm["halted_since"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        save_hwm(hwm)
        return True, current, drawdown_pct, peak

    if hwm["halted"]:
        hwm["halted"]       = False
        hwm["halted_since"] = None
    save_hwm(hwm)
    return False, current, drawdown_pct, peak


def get_equity_history() -> list[float]:
    """Return stored equity curve for dynamic sizing (populated by check_drawdown_circuit)."""
    hwm = load_hwm()
    return hwm.get("equity_history", [])


def vol_target_scalar(equity_history: list[float], target_vol: float = 0.12) -> float:
    # Returns vol-target scalar [0.25, 2.0]; 1.0 if <21 obs or zero vol
    if len(equity_history) < 21: return 1.0
    ret = np.diff(equity_history) / np.array(equity_history[:-1])
    rv = np.std(ret[-20:]) * np.sqrt(252)
    return float(np.clip(target_vol / rv, 0.25, 2.0)) if rv > 0 else 1.0

def chandelier_exit(high_series: list[float], atr: float, multiplier: float = 3.0) -> float:
    return max(high_series) - multiplier * atr  # highest-high trailing stop

def drawdown_radar_score(equity_history: list[float], vix: float = 20.0) -> int:
    # Composite danger score 0-100; block new trades if >60
    if len(equity_history) < 20:
        return 0
    ret = np.diff(equity_history) / np.array(equity_history[:-1])
    v20 = float(np.std(ret[-20:]) * 252**0.5)
    score  = 25 if (v20 > 0 and float(np.std(ret[-5:]) * 252**0.5) > 1.5 * v20) else 0
    score += 25 if vix > 30 else (12 if vix > 20 else 0)
    sharpe = float(np.mean(ret[-20:]) * 252 / v20) if v20 > 0 else 0.0
    score += 34 if sharpe < 0 else (20 if sharpe < 0.5 else 0)
    return int(np.clip(score, 0, 100))


# ── Position sizing ───────────────────────────────────────────────────────────

def _fixed_risk_size(
    price: float, atr: float, cash: float, confidence: int, market: str
) -> float:
    """Fixed 1% portfolio risk per trade, scaled by confidence. Internal helper."""
    risk  = cash * 0.01
    stop  = atr * STOP_LOSS_ATR
    if stop <= 0 or price <= 0:
        return 0
    qty   = min(risk / stop, MAX_TRADE_USD / price)
    scale = (confidence - MIN_CONFIDENCE) / (100 - MIN_CONFIDENCE)
    qty  *= max(0.3, min(1.0, scale))
    return max(1, int(qty)) if market != "crypto" else round(qty, 6)


def kelly_position_size(price: float, atr: float, cash: float, confidence: int, market: str) -> float:
    """
    Kelly criterion position size. Uses half-Kelly for safety.
    Falls back to _fixed_risk_size() when trade history is too short or degenerate.

    Returns quantity (shares for stocks, units for crypto).
    """
    if not TRADE_HIST.exists():
        return _fixed_risk_size(price, atr, cash, confidence, market)

    try:
        hist = json.loads(TRADE_HIST.read_text())
    except Exception:
        return _fixed_risk_size(price, atr, cash, confidence, market)

    if len(hist) < 5:
        return _fixed_risk_size(price, atr, cash, confidence, market)

    recent = hist[-20:]
    wins_pct   = [t.get("pnl_pct", 0) / 100 for t in recent if t.get("pnl", 0) > 0]
    losses_pct = [t.get("pnl_pct", 0) / 100 for t in recent if t.get("pnl", 0) <= 0]

    avg_win  = sum(wins_pct)   / len(wins_pct)   if wins_pct   else 0.0
    avg_loss = sum(losses_pct) / len(losses_pct) if losses_pct else 0.0

    # avg_loss is negative (losses); kelly() uses abs(avg_loss) internally
    if avg_loss == 0:
        return _fixed_risk_size(price, atr, cash, confidence, market)

    sample   = len(recent)
    wins     = sum(1 for t in recent if t.get("pnl", 0) > 0)
    win_rate = wins / sample   # fraction in [0, 1]

    raw_kelly  = kelly(win_rate, avg_win, avg_loss)
    half_kelly = raw_kelly / 2
    fraction   = max(0.01, min(0.25, half_kelly))   # clamp: 1%–25%

    stop      = atr * STOP_LOSS_ATR
    if stop <= 0 or price <= 0:
        return _fixed_risk_size(price, atr, cash, confidence, market)

    dollar_risk = cash * fraction
    qty         = dollar_risk / stop
    return max(1, int(qty)) if market != "crypto" else round(qty, 6)


def size_position(
    price: float, atr: float, cash: float, confidence: int, market: str
) -> float:
    """Position size with automatic Kelly upgrade and volatility targeting."""
    qty = _fixed_risk_size(price, atr, cash, confidence, market)
    if TRADE_HIST.exists():
        try:
            hist = json.loads(TRADE_HIST.read_text())
            if len(hist) >= 5:
                qty = kelly_position_size(price, atr, cash, confidence, market)
        except Exception:
            pass
    qty *= vol_target_scalar(get_equity_history())
    return max(1, int(qty)) if market != "crypto" else round(qty, 6)


# ── Trading edge formulas (Packt Ch6) ────────────────────────────────────────
# Source: "The Trading Edge is a Number, and Here is the Formula"

def expectancy(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Arithmetic expectancy per trade. win_rate in [0,1], avg_win/loss as fractions."""
    return win_rate * avg_win + (1 - win_rate) * avg_loss


def george(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Geometric gain expectancy (compounds correctly). Positive = strategy has edge."""
    return (1 + avg_win) ** win_rate * (1 + avg_loss) ** (1 - win_rate) - 1


def kelly(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Kelly fraction: optimal % of capital per trade. Half-Kelly = kelly()/2 in practice."""
    if avg_loss == 0 or avg_win == 0:
        return 0.0
    return win_rate / abs(avg_loss) - (1 - win_rate) / avg_win


def get_win_rate(n: int = 20) -> dict:
    """
    Compute win rate from last n closed trades in trade_history.json.
    A trade is a WIN if pnl > 0, LOSS if pnl <= 0.
    Returns dict with win_rate (0-100), sample_size, avg_pnl, streak.
    """
    if not TRADE_HIST.exists():
        return {"win_rate": None, "sample_size": 0, "avg_pnl": None, "streak": 0,
                "summary": "Insufficient history"}
    try:
        hist = json.loads(TRADE_HIST.read_text())
    except Exception:
        return {"win_rate": None, "sample_size": 0, "avg_pnl": None, "streak": 0,
                "summary": "Insufficient history"}

    if len(hist) < 3:
        return {"win_rate": None, "sample_size": len(hist), "avg_pnl": None, "streak": 0,
                "summary": "Insufficient history"}

    recent = hist[-n:]
    sample = len(recent)
    wins   = sum(1 for t in recent if t.get("pnl", 0) > 0)
    pnls   = [t.get("pnl", 0) for t in recent]
    avg_pnl = round(sum(pnls) / sample, 2)
    win_rate = round(wins / sample * 100, 1)

    # Current streak: +N = consecutive wins, -N = consecutive losses
    streak = 0
    for t in reversed(recent):
        is_win = t.get("pnl", 0) > 0
        if streak == 0:
            streak = 1 if is_win else -1
        elif streak > 0 and is_win:
            streak += 1
        elif streak < 0 and not is_win:
            streak -= 1
        else:
            break

    # Trading edge: expectancy & geometric edge (Packt Ch6)
    wins_pct   = [t.get("pnl_pct", 0) / 100 for t in recent if t.get("pnl", 0) > 0]
    losses_pct = [t.get("pnl_pct", 0) / 100 for t in recent if t.get("pnl", 0) <= 0]
    avg_win_f  = sum(wins_pct)  / len(wins_pct)  if wins_pct  else 0.0
    avg_loss_f = sum(losses_pct)/ len(losses_pct) if losses_pct else 0.0
    wr_f       = wins / sample
    exp        = round(expectancy(wr_f, avg_win_f, avg_loss_f) * 100, 2)
    geo        = round(george(wr_f, avg_win_f, avg_loss_f) * 100, 2)

    pnl_sign = "+" if avg_pnl >= 0 else ""
    streak_sign = "+" if streak > 0 else ""
    exp_sign = "+" if exp >= 0 else ""
    summary = (
        f"Win rate {win_rate:.0f}% ({wins}/{sample} trades) | "
        f"Avg P&L {pnl_sign}${avg_pnl} | "
        f"Streak: {streak_sign}{streak} | "
        f"Expectancy: {exp_sign}{exp}% | Geo-edge: {geo:+.2f}%"
    )

    return {
        "win_rate":    win_rate,
        "sample_size": sample,
        "avg_pnl":     avg_pnl,
        "streak":      streak,
        "expectancy":  exp,
        "geo_edge":    geo,
        "summary":     summary,
    }


# ── Dynamic risk appetite (Packt Ch8 "Position Sizing") ──────────────────────
# Source: Algorithmic Short-Selling with Python, Chapter 8
# Reduces position size convexly as drawdown approaches tolerance.

def risk_appetite(equity_curve: list[float],
                  tolerance: float = -0.10,
                  mn: float = 0.005,
                  mx: float = 0.02,
                  span: int = 5,
                  shape: int = 1) -> float:
    """
    Dynamic risk per trade based on drawdown from peak equity.

    Args:
        equity_curve: list of portfolio values over time
        tolerance:    max acceptable drawdown (e.g. -0.10 = -10%)
        mn:           minimum risk fraction when at tolerance (e.g. 0.005 = 0.5%)
        mx:           maximum risk fraction at new equity highs (e.g. 0.02 = 2%)
        span:         EMA span to smooth the appetite curve
        shape:        1=convex (cut risk fast), -1=concave (cut risk slow), 0=linear

    Returns:
        Current risk fraction (between mn and mx) to apply to position sizing.
    """
    eqty = pd.Series(equity_curve)
    watermark = eqty.expanding().max()
    drawdown  = eqty / watermark - 1
    ddr       = 1 - np.minimum(drawdown / tolerance, 1)   # 0=at tolerance, 1=at peak
    avg_ddr   = ddr.ewm(span=span).mean()

    if shape == 1:
        _power = mx / mn      # convex — risk drops fast near tolerance
    elif shape == -1:
        _power = mn / mx      # concave — risk drops slowly
    else:
        _power = 1.0          # linear

    appetite = mn + (mx - mn) * (avg_ddr ** _power)
    return round(float(appetite.iloc[-1]), 5)


def dynamic_position_size(cash: float, price: float, atr: float,
                          equity_history: list[float]) -> int:
    """
    ATR-based position size scaled by dynamic risk appetite.
    Replaces fixed MAX_TRADE_USD with equity-curve-aware sizing.

    Risk per trade = risk_appetite × peak_equity
    Shares         = risk_per_trade / (STOP_LOSS_ATR × ATR)
    """
    if not equity_history or price <= 0 or atr <= 0:
        return max(1, int(MAX_TRADE_USD / price))

    peak    = max(equity_history)
    risk_f  = risk_appetite(equity_history)
    risk_usd = peak * risk_f                         # dollars at risk per trade
    shares  = int(risk_usd / (STOP_LOSS_ATR * atr)) # units sized to 1 ATR stop
    shares = max(1, min(shares, int(MAX_TRADE_USD / price)))
    return shares


# ── Correlation guard ─────────────────────────────────────────────────────────

def get_correlated_symbols(symbol: str, positions: dict, threshold: float = 0.7) -> list[str]:
    """
    Return list of open position symbols correlated > threshold with the given symbol.
    Uses 60-day daily closes from yfinance. Returns [] if data unavailable.
    """
    if not positions:
        return []
    try:
        import yfinance as yf
        open_syms = list(positions.keys())
        all_syms = [symbol] + open_syms
        # Map bare symbols to yf tickers
        tickers = []
        for s in all_syms:
            if s in ("BTC", "ETH"):
                tickers.append(s + "-USD")
            elif "." not in s:
                tickers.append(s)
            else:
                tickers.append(s)
        data = yf.download(tickers, period="60d", auto_adjust=True, progress=False)["Close"]
        if isinstance(data, pd.Series):
            return []
        corr = data.pct_change().dropna().corr()
        sym_col = symbol + "-USD" if symbol in ("BTC", "ETH") else symbol
        if sym_col not in corr.columns:
            return []
        correlated = []
        for open_sym in open_syms:
            col = open_sym + "-USD" if open_sym in ("BTC", "ETH") else open_sym
            if col in corr.columns:
                val = corr.loc[sym_col, col]
                if abs(val) >= threshold:
                    correlated.append(open_sym)
        return correlated
    except Exception:
        return []

__all__ = ['size_position', 'kelly_position_size', 'check_stops', 'mark_partial_done', 'record_open', 'record_close', 'recent_trade_context', 'check_drawdown_circuit', 'get_equity_history', 'get_win_rate', 'vol_target_scalar', 'chandelier_exit', 'drawdown_radar_score']
