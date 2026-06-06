"""
performance.py — Portfolio performance analytics.
Computes Sharpe, Sortino, Max Drawdown, Win Rate, Profit Factor,
and Avg Hold Time from trade_history.json + equity curve.
"""

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

DIR        = Path(__file__).parent.parent
TRADE_HIST = DIR / "trade_history.json"
HWM_FILE   = DIR / "portfolio_hwm.json"

RISK_FREE_ANNUAL = 0.055  # ~5.5% (US T-bill proxy)
TRADING_DAYS     = 252


def _load_trades() -> list[dict]:
    if not TRADE_HIST.exists():
        return []
    try:
        return json.loads(TRADE_HIST.read_text())
    except Exception:
        return []


def _load_equity_history() -> list[float]:
    if not HWM_FILE.exists():
        return []
    try:
        return json.loads(HWM_FILE.read_text()).get("equity_history", [])
    except Exception:
        return []


def _daily_returns(equity: list[float]) -> np.ndarray:
    if len(equity) < 2:
        return np.array([])
    arr = np.array(equity, dtype=float)
    return np.diff(arr) / arr[:-1]


def sharpe_ratio(equity: list[float]) -> float | None:
    rets = _daily_returns(equity)
    if len(rets) < 20:
        return None
    rf_daily = RISK_FREE_ANNUAL / TRADING_DAYS
    excess   = rets - rf_daily
    std      = np.std(excess, ddof=1)
    if std == 0:
        return None
    return round(float(np.mean(excess) / std * math.sqrt(TRADING_DAYS)), 3)


def sortino_ratio(equity: list[float]) -> float | None:
    rets = _daily_returns(equity)
    if len(rets) < 20:
        return None
    rf_daily  = RISK_FREE_ANNUAL / TRADING_DAYS
    excess    = rets - rf_daily
    downside  = excess[excess < 0]
    if len(downside) == 0:
        return None
    downside_std = np.std(downside, ddof=1)
    if downside_std == 0:
        return None
    return round(float(np.mean(excess) / downside_std * math.sqrt(TRADING_DAYS)), 3)


def max_drawdown(equity: list[float]) -> float:
    """Returns max drawdown as a positive fraction (e.g. 0.12 = 12%)."""
    if len(equity) < 2:
        return 0.0
    arr        = np.array(equity, dtype=float)
    running_max = np.maximum.accumulate(arr)
    dd         = (running_max - arr) / np.where(running_max == 0, 1, running_max)
    return round(float(np.max(dd)), 4)


def win_rate_and_profit_factor(trades: list[dict]) -> dict:
    closed = [t for t in trades if t.get("pnl") is not None]
    if not closed:
        return {"win_rate": None, "profit_factor": None, "total_trades": 0}

    wins   = [t["pnl"] for t in closed if t["pnl"] > 0]
    losses = [t["pnl"] for t in closed if t["pnl"] <= 0]

    gross_profit = sum(wins)
    gross_loss   = abs(sum(losses)) or 1e-9

    return {
        "win_rate":       round(len(wins) / len(closed), 4),
        "profit_factor":  round(gross_profit / gross_loss, 3),
        "total_trades":   len(closed),
        "winning_trades": len(wins),
        "losing_trades":  len(losses),
        "avg_win":        round(sum(wins)   / len(wins),   2) if wins   else 0,
        "avg_loss":       round(sum(losses) / len(losses), 2) if losses else 0,
    }


def avg_hold_days(trades: list[dict]) -> float | None:
    durations = []
    for t in trades:
        try:
            entry = datetime.fromisoformat(t["entry_date"])
            exit_ = datetime.fromisoformat(t.get("exit_date") or t.get("close_date") or "")
            durations.append((exit_ - entry).days)
        except Exception:
            continue
    if not durations:
        return None
    return round(sum(durations) / len(durations), 1)


def get_performance_report() -> dict:
    """
    Returns a structured performance report dict.
    Safe to call at any time — returns None for metrics that need more data.
    """
    trades  = _load_trades()
    equity  = _load_equity_history()
    wrf     = win_rate_and_profit_factor(trades)
    hold    = avg_hold_days(trades)
    sharpe  = sharpe_ratio(equity)
    sortino = sortino_ratio(equity)
    mdd     = max_drawdown(equity)

    recent_pnl = sum(t.get("pnl", 0) or 0 for t in trades[-20:])
    total_pnl  = sum(t.get("pnl", 0) or 0 for t in trades)

    return {
        "sharpe_ratio":       sharpe,
        "sortino_ratio":      sortino,
        "max_drawdown_pct":   round(mdd * 100, 2),
        "win_rate_pct":       round(wrf["win_rate"] * 100, 2) if wrf["win_rate"] is not None else None,
        "profit_factor":      wrf["profit_factor"],
        "total_trades":       wrf["total_trades"],
        "winning_trades":     wrf.get("winning_trades"),
        "losing_trades":      wrf.get("losing_trades"),
        "avg_win_usd":        wrf.get("avg_win"),
        "avg_loss_usd":       wrf.get("avg_loss"),
        "avg_hold_days":      hold,
        "total_pnl_usd":      round(total_pnl, 2),
        "recent_20_pnl_usd":  round(recent_pnl, 2),
        "equity_data_points": len(equity),
    }


def format_for_summary(report: dict) -> str:
    """Human-readable block for Telegram / daily email."""
    def fmt(val, suffix="", na="N/A"):
        return f"{val}{suffix}" if val is not None else na

    lines = [
        "── Performance Report ──────────────────",
        f"  Sharpe    : {fmt(report['sharpe_ratio'])}",
        f"  Sortino   : {fmt(report['sortino_ratio'])}",
        f"  Max DD    : {fmt(report['max_drawdown_pct'], '%')}",
        f"  Win Rate  : {fmt(report['win_rate_pct'], '%')}",
        f"  Pft Factor: {fmt(report['profit_factor'])}",
        f"  Trades    : {report['total_trades']}  "
        f"(W:{report.get('winning_trades','?')} / L:{report.get('losing_trades','?')})",
        f"  Avg Win   : ${fmt(report['avg_win_usd'])}  "
        f"Avg Loss: ${fmt(report['avg_loss_usd'])}",
        f"  Avg Hold  : {fmt(report['avg_hold_days'], 'd')}",
        f"  Total PnL : ${fmt(report['total_pnl_usd'])}",
        f"  Last 20   : ${fmt(report['recent_20_pnl_usd'])}",
        "────────────────────────────────────────",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    report = get_performance_report()
    print(format_for_summary(report))
