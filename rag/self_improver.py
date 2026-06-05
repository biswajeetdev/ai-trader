"""
self_improver.py — QuantAgent-style outer loop.

After each closed trade:
  1. Record which signals were present (options_flow, whale, insider, macro, poly, social)
  2. Record outcome (WIN/LOSS, pnl_pct)
  3. Update signal attribution stats in rag/agent_calibration.json
  4. Recompute calibration prompt injected into ARBITER_SYSTEM

The arbiter is thus told: "options_flow has 73% win rate (18 trades),
social_only has 31% win rate (13 trades)" — and weights signals accordingly.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

CALIB_FILE = Path(__file__).parent / "agent_calibration.json"
TRADE_HISTORY = Path(__file__).parent.parent / "trade_history.json"
ROLLING_WINDOW = 50
KNOWN_SIGNALS = ["options_flow", "whale", "insider", "macro", "poly", "social", "news"]


def _load() -> dict:
    """Load calibration JSON, return empty scaffold on any failure."""
    try:
        if CALIB_FILE.exists():
            data = json.loads(CALIB_FILE.read_text())
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {"signal_stats": {}, "regime_stats": {}, "seen_trades": [],
            "attribution_log": [], "last_updated": None}


def _save(data: dict):
    """Persist calibration data with updated timestamp."""
    data["last_updated"] = datetime.now(timezone.utc).isoformat()
    CALIB_FILE.write_text(json.dumps(data, indent=2))


def _trade_key(symbol: str, action: str, outcome_pct: float) -> str:
    """Build a stable dedup key for a trade record."""
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"{symbol}_{action}_{outcome_pct:.3f}_{date}"


def record_signal_outcome(symbol: str, action: str, outcome_pct: float,
                          signals_present: list):
    """Record which signals fired and whether the trade won or lost."""
    data = _load()
    key = _trade_key(symbol, action, outcome_pct)
    seen = data.setdefault("seen_trades", [])

    if key in seen:
        return

    is_win = outcome_pct > 0
    sig_stats = data.setdefault("signal_stats", {})

    for sig in signals_present:
        entry = sig_stats.setdefault(sig, {"wins": 0, "total": 0})
        entry["total"] += 1
        if is_win:
            entry["wins"] += 1

    log = data.setdefault("attribution_log", [])
    log.append({
        "symbol": symbol, "action": action,
        "outcome_pct": round(outcome_pct, 4),
        "signals": signals_present, "win": is_win,
        "ts": datetime.now(timezone.utc).isoformat(),
    })
    data["attribution_log"] = log[-ROLLING_WINDOW:]

    seen.append(key)
    data["seen_trades"] = seen[-200:]
    _save(data)


def get_calibration_prompt() -> str:
    """Build the SIGNAL CALIBRATION block for injection into ARBITER_SYSTEM."""
    try:
        data = _load()
        sig_stats = data.get("signal_stats", {})
        total = sum(v["total"] for v in sig_stats.values())
        if total < 5:
            return ""

        lines = [f"SIGNAL CALIBRATION (from {total} past trades):"]

        rows = []
        for sig, stat in sig_stats.items():
            if stat["total"] == 0:
                continue
            rate = stat["wins"] / stat["total"]
            rows.append((sig, rate, stat["total"]))
        rows.sort(key=lambda x: -x[1])

        for sig, rate, n in rows:
            pct = round(rate * 100)
            if rate >= 0.60:
                label = "OVERWEIGHT"
            elif rate < 0.40:
                label = "UNDERWEIGHT"
            elif rate >= 0.50:
                label = "TRUST"
            else:
                label = "NEUTRAL"
            lines.append(f"  {sig:<14}: {pct}% win rate ({n} trades) — {label}")

        lines.append("")
        lines.append("Adjust your arbiter weights: OVERWEIGHT signals (>60%) deserve extra weight.")
        lines.append("UNDERWEIGHT signals (<40%) should require corroboration from higher-priority sources.")
        return "\n".join(lines)
    except Exception:
        return ""


def _regime_line(data: dict) -> str:
    """Format per-regime win rates into a single summary string."""
    reg = data.get("regime_stats", {})
    if not reg:
        return ""
    parts = []
    for regime, stat in reg.items():
        if stat["total"] == 0:
            continue
        pct = round(stat["wins"] / stat["total"] * 100)
        parts.append(f"{regime} {pct}% ({stat['total']}t)")
    return ("Regime performance: " + ", ".join(parts)) if parts else ""


def get_regime_calibration() -> str:
    """Return signal calibration combined with per-regime win rates."""
    try:
        data = _load()
        signal_block = get_calibration_prompt()
        regime_part = _regime_line(data)
        if not signal_block and not regime_part:
            return ""
        parts = [p for p in [signal_block, regime_part] if p]
        return "\n".join(parts)
    except Exception:
        return ""


def update_from_trade_history() -> int:
    """Bootstrap regime stats from trade_history.json; signals marked unattributed."""
    if not TRADE_HISTORY.exists():
        return 0
    try:
        history = json.loads(TRADE_HISTORY.read_text())
    except Exception:
        return 0

    data = _load()
    seen = set(data.get("seen_trades", []))
    reg_stats = data.setdefault("regime_stats", {})
    bootstrapped = 0

    for trade in history:
        if not isinstance(trade, dict):
            continue

        symbol = trade.get("symbol", "")
        action = trade.get("action", trade.get("side", ""))
        pnl = float(trade.get("pnl_pct", trade.get("pnl", 0)) or 0)
        date = str(trade.get("date", trade.get("timestamp", "")))[:10]
        key = f"{symbol}_{action}_{pnl:.3f}_{date}_hist"

        if key in seen:
            continue

        regime = str(trade.get("regime", "MIXED")).upper()
        if regime not in ("BOT_DRIVEN", "HUMAN_DRIVEN"):
            regime = "MIXED"

        entry = reg_stats.setdefault(regime, {"wins": 0, "total": 0})
        entry["total"] += 1
        if pnl > 0:
            entry["wins"] += 1

        seen.add(key)
        bootstrapped += 1

    data["seen_trades"] = list(seen)[-200:]
    if bootstrapped:
        _save(data)
    return bootstrapped
