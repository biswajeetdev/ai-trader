"""dashboard/state.py — Thread-safe state writer for the Bloomberg terminal.
Trader calls these functions; terminal.py reads dashboard_state.json live.
"""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

STATE_FILE = Path(__file__).parent.parent / "dashboard_state.json"
_lock = threading.Lock()

_EMPTY: dict = {
    "account": {}, "macro": {}, "positions": [],
    "signal_feed": [], "trade_log": [],
    "strategies": {}, "lessons": [],
    "radar": {}, "win_rate": {},
    "pipeline": {},
    "current_symbol": "", "status": "IDLE",
    "last_updated": None,
}


def _load() -> dict:
    try:
        if STATE_FILE.exists():
            return json.loads(STATE_FILE.read_text())
    except Exception:
        pass
    return {k: (v.copy() if isinstance(v, dict) else list(v) if isinstance(v, list) else v)
            for k, v in _EMPTY.items()}


def _save(data: dict) -> None:
    data["last_updated"] = datetime.now(timezone.utc).strftime("%H:%M:%S")
    STATE_FILE.write_text(json.dumps(data, indent=2))


# ── Public writers ────────────────────────────────────────────────────────────

def set_status(status: str) -> None:
    with _lock:
        d = _load(); d["status"] = status; _save(d)


def set_current_symbol(symbol: str) -> None:
    with _lock:
        d = _load(); d["current_symbol"] = symbol; _save(d)


def update_account(equity: float, cash: float, pnl_today: float = 0.0) -> None:
    base = max(equity - pnl_today, 1)
    with _lock:
        d = _load()
        d["account"] = {
            "equity": round(equity, 2), "cash": round(cash, 2),
            "pnl_today": round(pnl_today, 2),
            "pnl_pct": round(pnl_today / base * 100, 2),
        }
        _save(d)


def update_macro(vix: float, spy_5d: float = 0.0, fg_score: int = 0,
                 bot_score: int = 50, regime: str = "") -> None:
    with _lock:
        d = _load()
        d["macro"] = {
            "vix": vix, "spy_5d": spy_5d, "fg_score": fg_score,
            "bot_score": bot_score, "regime": regime,
        }
        _save(d)


def update_positions(positions: list) -> None:
    """positions: list of dicts from load_positions() or Alpaca portfolio."""
    rows = []
    for p in positions:
        entry = float(p.get("entry_price") or p.get("avg_entry_price") or 0)
        curr  = float(p.get("current_price") or entry)
        pnl   = round((curr - entry) / entry * 100, 2) if entry else 0.0
        rows.append({
            "symbol":  p.get("symbol", ""),
            "qty":     p.get("qty", p.get("quantity", 0)),
            "entry":   round(entry, 2),
            "current": round(curr, 2),
            "pnl_pct": pnl,
            "market":  p.get("market", ""),
        })
    with _lock:
        d = _load(); d["positions"] = rows; _save(d)


def add_signal_event(symbol: str, action: str, conf: int,
                     bull: str, bear: str, consensus: str, reason: str) -> None:
    try:
        import pytz
        ts = datetime.now(pytz.timezone("America/New_York")).strftime("%H:%M:%S")
    except Exception:
        ts = datetime.now().strftime("%H:%M:%S")
    with _lock:
        d    = _load()
        feed = d.get("signal_feed", [])
        feed.append({
            "time": ts, "symbol": symbol, "action": action,
            "conf": conf, "bull": bull[:80], "bear": bear[:80],
            "consensus": consensus, "reason": reason[:120],
        })
        d["signal_feed"] = feed[-40:]
        _save(d)


def add_trade(action: str, symbol: str, qty: float,
              price: float, reason: str = "") -> None:
    try:
        import pytz
        ts = datetime.now(pytz.timezone("America/New_York")).strftime("%H:%M:%S")
    except Exception:
        ts = datetime.now().strftime("%H:%M:%S")
    with _lock:
        d   = _load()
        log = d.get("trade_log", [])
        log.append({
            "time": ts, "action": action, "symbol": symbol,
            "qty": qty, "price": round(price, 2), "reason": reason[:70],
        })
        d["trade_log"] = log[-50:]
        _save(d)


def update_strategies(calib_data: dict) -> None:
    """calib_data: strategy_calibration.json strategies dict."""
    rows = {}
    for name, s in calib_data.items():
        a, b = s.get("alpha", 1), s.get("beta", 1)
        rows[name] = {
            "score":  round(a / (a + b), 2),
            "wins":   s.get("wins", 0),
            "losses": s.get("losses", 0),
        }
    with _lock:
        d = _load(); d["strategies"] = rows; _save(d)


def update_lessons(lessons: list) -> None:
    with _lock:
        d = _load(); d["lessons"] = lessons[-6:]; _save(d)


def update_radar(score: int, level: str) -> None:
    with _lock:
        d = _load(); d["radar"] = {"score": score, "level": level}; _save(d)


def update_pipeline(stage: str, status: str, latency_ms: int = 0) -> None:
    """Update a pipeline stage. stage e.g. 'yfinance','debate','arbiter','alpaca'."""
    with _lock:
        d = _load()
        d.setdefault("pipeline", {})[stage] = {
            "status":     status,
            "latency_ms": latency_ms,
            "ts":         datetime.now(timezone.utc).strftime("%H:%M:%S"),
        }
        _save(d)


def update_win_rate(rate: float, trades: int) -> None:
    with _lock:
        d = _load(); d["win_rate"] = {"rate": round(rate, 1), "trades": trades}; _save(d)
