"""Pending trade queue for Telegram approval flow.

Trades are written to pending_trades.json and picked up by approval_bot.py.
Thread-safe via a file-level lock.
"""

import json, threading, time, uuid
from pathlib import Path

_QUEUE_FILE = Path(__file__).parent.parent / "pending_trades.json"
_lock = threading.Lock()
APPROVAL_TIMEOUT_S = 900  # 15 min


def _read() -> dict:
    if _QUEUE_FILE.exists():
        try:
            return json.loads(_QUEUE_FILE.read_text())
        except Exception:
            pass
    return {}


def _write(data: dict):
    _QUEUE_FILE.write_text(json.dumps(data, indent=2))


def queue_trade(cfg: dict, symbol: str, market: str, action: str,
                qty, price: float, conf: int, reason: str,
                atr: float, message_id: int | None = None) -> str:
    """Save a trade to the queue and return its trade_id."""
    trade_id = uuid.uuid4().hex[:10]
    entry = {
        "trade_id":  trade_id,
        "symbol":    symbol,
        "market":    market,
        "action":    action,
        "qty":       qty,
        "price":     price,
        "conf":      conf,
        "reason":    reason,
        "atr":       atr,
        "message_id": message_id,
        "chat_id":   cfg.get("telegram_chat_id", ""),
        "status":    "pending",
        "queued_at": time.time(),
        "expires_at": time.time() + APPROVAL_TIMEOUT_S,
    }
    with _lock:
        data = _read()
        data[trade_id] = entry
        _write(data)
    return trade_id


def get_trade(trade_id: str) -> dict | None:
    with _lock:
        return _read().get(trade_id)


def update_message_id(trade_id: str, message_id: int):
    with _lock:
        data = _read()
        if trade_id in data:
            data[trade_id]["message_id"] = message_id
            _write(data)


def mark_done(trade_id: str, status: str, result: dict | None = None):
    """status: 'approved' | 'skipped' | 'expired'"""
    with _lock:
        data = _read()
        if trade_id in data:
            data[trade_id]["status"] = status
            data[trade_id]["result"] = result or {}
            data[trade_id]["resolved_at"] = time.time()
            _write(data)


def get_expired() -> list:
    now = time.time()
    with _lock:
        data = _read()
    return [t for t in data.values()
            if t["status"] == "pending" and t["expires_at"] < now]


def get_pending() -> list:
    with _lock:
        data = _read()
    return [t for t in data.values() if t["status"] == "pending"]


def cleanup_old(max_age_s: int = 86400 * 3):
    """Remove trades resolved more than 3 days ago."""
    cutoff = time.time() - max_age_s
    with _lock:
        data = _read()
        data = {k: v for k, v in data.items()
                if v.get("resolved_at", time.time()) > cutoff
                or v["status"] == "pending"}
        _write(data)
