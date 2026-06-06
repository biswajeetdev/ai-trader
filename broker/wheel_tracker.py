"""wheel_tracker.py — Track Wheel strategy state per symbol.
States: IDLE → SHORT_PUT → ASSIGNED → COVERED_CALL → IDLE
"""

import json
from datetime import date
from pathlib import Path

try:
    from alpaca.trading.client import TradingClient
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

STATE_FILE = Path(__file__).parent / "wheel_state.json"

VALID_STATES = {"IDLE", "SHORT_PUT", "ASSIGNED", "COVERED_CALL"}


def _load() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {}


def _save(data: dict) -> None:
    STATE_FILE.write_text(json.dumps(data, indent=2))


def get_wheel_state(symbol: str) -> dict:
    # Returns current wheel state for symbol, defaulting to IDLE
    data = _load()
    return data.get(symbol, {
        "symbol":                  symbol,
        "state":                   "IDLE",
        "cost_basis":              None,
        "total_premium_collected": 0.0,
        "updated_at":              None,
    })


def update_wheel_state(symbol: str, new_state: str, **kwargs) -> dict:
    # Advance state machine; accumulates premium_collected automatically
    if new_state not in VALID_STATES:
        raise ValueError(f"Invalid state: {new_state}. Must be one of {VALID_STATES}")

    data  = _load()
    entry = data.get(symbol, {
        "symbol":                  symbol,
        "state":                   "IDLE",
        "cost_basis":              None,
        "total_premium_collected": 0.0,
    })

    entry["state"]      = new_state
    entry["updated_at"] = date.today().isoformat()

    # Accumulate premium when caller provides it
    if "premium_collected" in kwargs:
        entry["total_premium_collected"] = round(
            entry.get("total_premium_collected", 0.0) + kwargs.pop("premium_collected"), 2
        )

    entry.update(kwargs)
    data[symbol] = entry
    _save(data)
    return entry


def get_assignable_symbols(cfg: dict) -> list[str]:
    # Check Alpaca portfolio for symbols with 100+ shares that have no covered call open
    if not ALPACA_AVAILABLE:
        return []

    k = cfg.get("alpaca_api_key", "")
    s = cfg.get("alpaca_secret_key", "")
    if not k or not s:
        return []

    try:
        from broker.covered_call_exec import load_positions as load_cc_positions
        cc_active = load_cc_positions()

        client    = TradingClient(k, s, paper=cfg.get("alpaca_paper", True))
        positions = client.get_all_positions()

        return [
            p.symbol for p in positions
            if float(p.qty) >= 100 and p.symbol not in cc_active
        ]
    except Exception:
        return []


def format_wheel_status() -> str:
    # One-line summary suitable for Telegram
    data = _load()
    if not data:
        return "Wheel: no active positions"

    parts = []
    for symbol, entry in data.items():
        state   = entry.get("state", "IDLE")
        premium = entry.get("total_premium_collected", 0.0)
        parts.append(f"{symbol}:{state}(${premium:.0f})")

    return "Wheel | " + " | ".join(parts)
