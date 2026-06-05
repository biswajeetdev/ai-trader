"""
ai4trade.py — AI4Trade paper-trading broker.
Handles authentication, order execution, and position fetching for ai4trade.ai.
"""

import json
import requests
from datetime import datetime
from pathlib import Path

DIR      = Path(__file__).parent.parent
BASE_URL = "https://ai4trade.ai"
TOKEN_FILE = DIR / ".token"


def _save_token(t: str) -> None:
    TOKEN_FILE.write_text(t)
    TOKEN_FILE.chmod(0o600)


def auth(cfg: dict) -> str:
    """Return a valid bearer token, logging in or registering as needed."""
    if TOKEN_FILE.exists():
        t = TOKEN_FILE.read_text().strip()
        if t:
            return t

    payload = {
        "name":     cfg.get("bot_name", "Michael_123"),
        "email":    cfg["email"],
        "password": cfg["password"],
    }
    resp = requests.post(f"{BASE_URL}/api/claw/agents/login", json=payload, timeout=15)
    if resp.ok:
        t = resp.json().get("token") or resp.json().get("access_token")
        if t:
            _save_token(t)
            return t

    resp = requests.post(f"{BASE_URL}/api/claw/agents/selfRegister", json=payload, timeout=15)
    if not resp.ok:
        raise RuntimeError(f"Auth failed: {resp.text[:200]}")
    t = resp.json().get("token") or resp.json().get("access_token")
    if not t:
        raise RuntimeError("No token in registration response")
    _save_token(t)
    print("  Registered — token saved.")
    return t


def refresh_token(cfg: dict) -> str:
    """Force re-auth by clearing the cached token, then authenticate."""
    if TOKEN_FILE.exists():
        TOKEN_FILE.unlink()
    return auth(cfg)


def get_profile(token: str) -> dict:
    r = requests.get(
        f"{BASE_URL}/api/claw/agents/me",
        headers={"Authorization": f"Bearer {token}"},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def execute_trade(
    token: str,
    symbol: str,
    market: str,
    action: str,
    qty: float,
    reason: str,
    dry_run: bool = False,
) -> dict | None:
    """Submit a signal to ai4trade.ai. Returns response dict or None on error."""
    if dry_run:
        print(f"    [DRY RUN] {action} {qty} {symbol} — {reason}")
        return {"dry_run": True}

    resp = requests.post(
        f"{BASE_URL}/api/signals/realtime",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={
            "market": market, "action": action.lower(), "symbol": symbol,
            "price": 0, "quantity": qty, "content": reason, "executed_at": "now",
        },
        timeout=15,
    )
    if not resp.ok:
        print(f"    [ERROR] {resp.status_code} — {resp.text[:200]}")
        return None
    return resp.json()


def get_positions_api(token: str) -> dict:
    """Fetch open positions from ai4trade.ai."""
    r = requests.get(
        f"{BASE_URL}/api/positions",
        headers={"Authorization": f"Bearer {token}"},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def with_token_refresh(cfg: dict, token: str, fn, *args, **kwargs):
    """Call fn(token, *args, **kwargs); on 401 refresh token once and retry."""
    try:
        return fn(token, *args, **kwargs)
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 401:
            token = refresh_token(cfg)
            return fn(token, *args, **kwargs)
        raise
