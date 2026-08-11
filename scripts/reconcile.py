#!/usr/bin/env python3
"""
reconcile.py — compare local bot state against broker truth.

The bot decides from local JSON (positions.json, short_put_positions.json,
portfolio_hwm.json). Those files drift silently: option assignments, manual
trades and failed writes never make it back. When they drift, every downstream
signal is computed against a portfolio that does not exist.

Read-only by design. It reports drift and exits non-zero when it finds any, so
it can be wired into cron. It never modifies positions — repairing state is a
decision for a human, not a side effect of a health check.

Usage:  python3 scripts/reconcile.py
Exit:   0 = in sync, 1 = drift found, 2 = could not reach broker
"""

import json
import sys
from datetime import date
from pathlib import Path

import requests

DIR = Path(__file__).resolve().parent.parent
ALPACA = "https://paper-api.alpaca.markets"
START_EQUITY = 100_000.0


def _cfg() -> dict:
    return json.loads((DIR / "config.json").read_text())


def _local(name: str) -> dict:
    p = DIR / name
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _broker(cfg: dict) -> tuple[dict, list]:
    h = {"APCA-API-KEY-ID": cfg["alpaca_api_key"],
         "APCA-API-SECRET-KEY": cfg["alpaca_secret_key"]}
    acct = requests.get(f"{ALPACA}/v2/account", headers=h, timeout=20).json()
    pos = requests.get(f"{ALPACA}/v2/positions", headers=h, timeout=20).json()
    return acct, (pos if isinstance(pos, list) else [])


def main() -> int:
    try:
        cfg = _cfg()
        acct, bpos = _broker(cfg)
    except Exception as e:
        print(f"[reconcile] BROKER UNREACHABLE: {type(e).__name__}: {str(e)[:120]}")
        return 2

    drift = []

    # ── Equity: broker truth vs what the bot has been reporting ───────────────
    equity = float(acct.get("equity", 0))
    hwm = _local("portfolio_hwm.json")
    hist = hwm.get("equity_history", [])
    reported = hist[-1] if hist else None

    print("=" * 66)
    print("  BROKER TRUTH")
    print("=" * 66)
    print(f"  equity        ${equity:>12,.2f}   ({equity - START_EQUITY:+,.2f} "
          f"/ {(equity / START_EQUITY - 1) * 100:+.2f}% from ${START_EQUITY:,.0f})")
    print(f"  cash          ${float(acct.get('cash', 0)):>12,.2f}")
    print(f"  long mkt val  ${float(acct.get('long_market_value', 0)):>12,.2f}")

    if reported is not None:
        gap = equity - reported
        print(f"\n  bot reports   ${reported:>12,.2f}   (gap ${gap:+,.2f})")
        if abs(gap) > 1.0:
            drift.append(f"equity gap ${gap:+,.2f} — bot reports ${reported:,.2f}, "
                         f"broker says ${equity:,.2f}")
        if len(set(hist)) == 1 and len(hist) > 3:
            drift.append(f"equity_history frozen — {len(hist)} identical values "
                         f"(${hist[0]:,.2f}); portfolio is not marked to market")

    # ── Positions: local vs broker ────────────────────────────────────────────
    local_eq = _local("positions.json")
    bmap = {p["symbol"]: p for p in bpos}

    print("\n" + "=" * 66)
    print("  POSITIONS")
    print("=" * 66)
    print(f"  {'symbol':<10}{'local qty':>10}{'broker qty':>12}{'unreal P&L':>13}  status")
    for sym in sorted(set(local_eq) | set(bmap)):
        lq = local_eq.get(sym, {}).get("quantity")
        bq = bmap.get(sym, {}).get("qty")
        upl = bmap.get(sym, {}).get("unrealized_pl")
        upl_s = f"{float(upl):+,.2f}" if upl is not None else "—"
        if lq is not None and bq is None:
            status = "PHANTOM — tracked locally, not at broker"
        elif lq is None and bq is not None:
            status = "UNTRACKED — at broker, bot is blind to it"
        elif str(lq) != str(bq).lstrip("-").split(".")[0] and float(lq) != float(bq):
            status = "QTY MISMATCH"
        else:
            status = "ok"
        if status != "ok":
            drift.append(f"{sym}: {status} (local={lq}, broker={bq})")
        print(f"  {sym:<10}{str(lq if lq is not None else '—'):>10}"
              f"{str(bq if bq is not None else '—'):>12}{upl_s:>13}  {status}")

    # ── Short puts: expired contracts still occupying slots ───────────────────
    sp = _local("short_put_positions.json")
    if sp:
        print("\n" + "=" * 66)
        print("  SHORT PUTS")
        print("=" * 66)
        today = date.today()
        for sym, p in sp.items():
            try:
                dte = (date.fromisoformat(p["expiry"]) - today).days
            except Exception:
                dte = None
            tag = "ok"
            if dte is not None and dte < 0:
                tag = f"EXPIRED {-dte}d ago — occupying a MAX_OPEN slot"
                drift.append(f"short put {sym} ${p['strike']}P expired {-dte}d ago "
                             f"and is still tracked")
            print(f"  {sym:<8} ${p['strike']:>8.2f}P  exp {p['expiry']}  "
                  f"credit ${p.get('credit', 0):>8,.2f}  {tag}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 66)
    if drift:
        print(f"  DRIFT FOUND — {len(drift)} issue(s)")
        print("=" * 66)
        for d in drift:
            print(f"  • {d}")
        print("\n  The bot is deciding against state that does not match the broker.")
        return 1
    print("  IN SYNC — local state matches broker")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
