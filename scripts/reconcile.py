#!/usr/bin/env python3
"""
reconcile.py — compare local bot state against broker truth.

The bot decides from local JSON (positions.json, short_put_positions.json,
portfolio_hwm.json). Those files drift silently: option assignments, manual
trades and failed writes never make it back. When they drift, every downstream
signal is computed against a portfolio that does not exist.

Read-only by DEFAULT: it reports drift and exits non-zero, so it can be wired
into cron. `--fix` adopts broker truth into positions.json, and adopted rows are
marked `unmanaged` so check_stops() will not act on a reconstructed cost basis —
promoting one to managed is an explicit `--arm` decision.

Usage:
  python3 scripts/reconcile.py                 # report only (default)
  python3 scripts/reconcile.py --fix           # adopt broker truth, unmanaged
  python3 scripts/reconcile.py --arm NVDA,MSFT # let the bot manage these
Exit: 0 = in sync, 1 = drift found, 2 = could not reach broker
"""

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from broker.state_io import atomic_write_json   # noqa: E402

DIR = Path(__file__).resolve().parent.parent
ALPACA = "https://paper-api.alpaca.markets"
DEFAULT_START_EQUITY = 100_000.0


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


def _baseline(cfg: dict) -> tuple[float, str]:
    """
    The account's real starting equity, from Alpaca's portfolio history
    (`base_value` over period=all). Hardcoding $100,000 silently misstates every
    return figure the moment the account is reset or funded differently.

    Returns (baseline, source) so an assumed value is always labelled as one.
    """
    try:
        h = {"APCA-API-KEY-ID": cfg["alpaca_api_key"],
             "APCA-API-SECRET-KEY": cfg["alpaca_secret_key"]}
        r = requests.get(f"{ALPACA}/v2/account/portfolio/history", headers=h,
                         params={"period": "all", "timeframe": "1D"}, timeout=20)
        if r.ok:
            d = r.json()
            base = d.get("base_value")
            if base:
                asof = d.get("base_value_asof") or ""
                eq = d.get("equity") or []
                # base_value can be 0 on a brand-new account; prefer first real point
                if float(base) <= 0 and eq:
                    first = next((e for e in eq if e and float(e) > 0), None)
                    if first:
                        return float(first), "broker portfolio history (first funded day)"
                return float(base), f"broker portfolio history{f' asof {asof}' if asof else ''}"
    except Exception:
        pass
    return DEFAULT_START_EQUITY, "ASSUMED DEFAULT — could not read broker history"


def _atr(symbol: str, market: str = "us-stock") -> float:
    """14-day ATR for an adopted position, so it has a usable risk unit."""
    try:
        import yfinance as yf
        tk = f"{symbol}-USD" if market == "crypto" else symbol
        df = yf.download(tk, period="60d", interval="1d", progress=False,
                         auto_adjust=True)
        if df.empty or len(df) < 15:
            return 0.0
        h, l, c = (df["High"].squeeze(), df["Low"].squeeze(), df["Close"].squeeze())
        pc = c.shift(1)
        tr = (h - l).combine((h - pc).abs(), max).combine((l - pc).abs(), max)
        return round(float(tr.rolling(14).mean().iloc[-1]), 4)
    except Exception:
        return 0.0


def _repair(local: dict, bmap: dict, fix: bool, arm: set) -> list[str]:
    """
    Mutate `local` toward broker truth. Returns a log of what changed.

    Adoption records the broker's avg_entry_price as the basis and derives stops
    from live ATR, but flags the row `unmanaged` — the basis is real while the
    ATR-derived stop is a guess about intent, and arming a stop on a large
    position could liquidate it at an arbitrary level. `--arm` is the human's
    signal that a row is safe to manage.
    """
    from broker.risk import STOP_LOSS_ATR, PROFIT_TARGET_ATR

    log = []
    today = datetime.now().strftime("%Y-%m-%d")

    if fix:
        for sym, bp in bmap.items():
            bqty = float(bp["qty"])
            direction = "SHORT" if bqty < 0 else "BUY"
            basis = float(bp["avg_entry_price"])
            market = "crypto" if "/" in sym or sym.endswith("USD") else "us-stock"

            if sym not in local:
                atr = _atr(sym, market)
                sgn = 1 if direction == "BUY" else -1
                local[sym] = {
                    "action": direction, "entry_price": basis, "quantity": abs(bqty),
                    "atr": atr, "atr_at_entry": atr, "market": market,
                    "entry_date": today,
                    "stop_price": round(basis - sgn * STOP_LOSS_ATR * atr, 4),
                    "target_price": round(basis + sgn * PROFIT_TARGET_ATR * atr, 4),
                    "trail_stop": round(basis - sgn * STOP_LOSS_ATR * atr, 4),
                    "partial_done": False,
                    "unmanaged": True, "adopted_from_broker": today,
                }
                if market != "crypto":
                    local[sym]["quantity"] = int(abs(bqty))
                log.append(f"ADOPTED  {sym:<6} {direction:<5} qty={abs(bqty):<8g} "
                           f"basis=${basis:<9.2f} atr={atr or 'n/a'}  [unmanaged]")
                continue

            lp = local[sym]
            new_qty = abs(bqty) if market == "crypto" else int(abs(bqty))
            if abs(float(lp.get("quantity", 0))) != abs(bqty):
                log.append(f"QTY      {sym:<6} {lp.get('quantity')} -> {new_qty} "
                           f"(broker truth)")
                lp["quantity"] = new_qty
            if str(lp.get("action", "BUY")).upper() != direction:
                log.append(f"DIR      {sym:<6} {lp.get('action')} -> {direction}")
                lp["action"] = direction
            if abs(float(lp.get("entry_price", 0)) - basis) > 0.01:
                log.append(f"BASIS    {sym:<6} ${lp.get('entry_price')} -> "
                           f"${round(basis, 4)} (broker blended avg)")
                lp["entry_price"] = round(basis, 4)
                # The existing stop/target were derived from the OLD basis and are
                # left alone on purpose: moving a live stop is a risk decision, not
                # a bookkeeping one. Flagged so it is a choice, not an oversight.
                if not lp.get("unmanaged"):
                    log.append(f"         {'':<6} ^ stop ${lp.get('stop_price')} / "
                               f"target ${lp.get('target_price')} NOT recomputed "
                               f"(still from old basis)")

        for sym in [s for s in local if s not in bmap]:
            log.append(f"REMOVED  {sym:<6} phantom — tracked locally, absent at broker")
            local.pop(sym)

    for sym in arm:
        if sym in local and local[sym].pop("unmanaged", None):
            local[sym]["armed_on"] = today
            log.append(f"ARMED    {sym:<6} now managed by check_stops "
                       f"(stop=${local[sym].get('stop_price')})")
        elif sym in local:
            log.append(f"ARMED    {sym:<6} already managed — no change")
        else:
            log.append(f"SKIP     {sym:<6} not in positions.json")

    return log


def main() -> int:
    ap = argparse.ArgumentParser(description="Reconcile local bot state against the broker.")
    ap.add_argument("--fix", action="store_true",
                    help="adopt broker truth into positions.json (adopted rows are unmanaged)")
    ap.add_argument("--arm", default="",
                    help="comma-separated symbols to promote to managed (clears unmanaged)")
    args = ap.parse_args()

    try:
        cfg = _cfg()
        acct, bpos = _broker(cfg)
    except Exception as e:
        print(f"[reconcile] BROKER UNREACHABLE: {type(e).__name__}: {str(e)[:120]}")
        return 2

    drift = []

    # ── Equity: broker truth vs what the bot has been reporting ───────────────
    equity = float(acct.get("equity", 0))
    baseline, base_src = _baseline(cfg)
    hwm = _local("portfolio_hwm.json")
    hist = hwm.get("equity_history", [])
    reported = hist[-1] if hist else None

    print("=" * 66)
    print("  BROKER TRUTH")
    print("=" * 66)
    pct = (equity / baseline - 1) * 100 if baseline else 0.0
    print(f"  equity        ${equity:>12,.2f}   ({equity - baseline:+,.2f} "
          f"/ {pct:+.2f}%)")
    print(f"  baseline      ${baseline:>12,.2f}   ({base_src})")
    print(f"  cash          ${float(acct.get('cash', 0)):>12,.2f}")
    print(f"  long mkt val  ${float(acct.get('long_market_value', 0)):>12,.2f}")
    if "ASSUMED" in base_src:
        drift.append(f"baseline is assumed (${baseline:,.2f}) — return figures unverified")

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

    # ── Repair (opt-in) ───────────────────────────────────────────────────────
    repaired = []
    if args.fix or args.arm:
        print("\n" + "=" * 66)
        print("  REPAIR")
        print("=" * 66)
        changed = repaired = _repair(
            local_eq, bmap, fix=args.fix,
            arm={s.strip().upper() for s in args.arm.split(",") if s.strip()})
        if changed:
            atomic_write_json(DIR / "positions.json", local_eq)
            for line in changed:
                print(f"  {line}")
            print(f"\n  positions.json updated ({len(changed)} change(s)).")
            print("  Adopted rows are marked `unmanaged`: visible and closable, but "
                  "check_stops()\n  will not fire on a reconstructed basis. Promote with "
                  "--arm SYM when verified.")
        else:
            print("  nothing to repair")

    # ── Verdict ───────────────────────────────────────────────────────────────
    # `drift` was collected BEFORE any repair, so it describes what was found on
    # entry. Say so explicitly rather than printing "DRIFT FOUND" about issues
    # that were just fixed.
    print("\n" + "=" * 66)
    if drift:
        print(f"  DRIFT FOUND ON ENTRY — {len(drift)} issue(s)")
        print("=" * 66)
        for d in drift:
            print(f"  • {d}")
        if repaired:
            print(f"\n  {len(repaired)} repaired above — re-run without --fix to confirm.")
            print("  Not repaired by --fix: expired short puts (settled by the bot's")
            print("  next run) and equity reporting (bot reads the ai4trade sim account,")
            print("  not Alpaca).")
        else:
            print("\n  The bot is deciding against state that does not match the broker.")
            print("  Run with --fix to adopt broker truth.")
        return 1
    print("  IN SYNC — local state matches broker")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
