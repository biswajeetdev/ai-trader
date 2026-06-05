"""
hyperliquid_arb.py — Hyperliquid funding rate arbitrage monitor (delta-neutral)
Free public API: https://api.hyperliquid.xyz/info

Strategy:
  - Long spot BTC/ETH on Alpaca/exchange
  - Short perpetual on Hyperliquid
  - Collect funding payments (positive rate = shorts receive payment)
  - Delta-neutral: spot gain = perp loss, profit = funding only
  - Expected: 15-30% annualized when rate > 10% (annualized) consistently

Alert threshold: annualized funding > 15% — this is the entry signal.
"""

import json, time, requests
from pathlib import Path
from datetime import datetime, timezone

CACHE_FILE = Path(__file__).parent.parent / "signals_cache.json"
CACHE_TTL  = 300   # 5 min — funding rates update every 8h but worth checking frequently

# Hyperliquid free public API
HL_META_URL  = "https://api.hyperliquid.xyz/info"
FUNDING_THRESHOLD_ANN = 0.15   # 15% annualized = entry signal
TOP_COINS = ["BTC", "ETH", "SOL", "AVAX", "ARB"]


def _cache(key, val=None):
    try:
        data = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
        if val is None:
            e = data.get(key)
            return e["data"] if e and time.time() - e.get("ts", 0) < CACHE_TTL else None
        data[key] = {"data": val, "ts": time.time()}
        CACHE_FILE.write_text(json.dumps(data, indent=2))
    except Exception:
        return None


def _fetch_funding_rates() -> list[dict]:
    """Fetch current funding rates from Hyperliquid public API."""
    try:
        r = requests.post(
            HL_META_URL,
            json={"type": "metaAndAssetCtxs"},
            timeout=10,
        )
        data = r.json()
        if not isinstance(data, list) or len(data) < 2:
            return []

        universe = data[0].get("universe", [])
        asset_ctxs = data[1]

        rates = []
        for i, asset in enumerate(universe):
            name = asset.get("name", "")
            if name not in TOP_COINS:
                continue
            if i >= len(asset_ctxs):
                continue
            ctx = asset_ctxs[i]
            funding_rate = float(ctx.get("funding", 0) or 0)
            mark_price   = float(ctx.get("markPx", 0) or 0)
            # Funding rate is per 8h period; annualize: * 3 * 365
            ann_rate = funding_rate * 3 * 365
            rates.append({
                "symbol":      name,
                "funding_8h":  round(funding_rate * 100, 4),   # as percentage
                "funding_ann": round(ann_rate * 100, 2),        # as percentage
                "mark_price":  round(mark_price, 2),
            })
        return rates
    except Exception:
        return []


def get_funding_arb_signals() -> list[dict]:
    """
    Returns list of arbitrage opportunities sorted by annualized funding rate.
    Each entry: {symbol, funding_8h, funding_ann, mark_price, signal, summary}
    """
    cached = _cache("hl_funding")
    if cached:
        return cached

    rates = _fetch_funding_rates()
    if not rates:
        return []

    signals = []
    for r in rates:
        ann = r["funding_ann"]
        if ann >= FUNDING_THRESHOLD_ANN * 100:
            sig = "ARB_OPPORTUNITY"
            summary = (f"HL FUNDING ARB: {r['symbol']} {ann:.1f}% ann "
                       f"({r['funding_8h']:.4f}%/8h) — long spot + short perp")
        elif ann >= 5.0:
            sig = "WATCH"
            summary = f"HL FUNDING: {r['symbol']} {ann:.1f}% ann — below threshold, watching"
        elif ann < 0:
            sig = "REVERSE_ARB"
            summary = f"HL FUNDING: {r['symbol']} {ann:.1f}% ann NEGATIVE — longs receive funding"
        else:
            sig = "NEUTRAL"
            summary = f"HL FUNDING: {r['symbol']} {ann:.1f}% ann — neutral"

        signals.append({
            **r,
            "signal":  sig,
            "summary": summary,
        })

    signals.sort(key=lambda x: x["funding_ann"], reverse=True)
    _cache("hl_funding", signals)
    return signals


def format_for_telegram(signals: list[dict]) -> str:
    """Format funding arb opportunities for Telegram alert."""
    arb = [s for s in signals if s["signal"] == "ARB_OPPORTUNITY"]
    if not arb:
        return ""
    lines = ["🔄 *Funding Rate Arb Opportunities*"]
    for s in arb:
        lines.append(f"• {s['symbol']}: {s['funding_ann']:.1f}% ann ({s['funding_8h']:.4f}%/8h)")
    lines.append("_Action: long spot + short perp on Hyperliquid_")
    return "\n".join(lines)


def get_best_arb() -> dict | None:
    """Returns the single best ARB_OPPORTUNITY or None."""
    signals = get_funding_arb_signals()
    arb = [s for s in signals if s["signal"] == "ARB_OPPORTUNITY"]
    return arb[0] if arb else None


if __name__ == "__main__":
    signals = get_funding_arb_signals()
    if not signals:
        print("No Hyperliquid data available")
    else:
        for s in signals:
            print(f"{s['symbol']:6} {s['funding_ann']:>7.2f}% ann  {s['signal']}")
        best = get_best_arb()
        if best:
            print(f"\nBest opportunity: {best['summary']}")
