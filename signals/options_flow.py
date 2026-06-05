"""
options_flow.py — Smart money options activity detector
Sources: yfinance option chains (free, no key)

What it detects:
  PUT/CALL ratio  — overall sentiment (<0.7 bullish, >1.2 bearish)
  Unusual volume  — strikes with volume > 5× open interest (someone knows something)
  OTM call sweep  — large block buys of out-of-money calls (speculative upside bets)
  OTM put sweep   — large OTM puts = someone hedging / betting downside hard
  Gamma wall      — strike with highest open interest = price magnet / resistance

Signal format matches social_pulse.py for drop-in LLM context.
"""

import time, json
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone
from pathlib import Path

CACHE_FILE = Path(__file__).parent.parent / "signals_cache.json"
CACHE_TTL  = 25 * 60   # 25 min — options data updates intraday

# Thresholds
UNUSUAL_VOL_MULT  = 5      # vol > 5× OI = unusual
OTM_SWEEP_MIN_VOL = 100    # minimum contracts for a sweep to matter
PC_BULLISH        = 0.7    # put/call ratio below this = bullish
PC_BEARISH        = 1.2    # above this = bearish


def _load_cache():
    if not CACHE_FILE.exists():
        return {}
    try:
        data = json.loads(CACHE_FILE.read_text())
        if time.time() - data.get("_ts", 0) > CACHE_TTL:
            return {}
        return data
    except Exception:
        return {}


def _save_cache(data):
    data["_ts"] = time.time()
    try:
        CACHE_FILE.write_text(json.dumps(data, indent=2))
    except Exception:
        pass


def _analyze_chain(ticker_sym, market):
    """
    Fetch nearest 2 expiries and analyse options chain.
    Returns dict of signals for this symbol.
    """
    result = {"symbol": ticker_sym, "signals": [], "pc_ratio": None,
              "gamma_wall": None, "unusual": []}
    try:
        yfp  = f"{ticker_sym}-USD" if market == "crypto" else ticker_sym
        tk   = yf.Ticker(yfp)
        exps = tk.options
        if not exps:
            return result

        price = tk.fast_info.get("lastPrice") or tk.fast_info.get("regularMarketPrice")
        if not price:
            hist  = tk.history(period="2d")
            price = float(hist["Close"].iloc[-1]) if not hist.empty else None
        if not price:
            return result

        # Analyse nearest 2 expiries for volume richness
        all_calls = []
        all_puts  = []
        for exp in exps[:2]:
            try:
                chain = tk.option_chain(exp)
                all_calls.append(chain.calls)
                all_puts.append(chain.puts)
            except Exception:
                continue

        if not all_calls:
            return result

        calls = pd.concat(all_calls, ignore_index=True)
        puts  = pd.concat(all_puts,  ignore_index=True)

        # Fill NaN
        for col in ["volume", "openInterest", "strike"]:
            if col in calls.columns: calls[col] = calls[col].fillna(0)
            if col in puts.columns:  puts[col]  = puts[col].fillna(0)

        # ── Put/Call ratio ─────────────────────────────────────────────────────
        total_call_vol = float(calls["volume"].sum())
        total_put_vol  = float(puts["volume"].sum())
        if total_call_vol > 0:
            pc = round(total_put_vol / total_call_vol, 2)
            result["pc_ratio"] = pc
            if pc < PC_BULLISH:
                result["signals"].append({
                    "type": "PC_RATIO", "direction": "BULLISH",
                    "detail": f"Put/call ratio {pc:.2f} (bullish threshold <{PC_BULLISH})",
                    "score": 0.55,
                })
            elif pc > PC_BEARISH:
                result["signals"].append({
                    "type": "PC_RATIO", "direction": "BEARISH",
                    "detail": f"Put/call ratio {pc:.2f} (bearish threshold >{PC_BEARISH})",
                    "score": -0.55,
                })

        # ── Gamma wall (highest OI strike = price magnet) ──────────────────────
        if "openInterest" in calls.columns and len(calls) > 0:
            top_call = calls.loc[calls["openInterest"].idxmax()]
            result["gamma_wall"] = round(float(top_call["strike"]), 2)

        # ── Unusual volume (calls) ─────────────────────────────────────────────
        if "openInterest" in calls.columns:
            calls = calls[calls["openInterest"] > 10].copy()
            calls["vol_oi_ratio"] = calls["volume"] / calls["openInterest"].replace(0, np.nan)
            unusual_calls = calls[
                (calls["vol_oi_ratio"] > UNUSUAL_VOL_MULT) &
                (calls["volume"] >= OTM_SWEEP_MIN_VOL)
            ].sort_values("volume", ascending=False)

            for _, row in unusual_calls.head(3).iterrows():
                strike  = float(row["strike"])
                otm_pct = round((strike - price) / price * 100, 1)
                vol     = int(row["volume"])
                result["unusual"].append({
                    "side": "CALL", "strike": strike, "otm_pct": otm_pct,
                    "volume": vol, "vol_oi_ratio": round(float(row["vol_oi_ratio"]), 1),
                })
                direction = "BULLISH"
                score     = 0.70 if otm_pct > 5 else 0.50  # OTM call sweep = stronger signal
                result["signals"].append({
                    "type": "UNUSUAL_CALL", "direction": direction,
                    "detail": (f"Unusual CALL sweep: ${strike} strike "
                               f"({otm_pct:+.1f}% OTM) vol={vol} "
                               f"({row['vol_oi_ratio']:.0f}×OI)"),
                    "score": score,
                })

        # ── Unusual volume (puts) ──────────────────────────────────────────────
        if "openInterest" in puts.columns:
            puts2 = puts[puts["openInterest"] > 10].copy()
            puts2["vol_oi_ratio"] = puts2["volume"] / puts2["openInterest"].replace(0, np.nan)
            unusual_puts = puts2[
                (puts2["vol_oi_ratio"] > UNUSUAL_VOL_MULT) &
                (puts2["volume"] >= OTM_SWEEP_MIN_VOL)
            ].sort_values("volume", ascending=False)

            for _, row in unusual_puts.head(2).iterrows():
                strike  = float(row["strike"])
                otm_pct = round((price - strike) / price * 100, 1)
                vol     = int(row["volume"])
                score   = -0.65 if otm_pct > 5 else -0.45
                result["signals"].append({
                    "type": "UNUSUAL_PUT", "direction": "BEARISH",
                    "detail": (f"Unusual PUT sweep: ${strike} strike "
                               f"({otm_pct:+.1f}% OTM) vol={vol}"),
                    "score": score,
                })

    except Exception as e:
        pass

    return result


def get_options_signals(watchlist_tickers=None, use_cache=True):
    """
    Fetch options flow signals for all tickers.
    Returns list of signal dicts sorted by score magnitude.
    """
    cache = _load_cache() if use_cache else {}
    if cache.get("options_signals"):
        return cache["options_signals"]

    tickers = watchlist_tickers or []
    markets = {}  # infer from ticker name
    for sym in tickers:
        markets[sym] = "crypto" if sym in ("BTC", "ETH", "DOGE", "SOL") else "us-stock"

    all_sigs = []
    for sym in tickers:
        data = _analyze_chain(sym, markets.get(sym, "us-stock"))
        pc   = data.get("pc_ratio")
        gw   = data.get("gamma_wall")

        for s in data["signals"]:
            all_sigs.append({
                "source":     "options-flow",
                "symbol":     sym,
                "tickers":    [sym],
                "event_type": s["type"],
                "direction":  s["direction"],
                "sentiment":  s["score"],
                "urgency":    "HIGH" if abs(s["score"]) >= 0.60 else "MEDIUM",
                "headline":   s["detail"][:200],
                "pc_ratio":   pc,
                "gamma_wall": gw,
                "pub":        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
            })
        time.sleep(0.5)

    all_sigs.sort(key=lambda x: abs(x["sentiment"]), reverse=True)
    cache["options_signals"] = all_sigs
    _save_cache(cache)
    return all_sigs


def format_for_llm(signals, symbol):
    relevant = [s for s in signals if symbol in s.get("tickers", [])]
    if not relevant:
        return "No unusual options activity."
    lines = []
    first = relevant[0]
    if first.get("pc_ratio"):
        lines.append(f"P/C ratio: {first['pc_ratio']:.2f} | Gamma wall: ${first.get('gamma_wall','N/A')}")
    for s in relevant[:3]:
        lines.append(f"[{s['urgency']}] {s['event_type']} → {s['direction']}: {s['headline'][:120]}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    tickers = sys.argv[1:] or ["NVDA", "AAPL", "MSFT"]
    print(f"Options flow for {tickers}...\n")
    sigs = get_options_signals(watchlist_tickers=tickers, use_cache=False)
    if not sigs:
        print("No unusual activity detected.")
    for s in sigs:
        print(f"  {s['urgency']:6} | {s['event_type']:15} | {s['direction']:7} "
              f"| {s['symbol']:6} | {s['headline'][:80]}")
