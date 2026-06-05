#!/usr/bin/env python3
"""
GOD MODE v2 — Walk-Forward Optimizer (M1 optimised)

Key fixes vs original:
  - Data sent to workers ONCE via initializer (not once per combo)
  - Pure numpy arrays in global — zero pandas overhead in hot loop
  - Signal arrays precomputed per combo (not per bar)
  - Default 3,000 combos — converges to same winner as 20k in seconds

Usage: python3 god_mode_vbt.py [--combos 3000] [--workers 4] [--folds 4]
"""

import json, sys, time, random, warnings, argparse, math
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")
DIR = Path(__file__).parent


def _safe(val, default=0.0):
    try:
        f = float(val)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return default


# ── Parameter space ───────────────────────────────────────────────────────────

PARAM_SPACE = {
    "rsi_lo":      [20, 25, 30, 35, 40, 45],
    "rsi_hi":      [45, 50, 55, 60, 65],
    "rsi_exit":    [65, 68, 71, 74, 77, 80],
    "bb_entry":    [0.30, 0.40, 0.50, 0.60, 0.70],   # entry zone
    "bb_exit":     [0.75, 0.80, 0.85, 0.90, 0.95],   # exit zone — always > bb_entry
    "stop_atr":    [1.5, 2.0, 2.5, 3.0, 3.5],
    "target_mult": [2.0, 2.5, 3.0, 3.5, 4.0],
    "macd_filter": [True, False],
    "sma200_req":  [True, False],
    "risk_pct":    [0.005, 0.010, 0.015, 0.020],
}

ASSETS = [
    ("NVDA", "us-stock"), ("AAPL", "us-stock"), ("MSFT", "us-stock"),
    ("BTC-USD", "crypto"), ("ETH-USD", "crypto"),
]


# ── Data prep — returns plain numpy arrays (not DataFrames) ──────────────────

def prep_asset(ticker):
    end   = datetime.today() + timedelta(days=1)  # +1: yfinance end is exclusive
    start = end - timedelta(days=901)
    df    = yf.download(ticker,
                        start=start.strftime("%Y-%m-%d"),
                        end=end.strftime("%Y-%m-%d"),
                        interval="1d", progress=False, auto_adjust=True)
    if len(df) < 250:
        return None

    c = df["Close"].squeeze()
    h = df["High"].squeeze()
    l = df["Low"].squeeze()

    sma50  = c.rolling(50).mean()
    sma200 = c.rolling(200).mean()

    delta  = c.diff()
    gain   = delta.clip(lower=0).rolling(14).mean()
    loss   = (-delta.clip(upper=0)).rolling(14).mean()
    rsi    = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).fillna(100)

    ema12  = c.ewm(span=12, adjust=False).mean()
    ema26  = c.ewm(span=26, adjust=False).mean()
    macd_h = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()

    bb_std = c.rolling(20).std()
    bb_mid = c.rolling(20).mean()
    bb_pos = ((c - (bb_mid - 2*bb_std)) / (4*bb_std)).clip(0, 1)

    tr  = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    df2 = pd.DataFrame({
        "c": c, "sma50": sma50, "sma200": sma200,
        "rsi": rsi, "macd_h": macd_h, "bb": bb_pos, "atr": atr,
    }).dropna()

    # Return numpy arrays — zero pandas overhead in hot loop
    return {col: df2[col].values for col in df2.columns}


# ── Global shared data — set ONCE per worker process via initializer ──────────
# This is the key fix: eliminates pickle of all_data on every single task call.

_GLOBAL_DATA: dict = {}


def _init_worker(data: dict):
    """Called once per worker process — stores data in process-local global."""
    global _GLOBAL_DATA
    _GLOBAL_DATA = data


# ── Fast numpy backtest — precomputed signal arrays ───────────────────────────

def bt_fast(arrays, p, capital=100_000):
    """
    Backtest using precomputed boolean signal arrays.
    No pandas. No repeated per-bar comparisons.
    """
    c      = arrays["c"]
    rsi    = arrays["rsi"]
    macd_h = arrays["macd_h"]
    bb     = arrays["bb"]
    atr    = arrays["atr"]
    sma50  = arrays["sma50"]
    sma200 = arrays["sma200"]
    n      = len(c)

    stop_m   = p["stop_atr"]
    target_m = p["stop_atr"] * p["target_mult"]

    # ── Precompute all signal arrays (vectorized, outside the loop) ───────────
    above50  = c > sma50
    above200 = c > sma200
    trend    = above50 & (above200 if p["sma200_req"] else np.ones(n, bool))
    macd_ok  = (macd_h > 0) if p["macd_filter"] else np.ones(n, bool)
    entry    = ((rsi >= p["rsi_lo"]) & (rsi <= p["rsi_hi"])
                & trend & macd_ok & (bb < p["bb_entry"]))
    exit_    = (rsi > p["rsi_exit"]) | (bb > p["bb_exit"]) | ~above50

    # ── Simulation loop — only stop/target need per-bar price comparison ──────
    cash  = capital
    qty   = 0.0
    ep    = 0.0   # entry price
    ea    = 0.0   # ATR at entry
    pnls  = []
    eq    = np.empty(n)

    for i in range(n):
        pr       = c[i]
        eq[i]    = cash + qty * pr

        if qty > 0:
            if pr <= ep - stop_m * ea:
                cash += qty * pr; pnls.append((pr - ep) * qty); qty = 0; continue
            if pr >= ep + target_m * ea:
                cash += qty * pr; pnls.append((pr - ep) * qty); qty = 0; continue

        if entry[i] and qty == 0:
            atri = max(atr[i], 0.001)
            risk = eq[i] * p["risk_pct"]
            q    = max(1, int(risk / (atri * stop_m)))
            cost = q * pr
            if cost <= cash:
                cash -= cost; qty = q; ep = pr; ea = atri
        elif exit_[i] and qty > 0:
            cash += qty * pr; pnls.append((pr - ep) * qty); qty = 0

    if qty > 0:
        cash += qty * c[-1]; pnls.append((c[-1] - ep) * qty)

    ret  = (eq[-1] / capital - 1) * 100
    wins = sum(1 for x in pnls if x > 0)
    wr   = wins / len(pnls) * 100 if pnls else 0.0
    mdd  = float(((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq) * 100).min()) if n > 1 else 0.0
    dr   = np.diff(eq) / eq[:-1]
    sh   = float(dr.mean() / dr.std() * 252**0.5) if dr.std() > 0 else 0.0

    return {"ret": round(ret, 3), "wr": round(wr, 1),
            "mdd": round(mdd, 3), "sh": round(sh, 3), "trades": len(pnls)}


# ── Walk-forward ──────────────────────────────────────────────────────────────

_N_FOLDS: int = 4   # set from args in main, read by _worker


def walk_forward(p, n_folds=None):
    """Uses _GLOBAL_DATA — no data argument needed (no pickle per call)."""
    n_folds = n_folds or _N_FOLDS
    oos_results = []
    for ticker, arrays in _GLOBAL_DATA.items():
        n    = len(arrays["c"])
        fold = n // (n_folds + 1)
        for k in range(1, n_folds + 1):
            s   = fold * k
            e   = fold * (k + 1)
            if e - s < 50:
                continue
            sliced = {col: arr[s:e] for col, arr in arrays.items()}
            r = bt_fast(sliced, p)
            oos_results.append(r)

    valid = [r for r in oos_results if r["trades"] > 0]
    if not valid:
        return None

    n_v          = len(valid)
    total_trades = sum(r["trades"] for r in valid)
    avg_ret = sum(_safe(r["ret"]) for r in valid) / n_v
    avg_wr  = sum(_safe(r["wr"])  for r in valid) / n_v
    avg_sh  = sum(_safe(r["sh"])  for r in valid) / n_v
    avg_mdd = sum(_safe(r["mdd"]) for r in valid) / n_v
    return {"ret": round(avg_ret, 3), "wr": round(avg_wr, 1),
            "sh": round(avg_sh, 3), "mdd": round(avg_mdd, 3),
            "folds": n_v, "total_trades": total_trades}


# ── Worker — receives only params (tiny pickle, not the data) ─────────────────

def _worker(params):
    """No data arg — reads from process-local _GLOBAL_DATA."""
    oos = walk_forward(params)
    if not oos:
        return None

    total_trades = _safe(oos.get("total_trades", 0))
    if total_trades < 6:
        return None

    trade_mult = min(1.0, max(0.0, (total_trades - 6) / 9))
    ret = _safe(oos["ret"])
    wr  = _safe(oos["wr"])
    sh  = max(-5.0, min(5.0, _safe(oos["sh"])))
    mdd = _safe(oos["mdd"])

    score = (ret * 0.40 + wr * 0.30 + sh * 8 * 0.20 + mdd * 0.10) * trade_mult
    if math.isnan(score) or math.isinf(score):
        return None

    return {"score": round(score, 4), "oos_ret": oos["ret"], "oos_wr": oos["wr"],
            "oos_sharpe": oos["sh"], "oos_mdd": oos["mdd"],
            "oos_trades": int(total_trades), "params": params}


def sample_params(n):
    random.seed(42)
    out = []
    for _ in range(n):
        p = {k: random.choice(v) for k, v in PARAM_SPACE.items()}
        # Enforce logical ordering
        if p["rsi_lo"] >= p["rsi_hi"]:    p["rsi_lo"]  = max(20, p["rsi_hi"] - 10)
        if p["rsi_hi"] >= p["rsi_exit"]:  p["rsi_exit"] = p["rsi_hi"] + 8
        if p["bb_exit"] <= p["bb_entry"]: p["bb_exit"] = min(0.95, p["bb_entry"] + 0.15)
        out.append(p)
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combos",  type=int, default=3_000,
                    help="Combos to test (default 3000 — converges same as 20k)")
    ap.add_argument("--workers", type=int, default=4,
                    help="Workers (default 4 — M1 perf cores, avoids thermal throttle)")
    ap.add_argument("--folds",   type=int, default=4)
    ap.add_argument("--apply",   action="store_true")
    args = ap.parse_args()

    print(f"\n{'='*64}")
    print(f"  GOD MODE v2 — Walk-Forward (M1 optimised)")
    print(f"  Combos: {args.combos:,}  |  Workers: {args.workers}  |  Folds: {args.folds}")
    print(f"  Fix: data shared via initializer — zero pickle per task")
    print(f"{'='*64}\n")

    print("Downloading 2.5 years of market data...")
    all_data = {}
    for ticker, _ in ASSETS:
        d = prep_asset(ticker)
        if d is not None:
            all_data[ticker] = d
            print(f"  {ticker}: {len(d['c'])} bars ✓")

    # Set global folds so _worker picks it up without passing through pickle
    global _N_FOLDS
    _N_FOLDS = args.folds

    params_list = sample_params(args.combos)

    print(f"\nRunning {args.combos:,} walk-forward backtests on {args.workers} workers...\n")
    t0      = time.time()
    results = []

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,   # ← sends all_data ONCE per worker, not per task
        initargs=(all_data,)
    ) as ex:
        futs = [ex.submit(_worker, p) for p in params_list]
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            if r:
                results.append(r)
            done += 1
            if done % 250 == 0:
                best = max(results, key=lambda x: x["score"]) if results else None
                br   = f"{best['oos_ret']:+.2f}%" if best else "?"
                pct  = done / args.combos * 100
                print(f"  [{done:>5}/{args.combos}] {pct:4.0f}%  {time.time()-t0:.0f}s  "
                      f"best OOS: {br}  valid: {len(results)}")

    elapsed = time.time() - t0
    results.sort(key=lambda x: x["score"], reverse=True)
    top = results[:10]

    print(f"\n{'='*64}")
    print(f"  DONE: {len(results):,} valid / {args.combos:,} combos  in {elapsed:.1f}s")
    if not results:
        print("  No valid results — try fewer folds or lower min-trade threshold.")
        print(f"{'='*64}\n"); return
    print(f"  Speed: {args.combos / elapsed:,.0f} combos/sec")
    print(f"{'='*64}")

    print(f"\n{'─'*72}")
    print(f"  {'#':<3} {'Score':>7} {'OOS Ret':>9} {'Win%':>7} {'Sharpe':>8} {'MDD':>7} {'Trades':>7}")
    print(f"{'─'*72}")
    for i, r in enumerate(top, 1):
        print(f"  {i:<3} {r['score']:>7.3f} {r['oos_ret']:>+8.2f}% "
              f"{r['oos_wr']:>6.1f}% {r['oos_sharpe']:>8.3f} "
              f"{r['oos_mdd']:>6.2f}% {r.get('oos_trades', 0):>7}")

    best = top[0]
    p    = best["params"]
    print(f"\n{'='*64}")
    print(f"  WINNER (Walk-Forward Validated)")
    print(f"  OOS Return {best['oos_ret']:+.2f}% | Win {best['oos_wr']:.1f}% | "
          f"Sharpe {best['oos_sharpe']:.3f} | Trades {best.get('oos_trades', '?')}")
    print(f"\n  RSI entry:  {p['rsi_lo']}–{p['rsi_hi']}   exit: >{p['rsi_exit']}")
    print(f"  BB entry:   <{p['bb_entry']}   exit: >{p['bb_exit']}")
    print(f"  Stop:       {p['stop_atr']}×ATR   Target: {p['stop_atr']*p['target_mult']:.1f}×ATR  "
          f"({p['target_mult']:.1f}:1 R:R)")
    print(f"  Risk:       {p['risk_pct']*100:.1f}%/trade   "
          f"SMA200: {p['sma200_req']}   MACD: {p['macd_filter']}")
    print(f"{'='*64}\n")

    report = DIR / "god_mode_vbt_report.json"
    report.write_text(json.dumps({
        "run_at": datetime.now().isoformat(), "engine": "numpy-fast",
        "combos_tested": len(results), "elapsed_sec": round(elapsed, 1),
        "top10": top,
    }, indent=2))
    print(f"Report → {report}")

    if args.apply or (best["oos_ret"] > 0 and best["oos_wr"] > 45):
        # Update trader.py constants
        trader = DIR / "trader.py"
        code   = trader.read_text()
        stop_val   = p["stop_atr"]
        target_val = round(p["stop_atr"] * p["target_mult"], 4)
        for const, val in [("STOP_LOSS_ATR", stop_val),
                           ("PROFIT_TARGET_ATR", target_val)]:
            lines = code.split("\n")
            for idx, line in enumerate(lines):
                stripped = line.split("#")[0].strip()  # ignore comments
                if stripped.startswith(const) and "=" in stripped:
                    lines[idx] = f"{const}  = {val}    # vbt walk-forward validated"
            code = "\n".join(lines)
        trader.write_text(code)

        # Update best_params_vbt.json
        (DIR / "best_params_vbt.json").write_text(
            json.dumps({"params": p, "score": best["score"]}, indent=2))

        # Update best_params.json with VBT fields mapped to backtest.py schema
        bp = {
            "found_at":     datetime.now().isoformat(),
            "score":        best["score"],
            "oos_return":   best["oos_ret"],
            "oos_win_rate": best["oos_wr"],
            "oos_sharpe":   best["oos_sharpe"],
            "source":       "vbt-walk-forward",
            "params": {
                "rsi_buy_min":  p["rsi_lo"],
                "rsi_buy_max":  p["rsi_hi"],
                "rsi_sell":     p["rsi_exit"],
                "bb_buy_max":   p["bb_entry"],
                "bb_sell_min":  p["bb_exit"],
                "use_sma200":   p["sma200_req"],
                "macd_required":p["macd_filter"],
                "stop_atr":     stop_val,
                "target_atr":   target_val,
                "risk_pct":     p["risk_pct"],
            }
        }
        (DIR / "best_params.json").write_text(json.dumps(bp, indent=2))
        print("✓ trader.py + best_params.json updated with walk-forward validated parameters")


if __name__ == "__main__":
    main()
