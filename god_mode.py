#!/usr/bin/env python3
"""
GOD MODE — Parallel Strategy Optimizer
8 CPU workers × 500+ parameter combos × all assets × 2 years
Finds the best trading configuration and auto-applies it to trader.py.

Usage: python3 god_mode.py [--workers N] [--minutes M]
"""

import json, os, sys, time, random, warnings, argparse
import numpy as np
import pandas as pd
import yfinance as yf
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from itertools import product

warnings.filterwarnings("ignore")
DIR = Path(__file__).parent

# ── Parameter space to search ─────────────────────────────────────────────────
PARAM_SPACE = {
    "rsi_buy_max":    [35, 40, 45, 50, 55, 60],       # RSI upper limit for BUY
    "rsi_buy_min":    [20, 25, 30],                    # RSI lower limit for BUY
    "rsi_sell":       [65, 68, 70, 72, 75, 78],        # RSI trigger for SELL
    "bb_buy_max":     [0.40, 0.50, 0.60, 0.70],        # BB position limit for BUY
    "bb_sell_min":    [0.75, 0.80, 0.85, 0.90],        # BB position trigger for SELL
    "use_sma200":     [True, False],                   # require SMA200 uptrend
    "macd_required":  [True, False],                   # require MACD hist positive for BUY
    "stop_atr":       [1.5, 2.0, 2.5, 3.0],            # stop loss multiplier
    "target_atr":     [3.0, 4.5, 6.0, 8.0],            # profit target multiplier
    "risk_pct":       [0.005, 0.010, 0.015, 0.020],    # portfolio risk per trade
}

ASSETS = [
    ("NVDA", "us-stock"), ("AAPL", "us-stock"), ("MSFT", "us-stock"),
    ("BTC-USD", "crypto"), ("ETH-USD", "crypto"),
]


# ── Data + indicators (module-level for multiprocessing pickling) ─────────────

def _compute(df):
    c = df["Close"].squeeze()
    h = df["High"].squeeze(); l = df["Low"].squeeze()
    sma20  = c.rolling(20).mean()
    sma50  = c.rolling(50).mean()
    sma200 = c.rolling(200).mean()
    delta  = c.diff()
    gain   = delta.clip(lower=0).rolling(14).mean()
    loss   = (-delta.clip(upper=0)).rolling(14).mean()
    rsi    = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).fillna(100)
    ema12  = c.ewm(span=12, adjust=False).mean()
    ema26  = c.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    macd_h = macd - macd.ewm(span=9, adjust=False).mean()
    bb_std = c.rolling(20).std()
    bb_mid = sma20
    bb_pos = ((c - (bb_mid - 2*bb_std)) / (4*bb_std)).clip(0, 1)
    tr     = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr    = tr.rolling(14).mean()
    return pd.DataFrame({
        "c": c, "sma50": sma50, "sma200": sma200,
        "rsi": rsi, "macd_h": macd_h, "bb": bb_pos, "atr": atr
    }).dropna()


def _run_one(params, ind, capital=100_000):
    """Simulate one parameter set on pre-computed indicator DataFrame."""
    cash    = capital
    qty     = 0.0
    in_pos  = False
    ep      = 0.0   # entry price
    ea      = 0.0   # entry atr
    trades  = []
    equity  = []

    stop_m   = params["stop_atr"]
    target_m = params["target_atr"]
    risk_pct = params["risk_pct"]

    for _, row in ind.iterrows():
        price = row["c"]
        atr   = max(row["atr"], 0.001)
        val   = cash + qty * price
        equity.append(val)

        # Auto stop / target
        if in_pos:
            if price <= ep - stop_m * ea:
                cash += qty * price
                pnl  = (price - ep) * qty
                trades.append({"pnl": pnl, "outcome": "STOP"})
                qty = 0; in_pos = False; continue
            if price >= ep + target_m * ea:
                cash += qty * price
                pnl  = (price - ep) * qty
                trades.append({"pnl": pnl, "outcome": "TARGET"})
                qty = 0; in_pos = False; continue

        # Strategy signal
        above50  = price > row["sma50"]
        above200 = price > row["sma200"]
        macd_ok  = row["macd_h"] > 0
        rsi_buy  = params["rsi_buy_min"] <= row["rsi"] <= params["rsi_buy_max"]
        bb_buy   = row["bb"] < params["bb_buy_max"]
        rsi_sell = row["rsi"] > params["rsi_sell"]
        bb_sell  = row["bb"] > params["bb_sell_min"]

        trend_ok = above50 and (above200 if params["use_sma200"] else True)
        macd_gd  = macd_ok if params["macd_required"] else True

        if not in_pos and rsi_buy and trend_ok and macd_gd and bb_buy:
            risk_amt = val * risk_pct
            stop_d   = atr * stop_m
            q        = max(1, int(risk_amt / stop_d))
            cost     = q * price
            if cost <= cash:
                cash -= cost; qty = q; in_pos = True; ep = price; ea = atr
        elif in_pos and (rsi_sell or bb_sell or not above50):
            cash += qty * price
            pnl   = (price - ep) * qty
            trades.append({"pnl": pnl, "outcome": "SIGNAL"})
            qty = 0; in_pos = False

    # Close open position
    if in_pos:
        last = ind["c"].iloc[-1]
        cash += qty * last
        trades.append({"pnl": (last - ep) * qty, "outcome": "EOD"})

    if not equity:
        return None

    eq   = pd.Series(equity)
    ret  = (eq.iloc[-1] / capital - 1) * 100
    wins = [t for t in trades if t["pnl"] > 0]
    wr   = len(wins) / len(trades) * 100 if trades else 0
    mdd  = ((eq - eq.cummax()) / eq.cummax() * 100).min()
    dr   = eq.pct_change().dropna()
    sh   = (dr.mean() / dr.std() * (252**0.5)) if dr.std() > 0 else 0

    return {"return": round(ret, 3), "win_rate": round(wr, 1),
            "trades": len(trades), "mdd": round(mdd, 3), "sharpe": round(sh, 3),
            "final": round(eq.iloc[-1], 2)}


def worker(task):
    """Single worker: test one param set on all assets."""
    params, data_dict = task
    results = []
    for (ticker, _), ind in data_dict.items():
        r = _run_one(params, ind)
        if r:
            results.append(r)
    if not results:
        return None, params
    avg_ret = sum(r["return"]   for r in results) / len(results)
    avg_wr  = sum(r["win_rate"] for r in results) / len(results)
    avg_sh  = sum(r["sharpe"]   for r in results) / len(results)
    avg_mdd = sum(r["mdd"]      for r in results) / len(results)
    total_t = sum(r["trades"]   for r in results)
    score   = avg_ret * 0.4 + avg_wr * 0.3 + avg_sh * 10 * 0.3  # composite
    return {
        "score": round(score, 4), "avg_return": round(avg_ret, 3),
        "avg_win_rate": round(avg_wr, 1), "avg_sharpe": round(avg_sh, 3),
        "avg_mdd": round(avg_mdd, 3), "total_trades": total_t,
        "params": params,
    }, params


# ── download all data once ────────────────────────────────────────────────────

def download_all():
    print("Downloading 2 years of market data for all assets...")
    end   = datetime.today() + timedelta(days=1)  # +1 so today's data is included
    start = end - timedelta(days=731)
    data  = {}
    for ticker, market in ASSETS:
        sym = f"{ticker}" if "-USD" in ticker else ticker
        df  = yf.download(sym, start=start.strftime("%Y-%m-%d"),
                          end=end.strftime("%Y-%m-%d"), interval="1d",
                          progress=False, auto_adjust=True)
        if len(df) >= 210:
            data[(ticker, market)] = _compute(df)
            print(f"  {ticker}: {len(df)} days ✓")
        else:
            print(f"  {ticker}: insufficient data, skipped")
    return data


# ── random parameter sampler ──────────────────────────────────────────────────

def sample_params(n, seed=42):
    random.seed(seed)
    samples = []
    for _ in range(n):
        p = {k: random.choice(v) for k, v in PARAM_SPACE.items()}
        # Sanity: rsi_buy_min < rsi_buy_max < rsi_sell
        if p["rsi_buy_min"] >= p["rsi_buy_max"]:
            p["rsi_buy_min"] = p["rsi_buy_max"] - 10
        if p["rsi_buy_max"] >= p["rsi_sell"]:
            p["rsi_sell"] = p["rsi_buy_max"] + 10
        # target > stop (reward > risk)
        if p["target_atr"] <= p["stop_atr"]:
            p["target_atr"] = p["stop_atr"] * 2
        samples.append(p)
    return samples


# ── auto-apply best params to trader.py ──────────────────────────────────────

def apply_to_trader(best):
    """Write best parameters as constants into trader.py."""
    trader = DIR / "trader.py"
    code   = trader.read_text()
    p      = best["params"]

    replacements = {
        "MIN_CONFIDENCE":      "70",   # keep
        "STOP_LOSS_ATR":       str(p["stop_atr"]),
        "PROFIT_TARGET_ATR":   str(p["target_atr"]),
    }

    lines = code.split("\n")
    for i, line in enumerate(lines):
        for const, val in replacements.items():
            if line.startswith(f"{const} ") and "=" in line:
                lines[i] = f"{const}  = {val}"
    trader.write_text("\n".join(lines))

    # Save best params record
    record = DIR / "best_params.json"
    record.write_text(json.dumps({
        "found_at": datetime.now().isoformat(),
        "score": best["score"],
        "avg_return": best["avg_return"],
        "avg_win_rate": best["avg_win_rate"],
        "avg_sharpe": best["avg_sharpe"],
        "params": p,
    }, indent=2))
    print(f"\nBest params saved → {record}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers",  type=int, default=os.cpu_count(), help="Parallel workers")
    parser.add_argument("--combos",   type=int, default=500,            help="Parameter combinations to test")
    parser.add_argument("--minutes",  type=int, default=60,             help="Max runtime in minutes")
    parser.add_argument("--apply",    action="store_true",              help="Auto-apply best params to trader.py")
    args = parser.parse_args()

    deadline = time.time() + args.minutes * 60

    print(f"\n{'='*66}")
    print(f"  GOD MODE — Parallel Strategy Optimizer")
    print(f"  Workers: {args.workers}  |  Combos: {args.combos}  |  Time: {args.minutes}min")
    print(f"{'='*66}\n")

    # Download data once — shared across all workers via closure
    data_dict = download_all()
    if not data_dict:
        sys.exit("[error] No market data downloaded")

    params_list = sample_params(args.combos)
    tasks       = [(p, data_dict) for p in params_list]

    results      = []
    completed    = 0
    start_time   = time.time()

    print(f"\nLaunching {args.workers} parallel workers on {len(tasks)} combos...\n")

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(worker, t): t for t in tasks}

        for fut in as_completed(futures):
            if time.time() > deadline:
                print(f"\n[Time limit reached — {completed} combos tested]")
                ex.shutdown(wait=False, cancel_futures=True)
                break

            result, _ = fut.result()
            if result:
                results.append(result)

            completed += 1
            if completed % 25 == 0:
                elapsed = time.time() - start_time
                best_so_far = max(results, key=lambda x: x["score"]) if results else None
                best_ret = f"{best_so_far['avg_return']:+.2f}%" if best_so_far else "?"
                print(f"  [{completed}/{len(tasks)}] {elapsed:.0f}s elapsed | "
                      f"Best return so far: {best_ret}")

    if not results:
        print("No results generated.")
        return

    # Sort by composite score
    results.sort(key=lambda x: x["score"], reverse=True)
    top10 = results[:10]

    elapsed = time.time() - start_time
    print(f"\n{'='*66}")
    print(f"  COMPLETED: {completed} combos in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Valid results: {len(results)}")
    print(f"{'='*66}")

    print(f"\n{'─'*66}")
    print(f"  TOP 10 PARAMETER CONFIGURATIONS")
    print(f"{'─'*66}")
    print(f"  {'#':<3} {'Score':>7} {'Ret':>7} {'Win%':>6} {'Sharpe':>8} {'MDD':>7} {'Trades':>7}")
    print(f"{'─'*66}")
    for i, r in enumerate(top10, 1):
        print(f"  {i:<3} {r['score']:>7.3f} {r['avg_return']:>+6.2f}% "
              f"{r['avg_win_rate']:>5.1f}% {r['avg_sharpe']:>8.3f} "
              f"{r['avg_mdd']:>6.2f}% {r['total_trades']:>7}")

    best = top10[0]
    p    = best["params"]
    print(f"\n{'='*66}")
    print(f"  WINNER — Score {best['score']:.3f}")
    print(f"  Return {best['avg_return']:+.2f}% | Win {best['avg_win_rate']:.1f}% | "
          f"Sharpe {best['avg_sharpe']:.3f} | MDD {best['avg_mdd']:.2f}%")
    print(f"\n  Best Parameters:")
    print(f"    RSI entry:      {p['rsi_buy_min']}–{p['rsi_buy_max']}")
    print(f"    RSI exit:       >{p['rsi_sell']}")
    print(f"    BB entry max:   {p['bb_buy_max']}")
    print(f"    BB exit min:    {p['bb_sell_min']}")
    print(f"    Stop loss:      {p['stop_atr']}×ATR")
    print(f"    Profit target:  {p['target_atr']}×ATR  ({p['target_atr']/p['stop_atr']:.1f}:1 R:R)")
    print(f"    Risk per trade: {p['risk_pct']*100:.1f}% of portfolio")
    print(f"    Require SMA200: {p['use_sma200']}")
    print(f"    Require MACD+:  {p['macd_required']}")
    print(f"{'='*66}\n")

    # Save full results
    report = DIR / "god_mode_report.json"
    report.write_text(json.dumps({
        "run_at": datetime.now().isoformat(),
        "combos_tested": completed,
        "elapsed_sec": round(elapsed, 1),
        "top10": top10,
        "all_results": results,
    }, indent=2))
    print(f"Full report → {report}")

    if args.apply or (best["avg_return"] > 0 and best["avg_win_rate"] > 40):
        print("\nAuto-applying best params to trader.py...")
        apply_to_trader(best)
        print("✓ trader.py updated with optimised parameters")
    else:
        print("\nNote: avg return is negative — NOT auto-applying to trader.py")
        print("      Run with --apply to force it anyway")

    return best


if __name__ == "__main__":
    main()
