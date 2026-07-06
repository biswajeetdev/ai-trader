#!/usr/bin/env python3
"""
AI-Trader LLM Signal Eval  —  offline, point-in-time evaluation of the debate brain.

WHY THIS EXISTS
  backtest.py / god_mode.py validate only the *rule* layer. The LLM debate layer
  (signals/debate_brain.py) — the bot's claimed source of alpha — is otherwise
  scored only by live online loops (rag/strategy_evolver, self_improver). This
  harness gives it an offline test: replay historical as-of dates, ask the real
  debate brain to decide using ONLY point-in-time price/technical context, then
  score each call against the realized forward return.

WHAT IT MEASURES
  For each (symbol, as-of date): the brain's action (BUY/SELL/SHORT/HOLD) and
  confidence vs. the H-day forward return. Reports directional hit-rate, average
  forward return per action bucket, and EDGE = avg fwd-return on BUY calls minus
  the unconditional baseline (does the brain's timing beat buy-and-hold?).

LOOK-AHEAD BIAS — READ THIS
  An LLM trained through some cutoff can *recall* a well-known stock's actual
  trajectory, inflating results on pre-cutoff dates (see arXiv 2510.07920
  "Profit Mirage" and 2601.13770 "Look-Ahead-Bench"). Mitigations here:
    * --anonymize masks the ticker so the model can't identify the asset.
    * Prefer as-of dates AFTER the model's training cutoff for a clean read.
  Only point-in-time PRICE/TECHNICAL context is fed; live-only alt-data
  (social, Congress, options flow, VIX, fundamentals) is NOT reconstructed and
  is passed empty — so this isolates the brain's edge on technicals alone.

Usage:
  python3 eval_llm.py --symbols NVDA,AAPL --dates 8 --horizon 20 [--anonymize] [--years 2]
"""

import sys, os, json, argparse, warnings, subprocess
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
DIR = Path(__file__).parent

from backtest import compute_all                      # point-in-time indicators (rolling = backward-only)
from data.history import get_daily                     # resilient OHLCV: cache -> yfinance -> Alpaca
from signals.debate_brain import debate_decide


def load_cfg() -> dict:
    """Load config.json from the worktree or the canonical ~/ai-trader checkout."""
    for p in (DIR / "config.json", Path.home() / "ai-trader" / "config.json"):
        if p.exists():
            cfg = json.loads(p.read_text())
            break
    else:
        cfg = {}
    cfg.setdefault("_fast_model", "gpt-4o-mini")
    # mirror run.sh: make GitHub Models reachable even under a stripped env
    if not os.environ.get("GITHUB_TOKEN"):
        try:
            os.environ["GITHUB_TOKEN"] = subprocess.check_output(
                ["gh", "auth", "token"], text=True).strip()
        except Exception:
            pass
    return cfg


def build_ind(ind_df: pd.DataFrame, i: int) -> dict:
    """Map compute_all() row i to the dict shape debate_decide expects (point-in-time)."""
    row   = ind_df.iloc[i]
    close = float(row["close"])
    prev1 = float(ind_df["close"].iloc[i - 1]) if i >= 1 else close
    prev5 = float(ind_df["close"].iloc[i - 5]) if i >= 5 else close
    return {
        "price":       round(close, 4),
        "rsi14":       round(float(row["rsi"]), 2),
        "macd_hist":   round(float(row["macd_h"]), 4),
        "bb_position": float(row["bb_pos"]) if not pd.isna(row["bb_pos"]) else 0.5,
        "sma50":       round(float(row["sma50"]), 4),
        "above_sma50": bool(close > float(row["sma50"])),
        "atr14":       round(float(row["atr"]), 4),
        "pct_1d":      round((close / prev1 - 1) * 100, 2),
        "pct_5d":      round((close / prev5 - 1) * 100, 2),
    }


def eval_symbol(symbol: str, market: str, n_dates: int, horizon: int,
                years: int, anonymize: bool, cfg: dict) -> list:
    ticker = f"{symbol}-USD" if market == "crypto" else symbol
    end = datetime.today() + timedelta(days=1)
    df  = get_daily(ticker, (end - timedelta(days=365 * years + 1)).strftime("%Y-%m-%d"),
                    end.strftime("%Y-%m-%d"))
    if df.empty or len(df) < 220:
        print(f"  {symbol}: insufficient history — skipped")
        return []

    ind_df = compute_all(df).dropna(subset=["sma50", "rsi", "atr"])  # keep DatetimeIndex
    # only sample dates that leave `horizon` future bars for scoring
    last_scorable = len(ind_df) - horizon - 1
    first         = 60                                   # warmup past indicator settle
    if last_scorable <= first:
        print(f"  {symbol}: not enough scorable window — skipped")
        return []
    idxs  = [int(first + k * (last_scorable - first) / (n_dates - 1)) for k in range(n_dates)] \
            if n_dates > 1 else [last_scorable]
    dates = ind_df.index                                 # positional iloc ↔ date alignment
    rows  = []
    label = "ASSET" if anonymize else symbol
    mkt   = market

    for i in idxs:
        ind = build_ind(ind_df, i)
        try:
            dec = debate_decide(label, mkt, ind, {}, {}, 100_000, "", "", "", cfg)
        except Exception as e:
            print(f"  {symbol} @{dates[i].date()}: brain error: {e}")
            continue
        fwd = float(ind_df["close"].iloc[i + horizon] / ind_df["close"].iloc[i] - 1) * 100
        rows.append({
            "symbol": symbol, "date": str(dates[i].date()),
            "action": dec.get("action", "HOLD"), "conf": dec.get("confidence", 0),
            "fwd_ret": round(fwd, 2),
        })
        print(f"  {symbol} @{dates[i].date()}  {rows[-1]['action']:5} "
              f"conf={rows[-1]['conf']:>3}%   fwd{horizon}d={fwd:+6.2f}%")
    return rows


def report(rows: list, horizon: int) -> dict:
    if not rows:
        print("\nNo decisions collected."); return {}
    base = sum(r["fwd_ret"] for r in rows) / len(rows)        # unconditional avg fwd return

    def hit(r):
        a = r["action"]
        if a == "BUY":            return r["fwd_ret"] > 0
        if a in ("SELL", "SHORT"): return r["fwd_ret"] < 0
        return None                                            # HOLD: not directional

    directional = [r for r in rows if hit(r) is not None]
    hits        = [r for r in directional if hit(r)]
    buys        = [r for r in rows if r["action"] == "BUY"]
    edge        = (sum(r["fwd_ret"] for r in buys) / len(buys) - base) if buys else 0.0

    print(f"\n{'='*60}")
    print(f"  LLM SIGNAL EVAL  —  {len(rows)} decisions, {horizon}-day horizon")
    print(f"{'='*60}")
    by = {}
    for r in rows:
        by.setdefault(r["action"], []).append(r["fwd_ret"])
    for a, v in sorted(by.items()):
        print(f"  {a:6} n={len(v):>3}   avg fwd-ret {sum(v)/len(v):+6.2f}%")
    print(f"  {'-'*40}")
    print(f"  baseline (all dates) avg fwd-ret : {base:+6.2f}%")
    print(f"  BUY edge vs baseline             : {edge:+6.2f}%")
    if directional:
        print(f"  directional hit-rate             : {len(hits)/len(directional)*100:5.1f}%"
              f"  ({len(hits)}/{len(directional)})")
    verdict = ("LLM adds timing edge" if edge > 0.5 else
               "no clear edge" if edge > -0.5 else "LLM timing hurts")
    print(f"  VERDICT: {verdict}")
    print(f"{'='*60}\n")
    return {"n": len(rows), "baseline": round(base, 2), "buy_edge": round(edge, 2),
            "hit_rate": round(len(hits)/len(directional)*100, 1) if directional else None,
            "verdict": verdict}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols",   type=str, default="NVDA,AAPL,MSFT")
    ap.add_argument("--dates",     type=int, default=8,  help="as-of dates sampled per symbol")
    ap.add_argument("--horizon",   type=int, default=20, help="forward-return scoring window (trading days)")
    ap.add_argument("--years",     type=int, default=2)
    ap.add_argument("--anonymize", action="store_true", help="mask ticker to reduce LLM look-ahead memorization")
    ap.add_argument("--market",    type=str, default="us-stock", choices=["us-stock", "crypto"])
    args = ap.parse_args()

    cfg = load_cfg()
    print(f"\n  LLM eval | model={cfg.get('_fast_model')} | "
          f"anonymize={args.anonymize} | horizon={args.horizon}d")
    if not args.anonymize:
        print("  ⚠  look-ahead bias risk: pre-cutoff dates may be memorized. Use --anonymize for a clean read.")

    all_rows = []
    for sym in [s.strip() for s in args.symbols.split(",") if s.strip()]:
        print(f"\n── {sym} ──")
        all_rows += eval_symbol(sym, args.market, args.dates, args.horizon,
                                args.years, args.anonymize, cfg)

    summary = report(all_rows, args.horizon)
    out = DIR / "llm_eval_report.json"
    out.write_text(json.dumps({"run_at": datetime.now().isoformat(),
                               "args": vars(args), "summary": summary,
                               "decisions": all_rows}, indent=2))
    print(f"Report saved → {out}")


if __name__ == "__main__":
    main()
