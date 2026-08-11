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
  CATALYST CONTEXT (point-in-time, on by default; --no-catalysts to disable)
  SEC EDGAR filings ARE dated, so three of the brain's top-weighted catalysts
  are reconstructed leak-free by pinning the filing-date window to <= as-of:
    * 8-K   material events (news/M&A)   via news_catalyst._edgar_8k
    * 13D/G/13F institutional/activist   via whale_tracker._search_edgar_filings
    * Form 4 insider transactions        via EDGAR full-text search
  Options flow has NO free point-in-time history and stays EXCLUDED (a hard
  ceiling on this eval). Because catalysts name the real entity, they are
  INCOMPATIBLE with --anonymize (which would then leak the ticker); requesting
  both disables catalysts. VIX/fundamentals/social remain passed empty.

Usage:
  python3 eval_llm.py --symbols NVDA,AAPL --dates 8 --horizon 20 [--anonymize] [--no-catalysts] [--years 2]
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
from signals.news_catalyst import _edgar_8k          # date-pinnable (as_of=) — 8-K events
from signals.whale_tracker import _search_edgar_filings  # date-pinnable (as_of=) — 13D/G/F

import time
import requests

EDGAR_FTS = "https://efts.sec.gov/LATEST/search-index"
EDGAR_UA  = {"User-Agent": "ai-trader research@example.com"}


def _edgar_form4(symbol, as_of, lookback_days=30):
    """Point-in-time Form 4 (insider) filings via EDGAR full-text search.
    Returns [{entity, filed}] with filing date <= as_of (leak-safe)."""
    try:
        end_dt = datetime.strptime(as_of, "%Y-%m-%d")
        start  = (end_dt - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        resp = requests.get(EDGAR_FTS, headers=EDGAR_UA, timeout=10,
                            params={"q": f'"{symbol}"', "forms": "4",
                                    "dateRange": "custom", "startdt": start, "enddt": as_of})
        if not resp.ok:
            return []
        out = []
        for h in resp.json().get("hits", {}).get("hits", [])[:5]:
            src = h.get("_source", {})
            filed = src.get("file_date", "")
            if filed and filed > as_of:            # defensive: never future
                continue
            names = src.get("display_names") or ["?"]
            ent = names[0] if isinstance(names, list) and names else "?"
            out.append({"entity": str(ent)[:60], "filed": filed})
        return out
    except Exception:
        return []


def pit_catalysts(symbol, as_of, market):
    """Reconstruct point-in-time catalyst context for an as-of date using ONLY
    SEC EDGAR filings dated <= as_of. Returns (news_ctx, whale_ctx, insider_ctx,
    sources:set). Options flow is not reconstructable and is left empty."""
    if market != "us-stock":                       # EDGAR is US equities only
        return "", "", "", set()
    sources = set()

    # 8-K material events (news / M&A / exec change) in the prior 14 days
    news = [s for s in _edgar_8k(symbol, lookback_days=14, as_of=as_of)
            if not s.get("pub") or s["pub"] <= as_of]
    news_ctx = ""
    if news:
        sources.add("8-K")
        news_ctx = "SEC 8-K material events (<=14d): " + "; ".join(
            s["headline"] for s in news[:3])
    time.sleep(0.3)                                # EDGAR politeness (~10 req/s cap)

    # 13D/G activist + 13F institutional in the prior 90 days (activist first)
    wf = [f for f in _search_edgar_filings(symbol, days_back=90, as_of=as_of)
          if not f.get("filed") or f["filed"] <= as_of]
    whale_ctx = ""
    if wf:
        sources.add("13D/F")
        act  = [f for f in wf if "13D" in f["form"] or "13G" in f["form"]]
        show = (act or wf)[:3]
        whale_ctx = "SEC institutional filings (<=90d): " + "; ".join(
            f"{f['form']} {f['entity']} ({f['filed']})" for f in show)
    time.sleep(0.3)

    # Form 4 insider transactions in the prior 30 days
    f4 = _edgar_form4(symbol, as_of, lookback_days=30)
    insider_ctx = ""
    if f4:
        sources.add("Form4")
        insider_ctx = "SEC Form 4 insider (<=30d): " + "; ".join(
            f"{x['entity']} ({x['filed']})" for x in f4[:3])
    time.sleep(0.3)

    return news_ctx, whale_ctx, insider_ctx, sources


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
                years: int, anonymize: bool, cfg: dict,
                use_catalysts: bool = True) -> list:
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
        ind   = build_ind(ind_df, i)
        as_of = str(dates[i].date())
        news_ctx = whale_ctx = insider_ctx = ""
        cat_sources = set()
        if use_catalysts:                          # point-in-time EDGAR (real ticker)
            news_ctx, whale_ctx, insider_ctx, cat_sources = pit_catalysts(
                symbol, as_of, market)
        try:
            dec = debate_decide(label, mkt, ind, {}, {}, 100_000,
                                "", insider_ctx, "", cfg,
                                news_ctx=news_ctx, whale_ctx=whale_ctx)
        except Exception as e:
            print(f"  {symbol} @{as_of}: brain error: {e}")
            continue
        fwd   = float(ind_df["close"].iloc[i + horizon] / ind_df["close"].iloc[i] - 1) * 100
        gated = dec.get("consensus") == "SKIP"
        rows.append({
            "symbol": symbol, "date": as_of,
            "action": dec.get("action", "HOLD"), "conf": dec.get("confidence", 0),
            "fwd_ret": round(fwd, 2),
            "catalysts": sorted(cat_sources), "gated": gated,
        })
        cat_tag = ("[" + ",".join(sorted(cat_sources)) + "]") if cat_sources else ("[gated]" if gated else "[--]")
        print(f"  {symbol} @{as_of}  {rows[-1]['action']:5} "
              f"conf={rows[-1]['conf']:>3}%   fwd{horizon}d={fwd:+6.2f}%  {cat_tag}")
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
    ap.add_argument("--no-catalysts", dest="catalysts", action="store_false",
                    help="disable point-in-time EDGAR catalyst reconstruction (technicals only)")
    ap.add_argument("--market",    type=str, default="us-stock", choices=["us-stock", "crypto"])
    args = ap.parse_args()

    # Catalysts name the real entity -> incompatible with anonymize (would leak
    # the ticker). Anonymize wins; disable catalysts with a clear warning.
    if args.anonymize and args.catalysts:
        print("  ⚠  --anonymize + catalysts are incompatible (EDGAR filings name the "
              "issuer, leaking the ticker). Disabling catalysts for this run.")
        args.catalysts = False

    cfg = load_cfg()
    cats = "EDGAR 8-K + 13D/G/F + Form 4 (point-in-time); options-flow EXCLUDED" \
           if args.catalysts else "none (technicals only)"
    print(f"\n  LLM eval | model={cfg.get('_fast_model')} | "
          f"anonymize={args.anonymize} | horizon={args.horizon}d")
    print(f"  catalysts: {cats}")
    if not args.anonymize:
        print("  ⚠  look-ahead bias risk: pre-cutoff dates may be memorized. "
              "Prefer post-cutoff as-of dates (or --anonymize for a technicals-only clean read).")

    all_rows = []
    for sym in [s.strip() for s in args.symbols.split(",") if s.strip()]:
        print(f"\n── {sym} ──")
        all_rows += eval_symbol(sym, args.market, args.dates, args.horizon,
                                args.years, args.anonymize, cfg,
                                use_catalysts=args.catalysts)

    summary = report(all_rows, args.horizon)

    # Catalyst coverage — how much of the sample actually carried a real catalyst
    if args.catalysts and all_rows:
        with_cat = [r for r in all_rows if r.get("catalysts")]
        gated    = [r for r in all_rows if r.get("gated")]
        from collections import Counter
        src_counts = Counter(s for r in with_cat for s in r["catalysts"])
        print(f"  catalyst coverage: {len(with_cat)}/{len(all_rows)} decisions had "
              f"an EDGAR catalyst | {len(gated)} gated (no catalyst + flat) | "
              f"by source: {dict(src_counts)}")
        print(f"  (options-flow excluded — no free point-in-time history)")
        summary["catalyst_coverage"] = {
            "with_catalyst": len(with_cat), "gated": len(gated),
            "by_source": dict(src_counts), "options_flow": "excluded",
        }

    out = DIR / "llm_eval_report.json"
    out.write_text(json.dumps({"run_at": datetime.now().isoformat(),
                               "args": vars(args), "summary": summary,
                               "decisions": all_rows}, indent=2))
    print(f"Report saved → {out}")


if __name__ == "__main__":
    main()
