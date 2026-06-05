#!/usr/bin/env python3
"""
AI-Trader Backtester v2
- Loads best_params.json for optimized RSI/BB/ATR params (auto-updates from god_mode)
- Regime-aware entry: trending markets allow higher RSI; mean-revert requires strict pullback
- Crypto-specific params: deeper RSI dips (20-45), no SMA50 requirement
- Metrics: return, win%, Sharpe, Calmar ratio, profit factor, avg hold days
- Lookahead-bias fix: signals on bar T, fills on bar T+1 open
- Commission: 0.1% round-trip

Usage: python3 backtest.py [--years 2] [--capital 100000] [--params best_params.json]
"""

import sys, json, argparse, warnings
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")
DIR = Path(__file__).parent
COMMISSION = 0.0005   # 0.05% per side


# ── Load optimized params (fallback to sensible defaults) ────────────────────

def load_params(path=None):
    p = DIR / (path or "best_params.json")
    if p.exists():
        data = json.loads(p.read_text())
        return data.get("params", data)
    return {
        "rsi_buy_min": 25, "rsi_buy_max": 60, "rsi_sell": 75,
        "bb_buy_max": 0.65, "bb_sell_min": 0.82,
        "stop_atr": 3.0, "target_atr": 6.0, "risk_pct": 0.015,
        "use_sma200": True, "macd_required": True,
    }


# ── Indicators ────────────────────────────────────────────────────────────────

def compute_all(df):
    close  = df["Close"].squeeze()
    open_  = df["Open"].squeeze()
    high   = df["High"].squeeze()
    low    = df["Low"].squeeze()
    vol    = df["Volume"].squeeze() if "Volume" in df.columns else pd.Series(dtype=float)

    sma20  = close.rolling(20).mean()
    sma50  = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()

    # SMA50 slope: positive = trending up over last 5 bars
    sma50_slope = sma50.diff(5)

    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).fillna(100)

    # RSI direction: 3-bar change (rising = momentum turning up)
    rsi_dir = rsi.diff(3)

    ema12  = close.ewm(span=12, adjust=False).mean()
    ema26  = close.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    sig    = macd.ewm(span=9, adjust=False).mean()
    macd_h = macd - sig

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_up  = bb_mid + 2 * bb_std
    bb_lo  = bb_mid - 2 * bb_std
    bb_pos = (close - bb_lo) / (bb_up - bb_lo).replace(0, np.nan)

    tr  = pd.concat([high - low,
                     (high - close.shift()).abs(),
                     (low  - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    vol_ratio = None
    if len(vol) == len(close):
        avg_vol   = vol.rolling(20).mean()
        vol_ratio = (vol / avg_vol.replace(0, np.nan)).fillna(1.0)
    else:
        vol_ratio = pd.Series(1.0, index=close.index)

    return pd.DataFrame({
        "close":       close,
        "open":        open_,
        "sma20":       sma20,
        "sma50":       sma50,
        "sma200":      sma200,
        "sma50_slope": sma50_slope,
        "rsi":         rsi,
        "rsi_dir":     rsi_dir,
        "macd_h":      macd_h,
        "bb_pos":      bb_pos,
        "atr":         atr,
        "vol_ratio":   vol_ratio,
    })


# ── Regime-aware signal ───────────────────────────────────────────────────────

def signal(row, p, market):
    """
    Three entry modes:
      TREND  — SMA50 rising + price > SMA50: buy momentum dips (RSI 40-70)
      PULLBACK — SMA50 flat/rising + classic dip: RSI rsi_lo-rsi_hi + MACD+ + BB<max
      CRYPTO — no SMA50 req, deeper RSI dips (20-45), MACD turning

    Exit: RSI > rsi_sell OR BB > bb_sell OR below SMA50 with MACD negative
    """
    rsi     = row["rsi"]
    macd_h  = row["macd_h"]
    bb_pos  = row["bb_pos"] if not pd.isna(row["bb_pos"]) else 0.5
    above50 = row["close"] > row["sma50"]
    above200 = row["close"] > row["sma200"] if not pd.isna(row["sma200"]) else True
    trend   = row["sma50_slope"] > 0        # SMA50 pointing up
    rsi_up  = row["rsi_dir"] > 0            # RSI rising (momentum turning)
    vol_ok  = row["vol_ratio"] >= 0.7       # not low-volume noise

    rsi_lo  = p.get("rsi_buy_min", 25)
    rsi_hi  = p.get("rsi_buy_max", 60)
    rsi_sel = p.get("rsi_sell", 75)
    bb_max  = p.get("bb_buy_max", 0.65)
    bb_sel  = p.get("bb_sell_min", 0.82)
    macd_ok = p.get("macd_required", True)
    sma200_req = p.get("use_sma200", True)

    if pd.isna(rsi) or pd.isna(macd_h):
        return "HOLD"

    # ── Crypto: separate logic ────────────────────────────────────────────────
    if market == "crypto":
        buy  = (20 <= rsi <= 45) and rsi_up and bb_pos < 0.38 and macd_h > -0.005
        sell = rsi > rsi_sel or bb_pos > bb_sel
        if buy:  return "BUY"
        if sell: return "SELL"
        return "HOLD"

    # ── US stocks: regime-aware ───────────────────────────────────────────────
    sma200_ok = above200 if sma200_req else True

    if trend and above50 and sma200_ok:
        # TREND mode: buy momentum dips up to RSI 68, relax BB to 0.72
        buy = (max(rsi_lo, 38) <= rsi <= min(rsi_hi + 8, 68)
               and bb_pos < min(bb_max + 0.07, 0.72)
               and vol_ok)
    else:
        # PULLBACK mode: strict filters, momentum turning required
        buy = (rsi_lo <= rsi <= rsi_hi
               and above50
               and (macd_h > 0 if macd_ok else True)
               and bb_pos < bb_max
               and rsi_up
               and vol_ok)

    sell = (rsi > rsi_sel
            or bb_pos > bb_sel
            or (not above50 and macd_h < 0))

    if buy and not sell: return "BUY"
    if sell:             return "SELL"
    return "HOLD"


# ── Per-asset simulation ──────────────────────────────────────────────────────

def run_asset(symbol, market, start, end, capital, params):
    ticker = f"{symbol}-USD" if market == "crypto" else symbol
    df     = yf.download(ticker, start=start, end=end,
                         interval="1d", progress=False, auto_adjust=True)
    if df.empty or len(df) < 80:
        return None

    ind = compute_all(df).dropna(subset=["sma50", "rsi", "atr"])

    stop_m   = params.get("stop_atr", 3.0)
    target_m = params.get("target_atr", 6.0)
    risk_pct = params.get("risk_pct", 0.015)

    # Lookahead fix: signal from bar T, execute at bar T+1 open
    ind["sig"] = ind.apply(lambda r: signal(r, params, market), axis=1)
    ind["sig"] = ind["sig"].shift(1)

    cash   = capital
    shares = 0.0
    ep     = 0.0         # entry price
    ea     = 0.0         # ATR at entry
    stop_p = 0.0
    tgt_p  = 0.0
    entry_date = None
    trades = []
    equity = []

    for date, row in ind.iterrows():
        close   = row["close"]
        fill    = row["open"]
        atr     = max(row["atr"], close * 0.005)
        sig     = row["sig"]
        value   = cash + shares * close
        equity.append({"date": date, "value": value})

        if pd.isna(fill) or fill <= 0 or pd.isna(sig):
            continue

        # ── Auto stop-loss / profit target ───────────────────────────────────
        if shares > 0:
            if close <= stop_p:
                pnl = (close - ep) * shares - close * shares * COMMISSION
                cash += shares * close * (1 - COMMISSION)
                days = (date - entry_date).days if entry_date else 0
                trades.append({"date": str(date.date()), "action": "STOP",
                               "price": round(close, 4), "qty": shares,
                               "pnl": round(pnl, 2), "hold_days": days})
                shares = 0; ep = ea = stop_p = tgt_p = 0; entry_date = None
                continue
            if close >= tgt_p:
                pnl = (close - ep) * shares - close * shares * COMMISSION
                cash += shares * close * (1 - COMMISSION)
                days = (date - entry_date).days if entry_date else 0
                trades.append({"date": str(date.date()), "action": "TARGET",
                               "price": round(close, 4), "qty": shares,
                               "pnl": round(pnl, 2), "hold_days": days})
                shares = 0; ep = ea = stop_p = tgt_p = 0; entry_date = None
                continue

        # ── Signal entry/exit ─────────────────────────────────────────────────
        if sig == "BUY" and shares == 0 and cash > fill:
            risk_amt = value * risk_pct
            stop_d   = atr * stop_m
            qty      = min(risk_amt / stop_d, 5_000 / fill)
            qty      = int(qty) if market != "crypto" else round(qty, 6)
            cost     = qty * fill * (1 + COMMISSION)
            if cost <= cash and qty > 0:
                cash -= cost
                shares = qty; ep = fill; ea = atr
                stop_p = round(ep - stop_m * ea, 4)
                tgt_p  = round(ep + target_m * ea, 4)
                entry_date = date
                trades.append({"date": str(date.date()), "action": "BUY",
                               "price": round(fill, 4), "qty": qty,
                               "stop": stop_p, "target": tgt_p})

        elif sig == "SELL" and shares > 0:
            pnl = (fill - ep) * shares - fill * shares * COMMISSION
            cash += shares * fill * (1 - COMMISSION)
            days = (date - entry_date).days if entry_date else 0
            trades.append({"date": str(date.date()), "action": "SELL",
                           "price": round(fill, 4), "qty": shares,
                           "pnl": round(pnl, 2), "hold_days": days})
            shares = 0; ep = ea = stop_p = tgt_p = 0; entry_date = None

    # Close any open position at last close
    if shares > 0:
        last = ind["close"].iloc[-1]
        pnl  = (last - ep) * shares - last * shares * COMMISSION
        cash += shares * last * (1 - COMMISSION)
        days = (ind.index[-1] - entry_date).days if entry_date else 0
        trades.append({"date": str(ind.index[-1].date()), "action": "SELL(EOD)",
                       "price": round(last, 4), "qty": shares,
                       "pnl": round(pnl, 2), "hold_days": days})

    # ── Metrics ───────────────────────────────────────────────────────────────
    eq = pd.Series([e["value"] for e in equity],
                   index=[e["date"] for e in equity])
    total_ret = (eq.iloc[-1] / capital - 1) * 100
    closed    = [t for t in trades if "pnl" in t]
    wins      = [t for t in closed if t["pnl"] > 0]
    losses    = [t for t in closed if t["pnl"] <= 0]
    win_rate  = len(wins) / len(closed) * 100 if closed else 0

    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss   = abs(sum(t["pnl"] for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    roll_max  = eq.cummax()
    drawdown  = ((eq - roll_max) / roll_max * 100).min()
    calmar    = -total_ret / drawdown if drawdown < 0 else 0

    daily_ret = eq.pct_change().dropna()
    sharpe    = (daily_ret.mean() / daily_ret.std() * 252 ** 0.5
                 if daily_ret.std() > 0 else 0)

    hold_days = [t["hold_days"] for t in closed if "hold_days" in t]
    avg_hold  = round(sum(hold_days) / len(hold_days), 1) if hold_days else 0

    return {
        "symbol":         symbol,
        "market":         market,
        "trades":         len(closed),
        "win_rate":       round(win_rate, 1),
        "total_return":   round(total_ret, 2),
        "max_drawdown":   round(drawdown, 2),
        "sharpe":         round(sharpe, 2),
        "calmar":         round(calmar, 2),
        "profit_factor":  round(profit_factor, 2),
        "avg_hold_days":  avg_hold,
        "final_value":    round(eq.iloc[-1], 2),
        "trade_log":      trades,
        "equity_series":  eq.to_dict(),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years",       type=int,   default=2)
    ap.add_argument("--capital",     type=float, default=100_000)
    ap.add_argument("--params",      type=str,   default=None)
    ap.add_argument("--funding-arb", action="store_true", help="Show Hyperliquid funding rates")
    ap.add_argument("--tearsheet",   action="store_true", help="Generate QuantStats HTML tearsheet")
    args = ap.parse_args()

    if getattr(args, 'funding_arb', False):
        from broker.hyperliquid_arb import get_funding_arb_signals
        signals = get_funding_arb_signals()
        for s in signals:
            print(f"{s['symbol']:6} {s['funding_ann']:>7.2f}% ann  {s['signal']:20}  {s['summary']}")
        return

    params   = load_params(args.params)
    watchlist = json.loads((DIR / "config.json").read_text()).get("watchlist", [])
    end   = datetime.today() + timedelta(days=1)  # +1: yfinance end is exclusive
    start = end - timedelta(days=365 * args.years + 1)

    print(f"\n{'='*64}")
    print(f"  BACKTEST v2  |  {start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')}")
    print(f"  Capital/asset: ${args.capital:,.0f}  |  Params: {args.params or 'best_params.json'}")
    print(f"  RSI buy: {params.get('rsi_buy_min',25)}–{params.get('rsi_buy_max',60)}  "
          f"sell: >{params.get('rsi_sell',75)}")
    print(f"  Stop: {params.get('stop_atr',3)}×ATR  Target: {params.get('target_atr',6)}×ATR  "
          f"Risk: {params.get('risk_pct',0.015)*100:.1f}%/trade")
    print(f"{'='*64}\n")

    results = []
    for item in watchlist:
        sym = item["symbol"]; mkt = item["market"]
        print(f"Running {sym}...", end=" ", flush=True)
        r = run_asset(sym, mkt, start.strftime("%Y-%m-%d"),
                      end.strftime("%Y-%m-%d"), args.capital, params)
        if r:
            results.append(r)
            flag = "✓" if r["total_return"] > 0 else "✗"
            print(f"{flag}  return={r['total_return']:+.1f}%  win={r['win_rate']:.0f}%  "
                  f"trades={r['trades']}  PF={r['profit_factor']}  "
                  f"sharpe={r['sharpe']}  avghold={r['avg_hold_days']}d")
        else:
            print("skipped")

    if not results:
        print("No results."); return

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print(f"{'Symbol':<7} {'Return':>8} {'Win%':>6} {'#':>5} "
          f"{'MaxDD':>7} {'Sharpe':>7} {'Calmar':>7} {'PF':>6} {'Hold':>6}")
    print(f"{'─'*72}")
    for r in results:
        print(f"{r['symbol']:<7} {r['total_return']:>+7.1f}% {r['win_rate']:>5.0f}% "
              f"{r['trades']:>5} {r['max_drawdown']:>6.1f}% {r['sharpe']:>7.2f} "
              f"{r['calmar']:>7.2f} {r['profit_factor']:>6.2f} {r['avg_hold_days']:>5.0f}d")

    avg_ret = sum(r["total_return"]  for r in results) / len(results)
    avg_win = sum(r["win_rate"]      for r in results) / len(results)
    avg_pf  = sum(r["profit_factor"] for r in results if r["profit_factor"] != float("inf")) / max(1, len(results))
    profitable = sum(1 for r in results if r["total_return"] > 0)

    print(f"{'─'*72}")
    print(f"{'AVERAGE':<7} {avg_ret:>+7.1f}% {avg_win:>5.0f}%")

    # ── Trade-level breakdown ─────────────────────────────────────────────────
    print(f"\n{'─'*64}")
    print("  TRADE LOG")
    print(f"{'─'*64}")
    for r in results:
        buys  = [t for t in r["trade_log"] if t["action"] == "BUY"]
        exits = [t for t in r["trade_log"] if "pnl" in t]
        if exits:
            best  = max(exits, key=lambda x: x["pnl"])
            worst = min(exits, key=lambda x: x["pnl"])
            print(f"  {r['symbol']:<5} {len(buys)} entries | "
                  f"best: ${best['pnl']:+,.0f} ({best['date']}) | "
                  f"worst: ${worst['pnl']:+,.0f} ({worst['date']})")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    if avg_ret > 8 and avg_win > 55 and avg_pf > 1.3:
        print(f"  VERDICT: GO LIVE ✓ — {profitable}/{len(results)} profitable  "
              f"avg {avg_ret:+.1f}%  PF {avg_pf:.2f}")
    elif avg_ret > 2 and avg_pf > 1.0:
        print(f"  VERDICT: PAPER TRADE — marginal edge  avg {avg_ret:+.1f}%  PF {avg_pf:.2f}")
    else:
        print(f"  VERDICT: REFINE — avg {avg_ret:+.1f}%  PF {avg_pf:.2f}  "
              f"Run god_mode to find better params")
    print(f"{'='*64}\n")

    # ── QuantStats tearsheets ─────────────────────────────────────────────────
    if args.tearsheet:
        try:
            import quantstats
            for r in results:
                if not r.get("equity_series"):
                    continue
                eq = pd.Series(r["equity_series"])
                if len(eq) < 5:
                    continue
                returns = eq.pct_change().dropna()
                sym = r["symbol"]
                out = f"backtest_{sym}.html"
                quantstats.reports.html(returns, output=out, title=f"AI-Trader {sym}")
                print(f"Tearsheet saved → {out}")
        except ImportError:
            print("Install quantstats: pip install quantstats")

    report = DIR / "backtest_report.json"
    report.write_text(json.dumps({
        "run_at":    datetime.now().isoformat(),
        "period":    {"start": start.strftime("%Y-%m-%d"), "end": end.strftime("%Y-%m-%d")},
        "capital":   args.capital,
        "params":    params,
        "results":   results,
        "summary":   {"avg_return": round(avg_ret, 2), "avg_win_rate": round(avg_win, 2),
                      "avg_profit_factor": round(avg_pf, 2), "profitable_assets": profitable},
    }, indent=2))
    print(f"Report saved → {report}")


if __name__ == "__main__":
    main()


# ── Parameter Optimization ────────────────────────────────────────────────────
def _run_combo(df, rsi_p, bb_w, bb_s, adx_thr, symbol="NVDA"):
    """Returns {return_pct, sharpe, max_dd, win_rate} for given params."""
    c = df["Close"].squeeze()
    h = df["High"].squeeze()
    l = df["Low"].squeeze()
    # RSI
    delta = c.diff()
    g = delta.clip(lower=0).rolling(rsi_p).mean()
    ls = (-delta.clip(upper=0)).rolling(rsi_p).mean()
    rsi = 100 - (100 / (1 + g / ls.replace(0, float('nan'))))
    # BB
    bb_mid = c.rolling(bb_w).mean()
    bb_std_s = c.rolling(bb_w).std()
    bb_up = bb_mid + bb_s * bb_std_s
    bb_lo = bb_mid - bb_s * bb_std_s
    bb_pos = (c - bb_lo) / (bb_up - bb_lo).replace(0, float('nan'))
    # ADX
    tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    up_mv = h.diff().clip(lower=0); dn_mv = (-l.diff()).clip(lower=0)
    plus_dm = up_mv.where(up_mv > dn_mv, 0.0)
    minus_dm = dn_mv.where(dn_mv >= up_mv, 0.0)
    tr14 = tr.rolling(14).mean().replace(0, float('nan'))
    plus_di = 100 * plus_dm.rolling(14).mean() / tr14
    minus_di = 100 * minus_dm.rolling(14).mean() / tr14
    di_sum = (plus_di + minus_di).replace(0, float('nan'))
    adx = (100*(plus_di-minus_di).abs()/di_sum).rolling(14).mean()
    # Signals
    buy  = (rsi < 35) & (bb_pos < 0.3) & (adx > adx_thr)
    sell = (rsi > 65) | (bb_pos > 0.85)
    # Simulate
    equity = [1.0]; position = 0; entry = 0; wins = 0; trades = 0
    for i in range(len(c)):
        if position == 0 and buy.iloc[i]:
            position = 1; entry = c.iloc[i]
        elif position == 1 and sell.iloc[i]:
            ret = c.iloc[i] / entry
            equity.append(equity[-1] * ret)
            trades += 1
            if ret > 1: wins += 1
            position = 0
        else:
            equity.append(equity[-1])
    eq = pd.Series(equity)
    rets = eq.pct_change().dropna()
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0
    peak = eq.cummax(); dd = ((eq - peak) / peak)
    return {
        "return_pct": round((equity[-1]-1)*100, 1),
        "sharpe": round(sharpe, 2),
        "max_dd": round(float(dd.min())*100, 1),
        "win_rate": round(wins/trades*100, 1) if trades > 0 else 0,
        "trades": trades,
        "params": f"RSI={rsi_p} BB={bb_w},{bb_s} ADX>{adx_thr}",
    }

def optimize_parameters(symbol="NVDA", period="2y"):
    import itertools, yfinance as yf
    print(f"\n{'='*60}")
    print(f"  Parameter Optimization — {symbol} {period}")
    print(f"{'='*60}")
    df = yf.download(symbol, period=period, auto_adjust=True, progress=False)
    if df.empty:
        print("No data"); return
    results = []
    for rsi_p, bb_w, bb_s, adx_thr in itertools.product(
        [10, 14, 20], [15, 20, 25], [1.5, 2.0, 2.5], [20, 25, 30]
    ):
        try:
            r = _run_combo(df, rsi_p, bb_w, bb_s, adx_thr)
            results.append(r)
        except Exception:
            pass
    results.sort(key=lambda x: x["sharpe"], reverse=True)
    print("\nTop 5 by Sharpe:")
    print(f"{'Params':<30} {'Return%':>8} {'Sharpe':>7} {'MaxDD%':>7} {'WinR%':>7} {'Trades':>7}")
    print("-"*70)
    for r in results[:5]:
        print(f"{r['params']:<30} {r['return_pct']:>8} {r['sharpe']:>7} {r['max_dd']:>7} {r['win_rate']:>7} {r['trades']:>7}")
    print(f"\nBest params: {results[0]['params']}")
    return results[0]
