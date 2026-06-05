# Strategy Teardown — AI-Trader

> **TL;DR:** A multi-signal autonomous trading bot built to paper-trade US equities and crypto. This document is an honest post-mortem: what the strategy does, three bugs we found and fixed, and what the numbers actually look like after the fixes. The goal is not to present a winning system — it is to demonstrate the engineering judgment to find and correct subtle quant bugs before they cost real money.

---

## What the Strategy Does

`trader.py` runs every 30 minutes via macOS launchd. On each run it:

1. **Fetches macro context** — VIX, QQQ/SPY 5-day returns, and a regime classifier that can declare event blackouts (FOMC, elections, earnings season).
2. **Pulls multi-source signals** — social media sentiment (Twitter/X, Reddit WSB), government/congressional insider trade filings, and earnings calendar proximity.
3. **Computes technical indicators** — RSI-14, MACD histogram, Bollinger Band position, SMA-20/50 crossover, ATR-14 (for position sizing and stops).
4. **Sends all context to an LLM** (Ollama locally → GitHub Models free tier → Anthropic as fallback) in a "debate brain" pattern: a bull agent and a bear agent each make a case, and consensus above 70% confidence threshold is required to trade.
5. **Executes paper orders** on ai4trade.ai and mirrors them to an Alpaca paper account, with ATR-based position sizing (1% portfolio risk per trade), a 3×ATR stop-loss, and a 6×ATR profit target.
6. **Sends a daily email summary** at market close.

`backtest.py` validates the *deterministic rule core* — the same RSI/MACD/BB/SMA-50 entry logic from the LLM prompt, expressed as pure Python without LLM calls. This separates what is measurable and reproducible from what is not.

---

## Bug 1 — Lookahead Bias in Backtest (Critical)

### What it was

In the original `run_asset()` loop, the signal was computed from row T's indicators and the fill was executed **at the same row's close price**:

```python
sig   = signal(row)         # derived from row T's close
...
if sig == "BUY":
    cost = qty * price      # price = row["close"] — same bar!
```

This is classic look-ahead bias. In reality, indicators are only known at end-of-day. You cannot buy *on* the bar that generated the signal; the earliest you can fill is the next trading day's open.

### How it was fixed

```python
# Generate signal series from close-of-day indicators
ind["sig"] = ind.apply(signal, axis=1)
# Shift forward: signal is available next day
ind["sig"] = ind["sig"].shift(1)
# Fill at next day's open (the bar where sig is now aligned)
fill_price = row["open"]
```

`compute_all()` now also returns the `open` column (adjusted open from yfinance) for use as the realistic fill price.

### Why it matters

Signals computed on a close bar and filled on the *same* close cannot be executed in practice. Any backtest that does this overstates returns. Fixing it degraded results slightly — as expected — because the entry sometimes gaps unfavorably overnight.

---

## Bug 2 — No Commission Model

### What it was

The original simulation had zero transaction costs. Every round-trip was free. For a strategy with 4–8 trades per asset over two years this is a small distortion numerically, but it is conceptually wrong and signals to any quant reviewer that the backtest is not production-ready.

### How it was fixed

A 0.1% round-trip commission (0.05% per side) is applied at both fill and exit:

```python
COMMISSION_PER_SIDE = 0.0005   # 0.05% per leg → 0.10% round-trip

# On buy:
commission = gross_cost * COMMISSION_PER_SIDE
cost       = gross_cost + commission

# On sell:
commission = gross_proceeds * COMMISSION_PER_SIDE
proceeds   = gross_proceeds - commission
```

This approximates realistic retail costs for US equities (PFOF brokers, spread, SEC fee). Crypto taker fees are typically higher (0.1–0.25%), so the model is actually conservative for BTC/ETH.

---

## Bug 3 — No Portfolio-Level Drawdown Stop

### What it was

The live bot (`trader.py`) had per-position stop-losses (3×ATR from entry) but no portfolio-level circuit breaker. If multiple positions all moved against the strategy simultaneously — correlated drawdown, common in a risk-off tape — the bot would keep opening new longs, compounding losses until the account was exhausted.

There was also a subtler problem: the bot is **stateless across runs** (launchd relaunches it every 30 min). Any in-memory peak calculation would reset each run and the breaker could never fire.

### How it was fixed

A persistent high-water mark is stored in `portfolio_hwm.json` between runs:

```python
HWM_FILE = DIR / "portfolio_hwm.json"   # persists across launchd runs
DRAWDOWN_HALT_PCT = 0.10                # halt BUYs at -10% from peak

def check_drawdown_circuit(token, cash):
    hwm           = load_hwm()           # load from disk
    current_value = _get_total_portfolio_value(token, cash)  # cash + positions

    if current_value > hwm["peak"]:
        hwm["peak"] = current_value      # running maximum, never resets

    drawdown_pct = (current_value - hwm["peak"]) / hwm["peak"]

    if drawdown_pct <= -DRAWDOWN_HALT_PCT:
        hwm["halted"] = True             # write back to disk
        save_hwm(hwm)
        return True, ...                 # caller blocks all BUY orders
```

The circuit breaker blocks **only new BUY orders**. Stop-loss and profit-target exits remain fully active — you can always get out of positions, just not into new ones during a drawdown.

Recovery clears the halt automatically when the portfolio climbs back above 90% of peak.

Note on LLM noise: this is a design decision, not a code bug. The LLM layer (temperature=0.1, multi-backend fallback chain, debate consensus) is inherently non-deterministic. The backtest deliberately tests only the deterministic rule layer (no LLM calls). This is correct research design — validate what is reproducible, treat the LLM as an advisory overlay that adds subjective judgment. The risk is that the LLM introduces noise that degrades the rule-layer edge; the honest mitigation is extended paper-trading to measure this empirically.

---

## Honest Backtest Results (After All Fixes)

**Period:** 2024-05-31 → 2026-05-31 (2 years)
**Starting capital per asset:** $100,000
**Commission:** 0.10% round-trip
**Lookahead bias:** eliminated (signals shifted +1 day, fills at next-day open)

| Symbol | Return | Win Rate | Trades | Max Drawdown | Sharpe |
|--------|--------|----------|--------|--------------|--------|
| NVDA   | +1.20% | 87.5%    | 8      | -0.52%       | 0.99   |
| AAPL   | -0.58% | 28.6%    | 7      | -0.63%       | -1.24  |
| MSFT   | -0.24% | 0.0%     | 2      | -0.36%       | -0.88  |
| BTC    | -0.58% | 25.0%    | 4      | -0.79%       | -0.64  |
| ETH    | -0.40% | 25.0%    | 4      | -0.61%       | -0.32  |
| **AVG**| **-0.10%**| **33.2%** | — | — | — |

### What these numbers mean

The strategy does not have a statistically significant edge on this 2-year sample. Average return of -0.10% is essentially flat after costs, with only 1 of 5 assets showing positive return. A buy-and-hold of NVDA over the same period returned roughly +180%.

This is the expected outcome for a medium-frequency momentum strategy on daily bars. Most edge in publicly described strategies is competed away or was never there. The honest verdict from `backtest.py` itself: **"REFINE STRATEGY — avg return -0.10%, not ready."**

The backtest also trades infrequently (2–8 round-trips per asset over 2 years), meaning these return estimates have extremely wide confidence intervals. More data or higher frequency would be needed to establish statistical significance.

---

## What It Would Take to Go Live

### SEBI Requirements (Indian Resident)

| Requirement | Details |
|-------------|---------|
| Algo trading registration | Must go through a SEBI-registered broker with API access (Zerodha, FYERS, Upstox) — direct market access requires institutional status |
| Algorithmic trading approval | Retail algo trading on Indian exchanges requires SEBI-approved API access from the broker; each algo must be logged and audited |
| F&O segment access | Requires exchange membership or sub-broker arrangement; direct options/futures not accessible to retail via API without broker gateway |
| Tax compliance | Short-term capital gains tax (STCG) at 15% + surcharge; algo profits are speculative income; quarterly advance tax applies if gains exceed ₹10,000 |
| KYC / account type | Standard demat + trading account; no special license required for personal trading via broker API |

### Broker API (India)

- **Zerodha Kite Connect** — well-documented Python SDK, ₹2,000/month API subscription, paper trading available
- **FYERS API** — free tier available, WebSocket streaming, suitable for low-frequency daily strategies
- **Alpaca (US markets)** — already integrated in this codebase (`broker/alpaca_exec.py`); paper trading is live and free

### Minimum Viable Production Checklist

- [ ] 6+ months of paper trading with real-time fills logged
- [ ] Out-of-sample test on data after strategy was coded (avoids data snooping)
- [ ] Walk-forward validation across 3+ market regimes (bull, bear, sideways)
- [ ] Statistical significance test: strategy Sharpe meaningfully above zero at p < 0.05
- [ ] Slippage model: not just commission, but realistic bid-ask spread impact at position size
- [ ] Live risk controls: position limits, daily loss limit, API error handling, alerting

---

## Why This Is a Portfolio Piece

This project demonstrates several engineering judgment calls that are harder to fake than raw code:

**1. Knowing what NOT to measure with an LLM.** The backtest explicitly excludes LLM calls and tests only the deterministic rule layer. This is the correct separation — the LLM is a signal overlay, not a replicable signal. Most junior quants would either test the full system (and get unprincipled results) or not test at all.

**2. Catching lookahead bias before it inflated metrics.** Lookahead bias is the most common error in backtesting and the hardest to catch if you are not looking for it. The original code looked correct; the bug required understanding the data model (yfinance's daily close bar, the execution sequence) to spot.

**3. Stateful risk controls in a stateless process.** The drawdown circuit breaker had to persist across launchd invocations. The naive implementation (in-memory peak) would never fire. Recognizing this and mirroring the existing `positions.json` pattern for the HWM file required understanding the runtime architecture.

**4. Honest reporting.** The system currently does not have a measurable edge. The STRATEGY_TEARDOWN says so plainly. A production quant team would rather see an engineer who ships honest numbers than one who optimizes metrics until the verdict sounds good.

**5. Full stack integration.** The project spans Python quantitative finance (yfinance, pandas, numpy), LLM API integration (Anthropic SDK, GitHub Models, Ollama), broker API integration (Alpaca paper trading), multi-agent debate architecture, scheduling (launchd plist), and async signal ingestion (social media, SEC filings, earnings calendars). That breadth in a single coherent codebase is itself a signal.
