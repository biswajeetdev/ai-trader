# AI-Trader

An autonomous paper-trading research system that combines classical technical analysis, 15+ signal sources, a multi-agent LLM debate engine, and adaptive risk controls. Built to explore what it takes to build a production-grade systematic trader — not to claim a live edge.

**Status:** Paper trading (Alpaca $100K + ai4trade $100K simulated). Not deployed with real capital.

---

## Architecture Overview

```
trader.py                    ← main loop (runs every 30 min via launchd/cron)
│
├── signals/                 ← parallel signal fetch (ThreadPoolExecutor)
│   ├── debate_brain.py      ← BULL / BEAR / ARBITER multi-agent LLM debate
│   ├── regime.py            ← BotScore: VIX + SPY momentum → regime classification
│   ├── screener.py          ← dynamic universe scanner (BB squeeze + ADX)
│   ├── polymarket_signals.py← crowd probability from prediction markets
│   ├── openbb_macro.py      ← FRED: yield curve, CPI, unemployment, Fed rate
│   ├── options_flow.py      ← unusual call sweeps, OTM put accumulation
│   ├── whale_tracker.py     ← 13F/13D institutional filings
│   ├── insider_trades.py    ← SEC Form 4, congressional disclosures
│   ├── news_catalyst.py     ← EDGAR 8-K, Google News M&A signals
│   ├── earnings_calendar.py ← earnings proximity + blackout (±5 days)
│   ├── fear_greed.py        ← CNN Fear & Greed (stock + crypto)
│   ├── social_pulse.py      ← Twitter/X, news sentiment
│   ├── wsb_social.py        ← Reddit WSB + StockTwits + Truth Social
│   ├── stock_ranker.py      ← cross-sectional ranking vs SPY + peers
│   ├── mean_reversion.py    ← RSI reversion + pairs z-score
│   ├── bb_patterns.py       ← Bollinger Band chart pattern detection
│   └── india_signals.py     ← NSE bulk deals, India VIX
│
├── broker/
│   ├── risk.py              ← Kelly sizing, stops, partial profit, circuit breaker
│   ├── alpaca_exec.py       ← Alpaca paper (bracket orders: stop 3×ATR, target 7.5×ATR)
│   ├── ai4trade.py          ← ai4trade.ai simulation broker
│   ├── approval_bot.py      ← Telegram approval gate (inline keyboard)
│   ├── approval_queue.py    ← thread-safe pending trade queue
│   ├── short_put_exec.py    ← Alpaca options: cash-secured puts
│   ├── zerodha_exec.py      ← Zerodha Kite (India, live-ready)
│   └── telegram_notifier.py ← trade alerts + daily summary
│
├── rag/
│   └── pattern_memory.py    ← ChromaDB RAG: index past trades, query similar setups
│
├── backtest.py              ← honest walk-forward backtest (no LLM, no lookahead)
└── scripts/
    ├── server_setup.sh      ← Oracle Cloud Free Tier deploy script
    ├── trader.service        ← systemd oneshot service
    └── trader.timer          ← systemd timer (every 30 min)
```

---

## Signal Stack

All signals are fetched in parallel. The LLM arbiter weights them by priority:

| Priority | Signal | Source |
|----------|--------|--------|
| 1 | **Options Flow** | Unusual call sweeps, OTM put accumulation |
| 2 | **Whale / 13D** | Activist hedge fund filings |
| 3 | **Govt Insider** | Congressional purchases (legal insider edge) |
| 4 | **News / M&A** | EDGAR 8-K, Google News catalyst |
| 5 | **Fear & Greed** | CNN index — extreme fear = contrarian BUY |
| 6 | **Prediction Markets** | Polymarket crowd probability + rate arbitrage |
| 7 | **Social** | Buffett/Musk mentions, WSB momentum |
| 8 | **Technicals** | BB squeeze, ADX, RSI, MACD, Supertrend |

---

## Decision Engine: Multi-Agent Debate

For each asset, three LLM agents run in parallel:

```
BULL analyst  ──┐
                ├──→ ARBITER (final call + confidence 0-100)
BEAR analyst  ──┘

Consensus: STRONG (+5 conf) | WEAK (no change) | SPLIT (-20 conf)
Threshold: confidence < 70 → HOLD regardless of action
```

LLM backend priority: **Ollama local** → **GitHub Models (gpt-4o-mini, free)** → **Anthropic fallback**

---

## Risk Management

| Control | Implementation |
|---------|---------------|
| **Position sizing** | Kelly criterion (half-Kelly, ≥5 trades) or fixed 1% portfolio risk |
| **Stop loss** | 3 × ATR below entry (GTC bracket) |
| **Profit target** | 7.5 × ATR above entry (GTC bracket, 2.5:1 R:R) |
| **Partial profit** | Sell 50% at 3.75×ATR gain → move stop to breakeven |
| **Correlation guard** | Skip if new position correlates >0.7 with existing holdings |
| **Circuit breaker** | Halt new longs if portfolio drawdown >10% from peak |
| **Earnings blackout** | No new entries within ±5 days of earnings |
| **Regime filter** | BotScore <35 → mean reversion mode; >65 → momentum mode |

---

## BotScore Regime Classifier

Scores current market on a 0-100 scale each run:

```
VIX < 15   → +25 pts  (calm, algo-driven)
SPY 5d > 0 → +15 pts  (trend)
Low intraday range → +10 pts

≥ 65 → BOT_DRIVEN  → momentum strategy (BB squeeze + trend)
≤ 35 → HUMAN_DRIVEN → mean reversion strategy (RSI + pairs)
```

---

## Dynamic Universe Screener

Instead of a fixed watchlist, the screener scans 30 US + 10 India symbols every 30 min and adds the top 3 BB-squeeze candidates to the active watchlist:

```
Score: +40 squeeze_released, +30 ADX>20, +20 price>SMA50, +10 volume spike
Filter: skip if avg_daily_volume < 500K
Cache: 30 min
```

---

## Short Put Strategy (Income)

Separate income layer that sells cash-secured puts on bullish, liquid underlyings:

- Criteria: RSI 30-60, above SMA50, bull regime (SMA50 > SMA200), >30 DTE
- Targeting ~4% OTM strikes, 26-35 DTE
- Current positions: MSFT $415P Jul-2 + AAPL $295P Jul-2 ($934 total credit)

---

## RAG Trade Memory

Every completed trade is indexed into ChromaDB. Before each new decision, the system queries for similar past setups:

```python
rag_ctx = get_rag_context(symbol, indicators, recent_news)
# → "3 similar setups found: 2 wins (avg +8.2%), 1 loss (-3.1%)"
# Piped into debate_decide() as rag_ctx= parameter
```

The arbiter is calibrated by historical win rate — if win rate <45%, confidence threshold tightens.

---

## LLM Backend

```
1. Ollama local   — qwen2.5:32b, deepseek-r1:14b, llama3.1:8b (offline, free)
2. GitHub Models  — gpt-4o-mini via free inference API (default, 50 req/min)
3. Anthropic      — claude-sonnet-4-6 (requires ANTHROPIC_API_KEY, fallback)
```

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure (copy and fill in your keys)
cp config.json.example config.json

# 3. Run
python3 trader.py --dry-run    # simulate, no orders placed
python3 trader.py              # paper trade

# 4. Telegram approval bot (optional, run alongside trader)
python3 broker/approval_bot.py &

# 5. Backtest the rule layer
python3 backtest.py

# 6. Deploy 24/7 (Oracle Cloud Free Tier)
bash scripts/server_setup.sh
```

**Required in `config.json`:**
```json
{
  "watchlist": [{"symbol": "NVDA", "market": "us-stock"}, ...],
  "alpaca_api_key": "...",
  "alpaca_secret_key": "...",
  "alpaca_paper": true,
  "telegram_bot_token": "...",
  "telegram_chat_id": "...",
  "openbb_fmp_key": "...",
  "openbb_fred_key": "..."
}
```

---

## Backtest Results (Honest)

Walk-forward backtest with lookahead-bias fix, 0.1% round-trip commission, no LLM in the loop:

| Symbol | 2yr Return | Win Rate | Sharpe |
|--------|-----------|----------|--------|
| NVDA   | +1.20%    | 87.5%    | 0.99   |
| AAPL   | -0.58%    | 28.6%    | -1.24  |
| MSFT   | -0.24%    | 0.0%     | -0.88  |
| BTC    | -0.58%    | 25.0%    | -0.64  |
| ETH    | -0.40%    | 25.0%    | -0.32  |

**Average: -0.10% over 2 years.** The mechanical rule layer alone has no edge — alpha comes from the LLM signal layer and signal prioritization. See [STRATEGY_TEARDOWN.md](STRATEGY_TEARDOWN.md) for the full post-mortem.

---

## Why This Exists

This is not a "I built a trading bot that beats the market" project. It demonstrates:

1. **Honest testing** — walk-forward backtest, lookahead-bias fix, realistic commission model
2. **Production risk controls** — drawdown halt, earnings blackout, correlation guard, regime filters
3. **LLM integration** — multi-agent debate as signal aggregator, not as oracle
4. **System design** — parallel signal fetch, RAG memory, approval gate, dual-broker execution
5. **Engineering judgment** — knowing what signals to trust and in what order

The value is in the architecture and judgment calls, not the P&L.

---

## Disclaimer

This project is for educational and research purposes only. Nothing here constitutes financial advice. Paper trading results do not predict real trading outcomes.
