#!/usr/bin/env python3
"""
AI-Trader — Fully Autonomous Edition
- Runs every 30 min during market hours (launchd)
- Stop-loss: auto-exit if price drops 3×ATR from entry
- Profit target: auto-exit at 2:1 reward-to-risk (6×ATR gain)
- Feedback loop: recent trade outcomes fed back into LLM context
- Daily email summary at market close
- LLM backend: Ollama (pendrive) → GitHub Models (free) → Anthropic
"""

import json, os, subprocess, sys, time, smtplib, requests
import yfinance as yf
import pandas as pd
from openai import OpenAI
from email.mime.text import MIMEText
from datetime import datetime, timezone, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from signals.social_pulse      import get_social_signals,      format_for_llm as social_fmt
from signals.insider_trades    import get_all_insider_signals,  format_for_llm as insider_fmt
from signals.wsb_social        import get_all_social,            format_for_llm as wsb_fmt
from signals.earnings_calendar import get_earnings_info,         format_for_llm as earn_fmt, earnings_signal
from signals.debate_brain      import debate_decide
from signals.bb_patterns       import detect_bb_pattern
from signals.regime            import get_full_regime_context,   format_for_llm as regime_fmt, get_market_regime_score
from signals.news_catalyst     import get_news_signals,          format_for_llm as news_fmt
from signals.options_flow      import get_options_signals,        format_for_llm as options_fmt
from signals.whale_tracker     import get_whale_signals,          format_for_llm as whale_fmt
from signals.fear_greed        import get_fear_greed_signals,     format_for_llm as fg_fmt, get_summary as fg_summary
from signals.india_signals     import get_nse_bulk_deals, get_india_vix, get_nse_options_flow, format_bulk_deals_for_llm
from broker.approval_queue    import queue_trade, update_message_id
from broker.telegram_notifier import send_approval_request
from broker.ai4trade   import auth, get_profile, execute_trade, get_positions_api, refresh_token
from rag.pattern_memory import get_rag_context
from broker.alpaca_exec import execute_alpaca_trade, get_alpaca_portfolio, export_alpaca_to_excel
from broker.telegram_notifier import send_trade_alert, send_daily_summary as tg_daily_summary, send_run_status
from signals.short_put_screener import find_short_put_opportunity, check_exits as sp_check_exits, load_positions as sp_load_positions
from broker.short_put_exec import execute_short_put, close_short_put
from signals.india_short_put_screener import (
    find_india_short_put_opportunity, check_india_exits,
    load_positions as india_sp_load_positions,
)
from broker.zerodha_exec import execute_india_short_put, close_india_short_put
from signals.stock_ranker import rank_watchlist, get_rank_context
from broker.risk        import (
    check_drawdown_circuit, check_stops, load_positions, save_positions,
    record_open, record_close, recent_trade_context, size_position, get_win_rate,
    dynamic_position_size, get_equity_history, get_correlated_symbols,
    mark_partial_done,
    STOP_LOSS_ATR, PROFIT_TARGET_ATR, MIN_CONFIDENCE, MAX_TRADE_USD,
)
from signals.mean_reversion      import rsi_reversion_signal, get_pairs_signals
from signals.screener            import get_screener_candidates
from signals.polymarket_signals  import get_prediction_signals, format_for_llm as poly_fmt

# ── paths ─────────────────────────────────────────────────────────────────────
DIR    = Path(__file__).parent
CONFIG = DIR / "config.json"
LOG    = DIR / "log.json"
EXCEL  = DIR / "holdings.xlsx"
# POSITIONS, TRADE_HIST, HWM_FILE, TOKEN_FILE → broker.risk / broker.ai4trade

# ── constants ─────────────────────────────────────────────────────────────────
LOG_MAX                = 500
EARNINGS_BLACKOUT_DAYS = 5
# STOP_LOSS_ATR, PROFIT_TARGET_ATR, MIN_CONFIDENCE, MAX_TRADE_USD live in broker.risk

# ── LLM backends ──────────────────────────────────────────────────────────────
OLLAMA_URL    = "http://localhost:11434/v1"
OLLAMA_MODELS = ["qwen2.5:32b", "deepseek-r1:14b", "llama3.3:70b", "llama3.1:8b", "phi4"]
GITHUB_MODEL       = "gpt-4o-mini"  # 50 req/min limit (gpt-4o was 50/day — hit every day)
GITHUB_MODEL_FAST  = "gpt-4o-mini"  # debate brain: same model, 50 req/min
ANTHROPIC_MDL = "claude-sonnet-4-6"

VALID_ACTIONS = {"BUY", "SELL", "SHORT", "COVER", "HOLD"}
VALID_MARKETS = {"us-stock", "crypto", "polymarket", "a-stock", "in-stock"}

_LLM_BACKEND = None   # cached per-process; avoids repeated `gh auth token` subprocess calls


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG + LLM BACKEND
# ══════════════════════════════════════════════════════════════════════════════

def gh_token():
    try:
        return subprocess.check_output(["gh", "auth", "token"], text=True).strip()
    except Exception:
        return None


def detect_llm_backend():
    """Priority: Ollama local (pendrive) → GitHub Models → Anthropic. Result cached per process."""
    global _LLM_BACKEND
    if _LLM_BACKEND is not None:
        return _LLM_BACKEND
    # 1. Ollama
    try:
        r = requests.get(f"{OLLAMA_URL.replace('/v1','')}/api/tags", timeout=2)
        if r.ok:
            available = {m["name"].split(":")[0] for m in r.json().get("models", [])}
            for m in OLLAMA_MODELS:
                if m.split(":")[0] in available:
                    _LLM_BACKEND = OpenAI(base_url=OLLAMA_URL, api_key="ollama"), m, f"Ollama/{m} (local)"
                    return _LLM_BACKEND
    except Exception:
        pass

    # 2. GitHub Models
    key = os.environ.get("GITHUB_TOKEN") or gh_token()
    if key:
        _LLM_BACKEND = (OpenAI(base_url="https://models.inference.ai.azure.com", api_key=key),
                        GITHUB_MODEL, f"{GITHUB_MODEL} via GitHub Models (free)")
        return _LLM_BACKEND

    # 3. Anthropic (via openai-compat isn't available — flag for main to handle)
    cfg_key = _read_config_raw().get("anthropic_api_key", "")
    ant_key = os.environ.get("ANTHROPIC_API_KEY") or cfg_key
    if ant_key:
        _LLM_BACKEND = (None, ANTHROPIC_MDL, "claude-sonnet-4-6 via Anthropic")
        return _LLM_BACKEND

    sys.exit("[error] No LLM backend. Run 'gh auth login' or set ANTHROPIC_API_KEY.")


def _read_config_raw():
    return json.loads(CONFIG.read_text()) if CONFIG.exists() else {}


def load_config():
    with open(CONFIG) as f:
        cfg = json.load(f)
    cfg["anthropic_api_key"] = (
        os.environ.get("ANTHROPIC_API_KEY") or cfg.get("anthropic_api_key", "")
    )
    for item in cfg.get("watchlist", []):
        if item.get("market") not in VALID_MARKETS:
            sys.exit(f"[error] Unknown market '{item.get('market')}'")
    return cfg


# ══════════════════════════════════════════════════════════════════════════════
# MARKET HOURS
# ══════════════════════════════════════════════════════════════════════════════

def is_market_open():
    """True if US stock market is currently open (Mon–Fri 13:30–20:00 UTC / EDT)."""
    now = datetime.now(timezone.utc)
    if now.weekday() >= 5:
        return False
    open_  = now.replace(hour=13, minute=30, second=0, microsecond=0)
    close_ = now.replace(hour=20, minute=0,  second=0, microsecond=0)
    return open_ <= now <= close_


def is_near_close():
    """True if within 35 min of market close — trigger daily summary."""
    now = datetime.now(timezone.utc)
    if now.weekday() >= 5:
        return False
    close_ = now.replace(hour=20, minute=0, second=0, microsecond=0)
    return timedelta(0) <= (close_ - now) <= timedelta(minutes=35)


def is_india_market_open():
    """True if NSE is currently open (Mon–Fri 09:15–15:30 IST = 03:45–10:00 UTC)."""
    now = datetime.now(timezone.utc)
    if now.weekday() >= 5:
        return False
    open_  = now.replace(hour=3, minute=45, second=0, microsecond=0)
    close_ = now.replace(hour=10, minute=0,  second=0, microsecond=0)
    return open_ <= now <= close_


# auth, get_profile → broker.ai4trade


# ══════════════════════════════════════════════════════════════════════════════
# MARKET DATA
# ══════════════════════════════════════════════════════════════════════════════

def compute_indicators(df):
    c = df["Close"].squeeze()
    h = df["High"].squeeze()
    l = df["Low"].squeeze()
    v = df["Volume"].squeeze() if "Volume" in df.columns else None

    if len(c) < 50: raise ValueError("< 50 days of data")

    price = float(c.iloc[-1])
    sma20 = float(c.rolling(20).mean().iloc[-1])
    sma50 = float(c.rolling(50).mean().iloc[-1])

    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(10).mean()
    loss  = (-delta.clip(upper=0)).rolling(10).mean()
    ll    = float(loss.iloc[-1])
    rsi   = 100.0 if ll == 0 else float(100 - 100 / (1 + gain.iloc[-1] / ll))

    ema12 = c.ewm(span=12,adjust=False).mean()
    ema26 = c.ewm(span=26,adjust=False).mean()
    macd  = ema12 - ema26
    sig   = macd.ewm(span=9,adjust=False).mean()

    bb_mid = c.rolling(25).mean()
    bb_std = c.rolling(25).std()
    bb_up  = float((bb_mid + 2*bb_std).iloc[-1])
    bb_lo  = float((bb_mid - 2*bb_std).iloc[-1])
    bb_pos = (price - bb_lo)/(bb_up - bb_lo) if (bb_up - bb_lo) > 0 else 0.5

    tr  = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])

    # Keltner Channels (Moon Dev BB Squeeze: BB inside KC = volatility compression)
    kc_atr   = tr.rolling(20).mean()
    kc_mid_s = c.rolling(20).mean()
    kc_up_s  = kc_mid_s + 1.5 * kc_atr
    kc_lo_s  = kc_mid_s - 1.5 * kc_atr

    # ADX 14 (pure pandas — no talib needed)
    up_mv    = h.diff().clip(lower=0)
    dn_mv    = (-l.diff()).clip(lower=0)
    plus_dm  = up_mv.where(up_mv > dn_mv, 0.0)
    minus_dm = dn_mv.where(dn_mv >= up_mv, 0.0)
    tr14     = tr.rolling(14).mean().replace(0, float("nan"))
    plus_di  = 100 * plus_dm.rolling(14).mean() / tr14
    minus_di = 100 * minus_dm.rolling(14).mean() / tr14
    di_sum   = (plus_di + minus_di).replace(0, float("nan"))
    adx_val  = round(float(((100*(plus_di-minus_di).abs()/di_sum).rolling(14).mean()).iloc[-1]), 1)

    # BB Squeeze: BB bands inside Keltner = volatility compressed, breakout incoming
    bb_up_s  = bb_mid + 2 * bb_std
    bb_lo_s  = bb_mid - 2 * bb_std
    sq_now   = bool(bb_up_s.iloc[-1] < kc_up_s.iloc[-1] and bb_lo_s.iloc[-1] > kc_lo_s.iloc[-1])
    sq_prev  = bool(bb_up_s.iloc[-2] < kc_up_s.iloc[-2] and bb_lo_s.iloc[-2] > kc_lo_s.iloc[-2])
    sq_rel   = sq_prev and not sq_now   # squeeze just released → breakout starting

    # ── Supertrend (period=10, multiplier=3.0) ────────────────────────────────
    st_atr   = tr.rolling(10).mean()
    hl2      = (h + l) / 2
    upper_st = hl2 + 3.0 * st_atr
    lower_st = hl2 - 3.0 * st_atr
    prev_dir = 1
    prev_up  = upper_st.iloc[0]
    prev_lo  = lower_st.iloc[0]
    dirs = []
    for i in range(len(c)):
        cu = upper_st.iloc[i]
        cl = lower_st.iloc[i]
        cu = min(cu, prev_up) if c.iloc[i-1] <= prev_up else cu
        cl = max(cl, prev_lo) if c.iloc[i-1] >= prev_lo else cl
        if prev_dir == -1 and c.iloc[i] > prev_up:
            d = 1
        elif prev_dir == 1 and c.iloc[i] < prev_lo:
            d = -1
        else:
            d = prev_dir
        dirs.append(d)
        prev_dir, prev_up, prev_lo = d, cu, cl
    st_dir = int(dirs[-1])  # 1 = bullish, -1 = bearish

    # ── StochRSI (period=14, smooth=3) ───────────────────────────────────────
    delta2  = c.diff()
    g2      = delta2.clip(lower=0).rolling(14).mean()
    ls2     = (-delta2.clip(upper=0)).rolling(14).mean()
    rsi_s   = 100 - (100 / (1 + g2 / ls2.replace(0, float('nan'))))
    rsi_min = rsi_s.rolling(14).min()
    rsi_max = rsi_s.rolling(14).max()
    stoch_k = ((rsi_s - rsi_min) / (rsi_max - rsi_min).replace(0, float('nan'))).rolling(3).mean()
    stoch_d = stoch_k.rolling(3).mean()
    stochrsi_k = round(float(stoch_k.iloc[-1]), 3) if not pd.isna(stoch_k.iloc[-1]) else 0.5
    stochrsi_d = round(float(stoch_d.iloc[-1]), 3) if not pd.isna(stoch_d.iloc[-1]) else 0.5

    # ── Ichimoku (tenkan=9, kijun=26, senkou B=52) ───────────────────────────
    tenkan  = (h.rolling(9).max()  + l.rolling(9).min())  / 2
    kijun   = (h.rolling(26).max() + l.rolling(26).min()) / 2
    senkou_a = (tenkan + kijun) / 2
    senkou_b = (h.rolling(52).max() + l.rolling(52).min()) / 2
    ichi_above_cloud = (price > float(senkou_a.iloc[-1]) and price > float(senkou_b.iloc[-1]))
    ichi_tk_cross    = (float(tenkan.iloc[-1]) > float(kijun.iloc[-1]))

    # ── OBV + MFI (volume-based) ──────────────────────────────────────────────
    obv_val = mfi_val = None
    if v is not None:
        obv_dir  = pd.Series(c.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)),
                             index=c.index)
        obv      = (v * obv_dir).cumsum()
        obv_slope = float(obv.iloc[-1]) > float(obv.rolling(10).mean().iloc[-1])
        tp_s     = (h + l + c) / 3
        mf       = tp_s * v
        pos_mf   = mf.where(tp_s > tp_s.shift(1), 0).rolling(14).sum()
        neg_mf   = mf.where(tp_s < tp_s.shift(1), 0).rolling(14).sum()
        neg_last = float(neg_mf.iloc[-1])
        mfi_val  = round(float(100 - 100 / (1 + float(pos_mf.iloc[-1]) / neg_last)), 1) if neg_last != 0 else 50.0
        obv_val  = bool(obv_slope)

    vol_ratio = None
    if v is not None and float(v.iloc[-1]) > 0:
        avg = float(v.rolling(20).mean().iloc[-1])
        vol_ratio = round(float(v.iloc[-1])/avg, 2) if avg > 0 else None

    return {
        "price": round(price,4), "sma20": round(sma20,4), "sma50": round(sma50,4),
        "above_sma20": price>sma20, "above_sma50": price>sma50,
        "rsi14": round(rsi,2),
        "macd": round(float(macd.iloc[-1]),4),
        "macd_signal": round(float(sig.iloc[-1]),4),
        "macd_hist": round(float(macd.iloc[-1]-sig.iloc[-1]),4),
        "bb_upper": round(bb_up,4), "bb_lower": round(bb_lo,4),
        "bb_position": round(bb_pos,2),
        "atr14": round(atr,4), "vol_ratio": vol_ratio,
        "adx": adx_val, "bb_squeeze": sq_now, "squeeze_released": sq_rel,
        "pct_1d": round((price/float(c.iloc[-2])-1)*100,2),
        "pct_5d": round((price/float(c.iloc[-6])-1)*100,2),
        "supertrend": st_dir,
        "stochrsi_k": stochrsi_k,
        "stochrsi_d": stochrsi_d,
        "ichi_above_cloud": ichi_above_cloud,
        "ichi_tk_bull": ichi_tk_cross,
        "obv_rising": obv_val,
        "mfi": mfi_val,
        "rsi_4h": None,
    }


def get_fundamentals(symbol, market):
    if market != "us-stock": return {}
    try:
        info = yf.Ticker(symbol).info
        ts   = info.get("earningsTimestamp") or info.get("earningsDate")
        dte  = None
        if ts:
            dt  = datetime.fromtimestamp(ts, tz=timezone.utc) if isinstance(ts,(int,float)) else ts
            dte = (dt - datetime.now(tz=timezone.utc)).days
        return {
            "pe_ratio": info.get("trailingPE"), "forward_pe": info.get("forwardPE"),
            "eps_growth": info.get("earningsGrowth"), "revenue_growth": info.get("revenueGrowth"),
            "analyst_rating": info.get("recommendationMean"),
            "analyst_label": info.get("recommendationKey"),
            "short_ratio": info.get("shortRatio"),
            "days_to_earnings": dte, "profit_margin": info.get("profitMargins"),
        }
    except Exception: return {}


def earnings_blackout(fund):
    dte = fund.get("days_to_earnings")
    return dte is not None and abs(dte) <= EARNINGS_BLACKOUT_DAYS


def get_macro():
    def _dl(ticker):
        # Always use explicit date range — period= is unreliable on weekends/holidays
        try:
            end   = datetime.now(timezone.utc)
            start = end - timedelta(days=10)
            df = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                             end=end.strftime("%Y-%m-%d"),
                             interval="1d", progress=False, auto_adjust=True)
            if not df.empty:
                return df
        except Exception:
            pass
        return pd.DataFrame()

    try:
        vix = _dl("^VIX"); qqq = _dl("QQQ"); spy = _dl("SPY")
        vv  = float(vix["Close"].squeeze().iloc[-1]) if not vix.empty else None
        qc  = float((qqq["Close"].squeeze().iloc[-1]/qqq["Close"].squeeze().iloc[0]-1)*100) if len(qqq)>=2 else None
        sc  = float((spy["Close"].squeeze().iloc[-1]/spy["Close"].squeeze().iloc[0]-1)*100) if len(spy)>=2 else None
        return {
            "vix": round(vv,2) if vv else None,
            "vix_level": "high_fear" if vv and vv>30 else ("elevated" if vv and vv>20 else "calm"),
            "qqq_5d_pct": round(qc,2) if qc else None,
            "spy_5d_pct": round(sc,2) if sc else None,
        }
    except Exception:
        return {}


# position management, circuit breaker, sizing → broker.risk


# ══════════════════════════════════════════════════════════════════════════════
# LLM DECISION
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM = """\
You are a veteran portfolio manager with 20 years of experience in equities and crypto.
You think like Paul Tudor Jones — disciplined, systematic, always respect the tape.

DECISION FRAMEWORK (apply in order):
1. MACRO FILTER: VIX > 30 → only HOLD or SELL. Never buy into fear.
2. SOCIAL CATALYST: HIGH urgency bullish signal from Elon Musk/Trump/Buffett → +15 confidence.
   HIGH urgency bearish signal → -20 confidence. Cramer bullish = INVERSE signal (bearish).
3. GOVT INSIDER: Congress/MP/NSE insider BUY → +10 confidence. SELL → -15 confidence.
4. TREND: Price must be above SMA50 to BUY (unless social catalyst overrides with urgency=HIGH).
5. ENTRY (BUY): RSI 30-60 + above SMA50 + MACD hist positive + BB < 60%
6. EXIT (SELL): RSI > 70 OR BB > 85% OR price breaks below SMA50
7. FUNDAMENTALS: Avoid P/E > 50 or earnings within 5 days.
8. CONFIDENCE 0-100: only act if >= 70. Missing data = lower confidence.

CALIBRATION:
- RSI 42, above SMA50, MACD hist +0.5, BB 35%, P/E 28, earnings 60d → BUY confidence=82
- RSI 78, BB 90% → SELL confidence=88
- RSI 29, below SMA50, MACD bearish → HOLD confidence=55
- VIX 32 → HOLD confidence=40

FEEDBACK — your recent track record (learn from this):
{feedback}

RULES: Cite actual numbers. Never invent. Missing macro = -10 confidence.

Respond ONLY with valid JSON, no markdown:
{{"reasoning":{{"technical":"<RSI MACD BB numbers>","fundamental":"<P/E earnings>","macro":"<VIX QQQ>","risks":"<top risks>","confidence":<0-100>}},"action":"BUY"|"SELL"|"HOLD","quantity":0,"reason":"<one line + key number>"}}"""


def llm_decide(symbol, market, ind, fund, macro, cash, cfg, has_position=False):
    client, model, label = detect_llm_backend()

    feedback = recent_trade_context()
    system   = SYSTEM.format(feedback=feedback)

    pos_context = ""
    pos = load_positions().get(symbol, {})
    if pos and has_position:
        pos_context = (f"\nOPEN POSITION: entry ${pos['entry_price']} | "
                       f"stop ${pos['stop_price']} | target ${pos['target_price']} | "
                       f"qty {pos['quantity']}")

    sq_state = ("SQUEEZE RELEASED — breakout starting, ADX " + str(ind.get("adx","?"))
                if ind.get("squeeze_released")
                else ("SQUEEZE ON — volatility compressing, wait" if ind.get("bb_squeeze") else "no squeeze"))
    tech = (f"TECHNICAL: Price ${ind['price']} | RSI {ind['rsi14']} | "
            f"SMA20 ${ind['sma20']} (above:{ind['above_sma20']}) | "
            f"SMA50 ${ind['sma50']} (above:{ind['above_sma50']}) | "
            f"MACD hist {ind['macd_hist']} ({'bullish' if ind['macd_hist']>0 else 'bearish'}) | "
            f"BB {int(ind['bb_position']*100)}% | ATR ${ind['atr14']} | "
            f"ADX {ind.get('adx','?')} | BB-Squeeze: {sq_state} | "
            f"1d:{ind['pct_1d']}% 5d:{ind['pct_5d']}%")
    # Mean reversion signal
    rev_sig = rsi_reversion_signal(ind)
    rev_str = f" | REVERSION: {rev_sig['reason']} (strength:{rev_sig['strength']})" if rev_sig else ""
    tech = tech + rev_str

    fund_str = ("FUNDAMENTAL: N/A (crypto)" if market != "us-stock" else
                f"FUNDAMENTAL: P/E {fund.get('pe_ratio','N/A')} | "
                f"Fwd P/E {fund.get('forward_pe','N/A')} | "
                f"EPS growth {fund.get('eps_growth','N/A')} | "
                f"Analyst {fund.get('analyst_label','N/A')} | "
                f"Days to earnings {fund.get('days_to_earnings','N/A')}")

    macro_str = (f"MACRO: VIX {macro.get('vix','N/A')} ({macro.get('vix_level','unknown')}) | "
                 f"QQQ 5d {macro.get('qqq_5d_pct','N/A')}% | SPY 5d {macro.get('spy_5d_pct','N/A')}%")

    # Social + insider signals (passed via cfg at call time)
    social_str  = cfg.get("_social_context",  {}).get(symbol, "No social signals.")
    insider_str = cfg.get("_insider_context", {}).get(symbol, "No insider trades.")

    user_msg = (f"{tech}\n{fund_str}\n{macro_str}\n"
                f"SOCIAL SIGNALS:\n{social_str}\n"
                f"GOVT/INSIDER TRADES:\n{insider_str}"
                f"{pos_context}\nCash: ${cash:,.0f}")

    # Anthropic path
    if client is None:
        import anthropic as ant
        ac  = ant.Anthropic(api_key=cfg["anthropic_api_key"])
        msg = ac.messages.create(model=model, max_tokens=350, system=system,
                                  messages=[{"role":"user","content":user_msg}])
        raw = msg.content[0].text.strip()
    else:
        resp = client.chat.completions.create(
            model=model, max_tokens=350, temperature=0.1,
            messages=[{"role":"system","content":system},{"role":"user","content":user_msg}])
        raw = resp.choices[0].message.content.strip().lstrip("```json").rstrip("```").strip()

    dec        = json.loads(raw)
    reasoning  = dec.get("reasoning", {})
    confidence = int(reasoning.get("confidence", 0))
    action     = str(dec.get("action","HOLD")).upper().strip()

    if action not in VALID_ACTIONS: raise ValueError(f"Invalid action '{action}'")
    if confidence < MIN_CONFIDENCE: action = "HOLD"

    qty = 0.0
    if action not in ("HOLD","COVER"):
        eq_hist = get_equity_history()
        if len(eq_hist) >= 3 and market != "crypto":
            # Dynamic sizing for stocks: risk_appetite scales down during drawdowns (Packt Ch8)
            base_qty = dynamic_position_size(cash, ind["price"], ind["atr14"], eq_hist)
            scale    = (confidence - MIN_CONFIDENCE) / (100 - MIN_CONFIDENCE)
            qty = max(1, int(base_qty * max(0.3, min(1.0, scale))))
        else:
            qty = size_position(ind["price"], ind["atr14"], cash, confidence, market)
        if qty * ind["price"] > MAX_TRADE_USD:
            qty = round(MAX_TRADE_USD / ind["price"], 6)

    return {"action": action, "quantity": qty, "confidence": confidence,
            "reason": str(dec.get("reason","")).strip()[:200], "reasoning": reasoning}


# execute_trade → broker.ai4trade


# get_positions_api → broker.ai4trade

# ══════════════════════════════════════════════════════════════════════════════
# EXCEL + LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def export_excel(token, cash):
    data = get_positions_api(token)
    positions = data.get("positions", [])
    rows = []
    for p in positions:
        qty   = float(p.get("quantity",0))
        price = float(p.get("current_price") or p.get("price",0))
        value = qty * price
        cost  = float(p.get("avg_cost") or p.get("cost_basis",price)) * qty
        pnl   = value - cost
        # Attach stop/target from local tracking
        local = load_positions().get(p.get("symbol",""), {})
        rows.append({
            "Symbol": p.get("symbol",""), "Market": p.get("market",""),
            "Qty": qty, "Entry ($)": round(cost/qty if qty else 0,4),
            "Current ($)": round(price,4), "Value ($)": round(value,2),
            "P&L ($)": round(pnl,2), "P&L (%)": round(pnl/cost*100 if cost else 0,2),
            "Stop ($)": local.get("stop_price",""), "Target ($)": local.get("target_price",""),
        })
    df_pos = pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
        "Symbol","Market","Qty","Entry ($)","Current ($)","Value ($)",
        "P&L ($)","P&L (%)","Stop ($)","Target ($)"])
    hv = sum(r["Value ($)"] for r in rows)
    df_sum = pd.DataFrame([{
        "Last Updated": datetime.now().strftime("%Y-%m-%d %H:%M IST"),
        "Cash ($)": round(cash,2), "Holdings ($)": round(hv,2),
        "Total ($)": round(cash+hv,2),
        "P&L vs $100K ($)": round(cash+hv-100_000,2),
        "P&L vs $100K (%)": round((cash+hv)/100_000*100-100,2),
    }])
    with pd.ExcelWriter(EXCEL, engine="openpyxl") as w:
        df_sum.to_excel(w, sheet_name="Summary",  index=False)
        df_pos.to_excel(w, sheet_name="Holdings", index=False)
        for sn, df in [("Summary",df_sum),("Holdings",df_pos)]:
            ws = w.sheets[sn]
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = min(
                    max(len(str(c.value or "")) for c in col)+4, 30)
    return len(positions)


def append_log(entry):
    logs = json.loads(LOG.read_text()) if LOG.exists() else []
    logs.append(entry)
    if len(logs) > LOG_MAX: logs = logs[-LOG_MAX:]
    LOG.write_text(json.dumps(logs, indent=2))


# ══════════════════════════════════════════════════════════════════════════════
# DAILY EMAIL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def send_daily_summary(cfg, cash, trades_today, positions_count):
    gmail_pass = cfg.get("gmail_app_password","")
    if not gmail_pass:
        return  # skip if not configured

    hist  = json.loads(TRADE_HIST.read_text()) if TRADE_HIST.exists() else []
    wins  = sum(1 for t in hist if t["outcome"]=="WIN")
    total = len(hist)
    pnl   = round(cash - 100_000, 2)
    sign  = "+" if pnl >= 0 else ""

    body = f"""AI-Trader Daily Summary — {datetime.now().strftime('%Y-%m-%d')}

Portfolio: ${cash:,.2f}  ({sign}${pnl:,.2f} vs $100K start)
Open positions: {positions_count}
Trades today: {len(trades_today)}
All-time: {wins}/{total} wins ({round(wins/total*100) if total else 0}% win rate)

Today's trades:
"""
    for t in trades_today:
        body += f"  {t['action']} {t['quantity']} {t['symbol']} — {t['reason']} (confidence {t['confidence']}%)\n"

    if not trades_today:
        body += "  No trades executed today (all HOLD or market closed)\n"

    body += "\nDashboard: https://ai4trade.ai/agent/10954"

    msg = MIMEText(body)
    msg["Subject"] = f"AI-Trader: ${cash:,.0f} | {sign}{pnl:,.0f} P&L"
    msg["From"]    = cfg["email"]
    msg["To"]      = cfg["email"]

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(cfg["email"], gmail_pass)
            s.send_message(msg)
        print(f"  Email sent → {cfg['email']}")
    except Exception as e:
        print(f"  [warn] Email failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_short_put_strategy(cfg: dict, watchlist: list, macro: dict,
                           market_open: bool, dry_run: bool) -> list:
    """
    Short put strategy loop — runs after the equity loop.
    1. Close positions hitting 50%-profit or DTE≤7.
    2. Screen US stocks for new entries (market must be open).
    Returns list of trade dicts for summary reporting.
    """
    trades = []

    # ── Check exits ───────────────────────────────────────────────────────────
    for pos in sp_check_exits():
        current = pos.get("current_premium") or pos.get("entry_premium", 0) * 0.01
        reason  = pos.get("close_reason", "exit")
        result  = close_short_put(cfg, pos, current, dry_run)
        if "error" in result:
            print(f"   [SP] Close {pos['symbol']} failed: {result['error']}")
        else:
            pnl = result.get("pnl", 0)
            print(f"   [SP] CLOSED {pos['symbol']} ${pos['strike']}P — {reason} | P&L ${pnl:+.0f}")
            trades.append({"action": "CLOSE_PUT", "symbol": pos["symbol"],
                           "reason": reason, "pnl": pnl})

    if not market_open:
        return trades

    # ── Screen for new entries ────────────────────────────────────────────────
    vix = macro.get("vix")
    us_stocks = [item for item in watchlist if item.get("market") == "us-stock"]

    for item in us_stocks:
        symbol = item["symbol"]
        try:
            df   = yf.download(symbol, period="1y", interval="1d", progress=False)
            if df.empty or len(df) < 50:
                print(f"   [SP] {symbol} skipped — not enough data")
                continue
            ind  = compute_indicators(df)
            fund = get_fundamentals(symbol, "us-stock")
            rsi  = ind.get("rsi14", 0)
            abv  = ind.get("above_sma50", False)
            earn = fund.get("days_to_earnings")

            # Regime filter (Packt Ch6 regime_breakout): skip puts in bear trends
            # Bull = SMA50 > SMA200 (golden cross) AND price above SMA50
            c = df["Close"].squeeze()
            sma200 = float(c.rolling(200).mean().iloc[-1]) if len(c) >= 200 else None
            regime_bull = (sma200 is not None and ind["sma50"] > sma200 and abv)
            sma200_str = f"{sma200:.0f}" if sma200 else "N/A"
            print(f"   [SP] {symbol} RSI={rsi:.1f} above_sma50={abv} "
                  f"sma200={sma200_str} regime={'BULL' if regime_bull else 'BEAR'} "
                  f"earnings={earn}d")
            if not regime_bull:
                print(f"   [SP] {symbol} skipped — bear regime (SMA50 < SMA200 or price < SMA50)")
                continue
            opp  = find_short_put_opportunity(symbol, ind, fund, vix)
            if not opp:
                continue

            # Only sell puts on LONG-signal stocks (top quintile relative strength)
            rank_info = cfg.get("_rank_map", {}).get(symbol, {})
            if rank_info.get("signal") == "SHORT":
                print(f"   [SP] {symbol} skipped — ranked SHORT (relative weakness, bad put-sell)")
                continue

            rank_tag = f" rank#{rank_info.get('rank','?')}({rank_info.get('signal','?')})" if rank_info else ""
            print(f"   [SP] {symbol}{rank_tag} ${opp['strike']}P exp {opp['expiry']} "
                  f"({opp['dte']}DTE, {opp['otm_pct']}% OTM) "
                  f"bid ${opp['premium']} → credit ${opp['credit']}")

            result = execute_short_put(cfg, opp, dry_run)
            if "error" in result:
                print(f"        ✗ {result['error']}")
            else:
                tag = "[DRY]" if dry_run else f"[{result.get('status','')}]"
                print(f"        ✓ {tag} order {str(result.get('alpaca_order_id',''))[:8]}")
                trades.append({"action": "SHORT_PUT", "symbol": symbol,
                               "strike": opp["strike"], "expiry": opp["expiry"],
                               "premium": opp["premium"], "credit": opp["credit"]})
        except Exception as e:
            print(f"   [SP] {symbol} skipped: {e}")

    return trades


def run_india_short_put_strategy(cfg: dict, watchlist: list,
                                  india_vix: float | None,
                                  market_open: bool, dry_run: bool) -> list:
    """
    Short put strategy for NSE individual stocks.
    Mirrors the US loop but uses NSE option chains + Zerodha execution.
    """
    trades = []

    # ── Close exits first ─────────────────────────────────────────────────────
    for pos in check_india_exits():
        current = pos.get("current_premium")
        reason  = pos.get("close_reason", "exit")
        result  = close_india_short_put(cfg, pos, current, dry_run)
        if "error" in result:
            print(f"   [IN-SP] Close {pos['symbol']} failed: {result['error']}")
        else:
            pnl = result.get("pnl_inr", 0)
            print(f"   [IN-SP] CLOSED {pos['symbol']} ₹{pos['strike']}P — "
                  f"{reason} | P&L ₹{pnl:+.0f}")
            trades.append({"action": "CLOSE_PUT_IN", "symbol": pos["symbol"],
                           "reason": reason, "pnl": pnl})

    if not market_open:
        return trades

    # ── Screen Indian stocks for new entries ──────────────────────────────────
    india_stocks = [i for i in watchlist if i.get("market") == "in-stock"]

    for item in india_stocks:
        ns_sym    = item["symbol"]           # e.g. "RELIANCE.NS"
        nse_sym   = ns_sym.replace(".NS", "") # e.g. "RELIANCE"
        try:
            df = yf.download(ns_sym, period="1y", interval="1d", progress=False)
            if df.empty or len(df) < 50:
                print(f"   [IN-SP] {nse_sym} skipped — not enough data")
                continue
            ind  = compute_indicators(df)
            fund = get_fundamentals(ns_sym, "in-stock")

            # Regime filter: SMA50 > SMA200 required (same as US)
            c      = df["Close"].squeeze()
            sma200 = float(c.rolling(200).mean().iloc[-1]) if len(c) >= 200 else None
            abv50  = ind.get("above_sma50", False)
            rsi    = ind.get("rsi14", 0)
            regime_bull = sma200 is not None and ind["sma50"] > sma200 and abv50
            sma200_str = f"{sma200:.0f}" if sma200 else "N/A"

            print(f"   [IN-SP] {nse_sym} RSI={rsi:.1f} sma200={sma200_str} "
                  f"regime={'BULL' if regime_bull else 'BEAR'}")
            if not regime_bull:
                print(f"   [IN-SP] {nse_sym} skipped — bear regime")
                continue

            opp = find_india_short_put_opportunity(nse_sym, ind, fund, india_vix)
            if not opp:
                print(f"   [IN-SP] {nse_sym} — no opportunity found")
                continue

            print(f"   [IN-SP] {nse_sym} ₹{opp['strike']}P exp {opp['expiry_str']} "
                  f"({opp['dte']}DTE, {opp['otm_pct']}% OTM) "
                  f"bid ₹{opp['premium']:.1f}/sh → credit ₹{opp['credit_inr']:.0f}")

            result = execute_india_short_put(cfg, opp, dry_run)
            if "error" in result:
                print(f"        ✗ {result['error']}")
            else:
                tag = "[DRY]" if dry_run else f"[{result.get('status','')}]"
                print(f"        ✓ {tag}")
                trades.append({"action": "SHORT_PUT_IN", "symbol": nse_sym,
                               "strike": opp["strike"], "expiry": opp["expiry"],
                               "credit_inr": opp["credit_inr"]})
        except Exception as e:
            print(f"   [IN-SP] {nse_sym} skipped: {e}")

    return trades


def main():
    dry_run     = "--dry-run" in sys.argv
    force       = "--force"   in sys.argv   # bypass market-hours check
    cfg         = load_config()
    _, _, label = detect_llm_backend()
    # Pass fast model key to debate brain (avoids 50 req/day gpt-4o limit)
    cfg["_fast_model"] = GITHUB_MODEL_FAST

    print(f"\n{'='*62}")
    print(f"  AI-Trader Autonomous  |  {datetime.now().strftime('%Y-%m-%d %H:%M IST')}")
    print(f"  Brain  : {label}")
    print(f"  Mode   : {'DRY RUN' if dry_run else 'PAPER TRADING'}")
    print(f"  Market : {'OPEN' if is_market_open() else 'CLOSED'}")
    print(f"{'='*62}\n")

    # Market hours gate — crypto still runs always
    market_open = is_market_open()

    print("Authenticating...")
    token   = auth(cfg)
    profile = get_profile(token)
    cash    = float(profile.get("cash", 100_000))
    print(f"Account : {profile.get('name')}  |  Cash: ${cash:,.2f}\n")

    # ── Drawdown circuit breaker (persists across runs via portfolio_hwm.json) ─
    dd_halted, port_value, dd_pct, peak = check_drawdown_circuit(token, cash)
    dd_msg = (f"Portfolio: ${port_value:,.0f}  Peak: ${peak:,.0f}  "
              f"Drawdown: {dd_pct*100:.1f}%")
    if dd_halted:
        print(f"  *** CIRCUIT BREAKER ACTIVE *** {dd_msg}")
        print(f"  All new BUY orders halted until portfolio recovers above "
              f"${peak * (1 - DRAWDOWN_HALT_PCT):,.0f} (90% of peak)\n")
    else:
        print(f"  Circuit breaker: OK  |  {dd_msg}\n")

    print("Macro + Regime context...")
    macro  = get_macro()
    regime = get_full_regime_context()
    cfg["_regime"] = regime

    try:
        bot_score = get_market_regime_score()
        cfg["_bot_score"] = bot_score
        hint = bot_score.get("strategy_hint", "neutral")
        score = bot_score.get("score", 50)
        print(f"  {regime_fmt(regime)} | BotScore:{score} ({hint})")
    except Exception:
        cfg["_bot_score"] = {}
        print(f"  {regime_fmt(regime)}")
    if macro.get("vix"):
        print(f"  VIX {macro['vix']} ({macro['vix_level']}) | "
              f"QQQ {macro.get('qqq_5d_pct','?')}% | SPY {macro.get('spy_5d_pct','?')}%")

    # Cross-sectional ranking (Auquan + Packt Ch4) — ranks US stocks vs SPY
    print("Computing cross-sectional rankings...")
    try:
        rank_map = rank_watchlist(cfg.get("watchlist", []))
        cfg["_rank_map"] = rank_map
        top = sorted(rank_map.items(), key=lambda x: x[1]["rank"])
        summary = "  ".join(f"{s}({d['signal'][0]}z{d['score']:+.1f})" for s, d in top)
        print(f"  Rankings: {summary}")
    except Exception as e:
        cfg["_rank_map"] = {}
        print(f"  [warn] Ranking failed: {e}")

    # Event blackout — skip ALL new positions
    if regime.get("event_blackout"):
        print(f"\n  ⚠ EVENT BLACKOUT: {regime['event_name']} — no new positions this run\n")

    # Dynamic screener — extend watchlist with squeeze candidates from wider universe
    try:
        fixed_syms = [i["symbol"] for i in cfg.get("watchlist", [])]
        screener_hits = get_screener_candidates(top_n=3, exclude=fixed_syms)
        if screener_hits:
            cfg["watchlist"] = cfg.get("watchlist", []) + screener_hits
            hits_str = ", ".join(h["symbol"] for h in screener_hits)
            print(f"  [SCREENER] Added {len(screener_hits)} squeeze candidates: {hits_str}")
    except Exception as _e:
        print(f"  [warn] Screener failed: {_e}")

    print()

    # Fetch all signal stacks in parallel (was sequential — ~7s saved per run)
    all_symbols = [item["symbol"] for item in cfg.get("watchlist", [])]
    print("Fetching signal stack (social, WSB, Congress, earnings)...")

    def _safe(fn, tickers):
        try:
            return fn(watchlist_tickers=tickers)
        except Exception:
            return []

    def _safe_earnings(item):
        try:
            return item["symbol"], get_earnings_info(item["symbol"], item["market"])
        except Exception:
            return item["symbol"], {}

    has_india = any(i["market"] == "in-stock" for i in cfg.get("watchlist", []))

    def _safe_india_bulk(tickers):
        try:
            return get_nse_bulk_deals(tickers)
        except Exception:
            return []

    def _safe_india_vix():
        try:
            return get_india_vix()
        except Exception:
            return {}

    with ThreadPoolExecutor(max_workers=8) as _ex:
        _t_social  = _ex.submit(_safe, get_social_signals,         all_symbols)
        _t_wsb     = _ex.submit(_safe, get_all_social,              all_symbols)
        _t_insider = _ex.submit(_safe, get_all_insider_signals,     all_symbols)
        _t_news    = _ex.submit(_safe, get_news_signals,            all_symbols)
        _t_options = _ex.submit(_safe, get_options_signals,         all_symbols)
        _t_whale   = _ex.submit(_safe, get_whale_signals,           all_symbols)
        _t_fg      = _ex.submit(_safe, get_fear_greed_signals,      all_symbols)
        _t_poly    = _ex.submit(_safe, get_prediction_signals,      all_symbols)
        _t_earn    = {_ex.submit(_safe_earnings, item): item for item in cfg.get("watchlist", [])}
        if has_india:
            _t_bulk_deals = _ex.submit(_safe_india_bulk, all_symbols)
            _t_india_vix  = _ex.submit(_safe_india_vix)

        social_signals  = _t_social.result()
        wsb_signals     = _t_wsb.result()
        insider_signals = _t_insider.result()
        news_signals    = _t_news.result()
        options_signals = _t_options.result()
        whale_signals   = _t_whale.result()
        fg_signals      = _t_fg.result()
        poly_signals    = _t_poly.result()
        earnings_map    = {fut.result()[0]: fut.result()[1] for fut in _t_earn}
        bulk_deals      = _t_bulk_deals.result() if has_india else []
        cfg["_bulk_deals"]  = bulk_deals
        cfg["_india_vix"]   = _t_india_vix.result() if has_india else {}

    total_signals = (len(social_signals) + len(wsb_signals) + len(insider_signals)
                     + len(news_signals) + len(options_signals) + len(whale_signals))
    print(f"  {total_signals} signals | {len(insider_signals)} govt/insider | "
          f"{len(wsb_signals)} WSB | {len(social_signals)} celebrity | "
          f"{len(news_signals)} news/M&A | {len(options_signals)} options | "
          f"{len(whale_signals)} whale/13F")
    print(f"  {fg_summary()}")

    # Build per-symbol context strings
    cfg["_social_context"]   = {s: social_fmt(social_signals, s) + "\n" + wsb_fmt(wsb_signals, s)
                                 for s in all_symbols}
    cfg["_insider_context"]  = {s: insider_fmt(insider_signals, s) for s in all_symbols}
    cfg["_news_context"]     = {s: news_fmt(news_signals, s) for s in all_symbols}
    cfg["_options_context"]  = {s: options_fmt(options_signals, s) for s in all_symbols}
    cfg["_whale_context"]    = {s: whale_fmt(whale_signals, s) for s in all_symbols}
    cfg["_fg_context"]       = {s: fg_fmt(fg_signals, s) for s in all_symbols}
    cfg["_poly_context"]     = {s: poly_fmt(poly_signals, s) for s in all_symbols}
    cfg["_earnings_map"]    = earnings_map
    poly_count = len([s for s in poly_signals if s.get("symbol") != "_macro"])
    macro_count = len([s for s in poly_signals if s.get("symbol") == "_macro"])
    if poly_signals:
        print(f"  [Polymarket] {poly_count} asset markets + {macro_count} macro markets loaded")
    print()

    local_positions = load_positions()
    trades_today    = []

    # ── Win-rate calibration (once per run, passed to arbiter) ───────────────
    win_rate_data    = get_win_rate(n=20)
    win_rate_summary = (win_rate_data.get("summary", "")
                        if win_rate_data.get("win_rate") is not None else "")

    # Pairs mean reversion check (once per run)
    try:
        pairs_sigs = get_pairs_signals()
        if pairs_sigs:
            for ps in pairs_sigs:
                print(f"   [PAIRS] {ps['reason']}")
    except Exception:
        pass

    for item in cfg.get("watchlist", []):
        symbol = item["symbol"]
        market = item["market"]

        # Crypto trades 24/7; stocks only during market hours
        if market == "us-stock" and not market_open and not force:
            continue
        if market == "in-stock" and not is_india_market_open() and not force:
            continue

        print(f"── {symbol} ({market})")
        try:
            if market == "crypto":
                ticker = f"{symbol}-USD"
            elif market == "in-stock":
                ticker = symbol  # already RELIANCE.NS etc
            else:
                ticker = symbol
            df     = yf.download(ticker, period="90d", interval="1d",
                                  progress=False, auto_adjust=True)
            ind    = compute_indicators(df)
            # Multi-timeframe: 4h RSI confirmation
            try:
                df_4h = yf.download(symbol if "." in symbol or market == "crypto"
                                     else symbol, period="60d", interval="1h",
                                     auto_adjust=True, progress=False)
                if not df_4h.empty and len(df_4h) >= 14:
                    c4 = df_4h["Close"].squeeze()
                    d4 = c4.diff()
                    g4 = d4.clip(lower=0).rolling(14).mean()
                    l4 = (-d4.clip(upper=0)).rolling(14).mean()
                    ll4 = float(l4.iloc[-1])
                    ind["rsi_4h"] = round(100.0 if ll4 == 0 else float(100 - 100 / (1 + g4.iloc[-1] / ll4)), 1)
            except Exception:
                pass
            if market == "in-stock":
                nse_sym = symbol.replace(".NS", "")
                options_ctx = get_nse_options_flow(nse_sym)
                insider_ctx = format_bulk_deals_for_llm(cfg.get("_bulk_deals", []), symbol)
                cfg["_insider_context"][symbol] = insider_ctx
                cfg["_options_context"][symbol] = options_ctx
            fund   = get_fundamentals(symbol, market)

            price = ind["price"]
            sq_tag = "  🔥SQ-RELEASE" if ind.get("squeeze_released") else ("  [SQ]" if ind.get("bb_squeeze") else "")
            st_arrow = "▲" if ind.get("supertrend", 1) == 1 else "▼"
            rsi4h_str = f"  4h-RSI:{ind['rsi_4h']}" if ind.get("rsi_4h") else ""
            print(f"   ${price}  RSI {ind['rsi14']}  ADX {ind.get('adx','?')}  ST{st_arrow}  "
                  f"StochRSI:{ind.get('stochrsi_k','?')}  "
                  f"MACD {'▲' if ind['macd_hist']>0 else '▼'}{abs(round(ind['macd_hist'],2))}  "
                  f"BB {int(ind['bb_position']*100)}%  ATR ${ind['atr14']}{sq_tag}{rsi4h_str}")

            # ── Stop-loss / profit target check ──────────────────────────────
            stop_signal = check_stops(price, symbol)
            if stop_signal and symbol in local_positions:
                if stop_signal == "PARTIAL_PROFIT":
                    # Sell 50%, move stop to breakeven, keep riding the rest
                    full_qty = local_positions[symbol]["quantity"]
                    half_qty = max(1, int(full_qty * 0.5)) if market != "crypto" else round(full_qty * 0.5, 6)
                    reason = f"AUTO PARTIAL: 50% exit at ${price} (3.75×ATR gain) — stop → breakeven"
                    print(f"   >>> PARTIAL PROFIT — selling {half_qty} of {full_qty} @ ${price}")
                    result = execute_trade(token, symbol, market, "SELL", half_qty, reason, dry_run)
                    if result and not dry_run:
                        mark_partial_done(symbol)
                        local_positions[symbol]["quantity"] = full_qty - half_qty
                    trades_today.append({"symbol":symbol,"action":"SELL","quantity":half_qty,
                                          "reason":reason,"confidence":100,"result":result})
                    print()
                    time.sleep(0.5)
                    continue
                if stop_signal == "STOP_LOSS":
                    reason = f"AUTO STOP-LOSS: price ${price} hit stop ${local_positions[symbol]['stop_price']}"
                elif stop_signal == "TRAIL_STOP":
                    reason = f"AUTO TRAIL-STOP: price ${price} hit trailing stop ${local_positions[symbol].get('trail_stop_price', local_positions[symbol]['stop_price'])}"
                else:
                    reason = f"AUTO PROFIT TARGET: price ${price} hit target ${local_positions[symbol]['target_price']}"
                print(f"   >>> {stop_signal} triggered! {reason}")
                qty = local_positions[symbol]["quantity"]
                result = execute_trade(token, symbol, market, "SELL", qty, reason, dry_run)
                if result and not dry_run:
                    record_close(symbol, price)
                trades_today.append({"symbol":symbol,"action":"SELL","quantity":qty,
                                      "reason":reason,"confidence":100,"result":result})
                print()
                time.sleep(0.5)
                continue

            # ── Earnings blackout (hard rule) ────────────────────────────────
            earn_info = cfg.get("_earnings_map", {}).get(symbol, {})
            earn_str  = earn_fmt(earn_info)
            _, earn_delta = earnings_signal(earn_info)
            if earn_delta <= -50:   # blackout zone
                print(f"   HOLD [earnings blackout — {earn_info.get('days_to_earnings','?')}d]\n")
                continue
            if earn_info:
                print(f"   {earn_str}")

            # ── Block new longs in crisis/event blackout ──────────────────────
            regime = cfg.get("_regime", {})
            if regime.get("event_blackout") and symbol not in local_positions:
                print(f"   SKIP — event blackout active\n")
                continue

            # ── Multi-agent debate brain ──────────────────────────────────────
            has_pos      = symbol in local_positions
            social_ctx   = cfg.get("_social_context",{}).get(symbol,"")
            ins_ctx      = cfg.get("_insider_context",{}).get(symbol,"")
            news_ctx     = cfg.get("_news_context",{}).get(symbol,"")
            options_ctx  = cfg.get("_options_context",{}).get(symbol,"")
            whale_ctx    = cfg.get("_whale_context",{}).get(symbol,"")
            fg_ctx       = cfg.get("_fg_context",{}).get(symbol,"")
            poly_ctx     = cfg.get("_poly_context",{}).get(symbol,"")
            earn_info    = cfg.get("_earnings_map",{}).get(symbol,{})
            earn_s       = earn_fmt(earn_info)
            rank_ctx     = get_rank_context(symbol, cfg.get("_rank_map", {}))

            try:
                rag_ctx = get_rag_context(symbol, market, ind)
            except Exception:
                rag_ctx = ""

            try:
                bb_pat = detect_bb_pattern(df, ind)
                bb_pattern_ctx = f"{bb_pat['pattern']} conf={bb_pat['confidence']}% — {bb_pat['description']}" if bb_pat['pattern'] != 'NONE' else ""
            except Exception:
                bb_pattern_ctx = ""

            try:
                dec = debate_decide(symbol, market, ind, fund, macro,
                                    cash, social_ctx, ins_ctx, earn_s, cfg,
                                    news_ctx=news_ctx, options_ctx=options_ctx,
                                    whale_ctx=whale_ctx, fg_ctx=fg_ctx,
                                    rag_ctx=rag_ctx,
                                    bb_pattern_ctx=bb_pattern_ctx,
                                    win_rate_summary=win_rate_summary,
                                    rank_ctx=rank_ctx,
                                    poly_ctx=poly_ctx)
                print(f"   [{dec.get('consensus','?')} consensus]  "
                      f"Bull: {dec.get('bull_arg','')[:60]}...")
                print(f"   Bear: {dec.get('bear_arg','')[:60]}...")
            except Exception as e:
                # fallback to single LLM if debate fails
                dec = llm_decide(symbol, market, ind, fund, macro, cash, cfg, has_pos)
            action, qty, conf, reason = dec["action"], dec["quantity"], dec["confidence"], dec["reason"]
            reasoning = dec["reasoning"]

            print(f"   Conf {conf}% | {reasoning.get('technical','')[:80]}")
            print(f"   Risks: {reasoning.get('risks','N/A')[:80]}")
            print(f"   → {action}", end="")

            if action != "HOLD" and qty > 0:
                # ── Drawdown circuit breaker: block new BUY only ──────────────
                if action == "BUY" and dd_halted:
                    print(f"  BLOCKED by drawdown circuit breaker "
                          f"(portfolio {dd_pct*100:.1f}% below peak — halt threshold: "
                          f"-{DRAWDOWN_HALT_PCT*100:.0f}%)")
                    print()
                    time.sleep(0.5)
                    continue

                # Correlation guard — skip if we already have a correlated position open
                if action == "BUY":
                    corr_syms = get_correlated_symbols(symbol, local_positions)
                    if corr_syms:
                        print(f"   [CORR-GUARD] Skipping {symbol} — correlated (>0.7) with open: {corr_syms}")
                        print()
                        time.sleep(0.5)
                        continue

                # Apply regime size multiplier
                size_mult = regime.get("size_mult", 1.0)
                if size_mult < 1.0 and action == "BUY":
                    qty = round(qty * size_mult, 6) if market=="crypto" else max(1,int(qty*size_mult))
                    reason += f" [size×{size_mult} — {regime.get('tier','?')} regime]"

                val = qty * price
                print(f"   qty={qty}  value=${val:,.0f}  — {reason}")

                # Execute on ai4trade.ai (paper) — Indian market queued for approval
                if market == "in-stock":
                    trade_id = queue_trade(cfg, symbol, market, action, qty,
                                           price, conf, reason, ind.get("atr14", 0))
                    msg_id = send_approval_request(cfg, trade_id, action, symbol,
                                                    qty, price, conf, reason)
                    if msg_id:
                        update_message_id(trade_id, msg_id)
                    print(f"   → queued for Telegram approval (trade_id={trade_id})")
                    result       = {"queued": True, "trade_id": trade_id}
                    alpaca_result = {"skipped": "awaiting approval"}
                else:
                    result = execute_trade(token, symbol, market, action, qty, reason, dry_run)
                    # Mirror to Alpaca — BUY stocks get native bracket (stop + target)
                    alpaca_result = execute_alpaca_trade(cfg, symbol, market, action, qty, reason,
                                                         dry_run, price=price, atr=ind.get("atr14"))
                if alpaca_result.get("alpaca_order_id"):
                    bracket_info = (f" stop=${alpaca_result['stop']} target=${alpaca_result['target']}"
                                    if alpaca_result.get("stop") else "")
                    print(f"   Alpaca order: {alpaca_result['alpaca_order_id'][:8]}... "
                          f"{alpaca_result.get('status','')}{bracket_info}")

                send_trade_alert(cfg, action, symbol, qty, price, reason, conf,
                                 alpaca_order_id=alpaca_result.get("alpaca_order_id", ""))

                if result and not dry_run:
                    if action == "BUY":
                        record_open(symbol, price, qty, ind["atr14"], market)
                    elif action in ("SELL","COVER") and symbol in local_positions:
                        record_close(symbol, price)
                trades_today.append({"symbol":symbol,"action":action,"quantity":qty,
                                      "reason":reason,"confidence":conf,
                                      "result":result,"alpaca":alpaca_result})
            else:
                print(f"  — {reason}")

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"   [SKIP — bad response] {e}")
        except Exception as e:
            if "429" in str(e) or "RateLimitReached" in str(e) or "rate limit" in str(e).lower():
                # Mechanical fallback when LLM rate-limited
                mech_action = "HOLD"
                mech_reason = "LLM rate-limited — mechanical signal"
                sq_rel = ind.get("squeeze_released", False)
                adx_v  = ind.get("adx", 0)
                st_v   = ind.get("supertrend", 1)
                sk_v   = ind.get("stochrsi_k", 0.5)
                rev    = rsi_reversion_signal(ind)

                strategy_hint = cfg.get("_bot_score", {}).get("strategy_hint", "neutral")

                if strategy_hint == "mean_reversion":
                    # HUMAN_DRIVEN market — prefer RSI reversion over momentum
                    if rev and rev["action"] == "BUY" and rev["strength"] > 0.4:
                        mech_action = "BUY"
                        mech_reason = f"[HUMAN regime] {rev['reason']}"
                    elif rev and rev["action"] == "SELL" and rev["strength"] > 0.4:
                        mech_action = "SELL"
                        mech_reason = f"[HUMAN regime] {rev['reason']}"
                elif strategy_hint == "momentum" or True:
                    # BOT_DRIVEN or neutral — squeeze + trend momentum
                    if sq_rel and adx_v > 20 and ind["macd_hist"] > 0 and st_v == 1:
                        mech_action = "BUY"
                        mech_reason = f"Squeeze release + ADX={adx_v} + Supertrend bull + MACD positive"
                    elif st_v == -1 and sk_v > 0.8:
                        mech_action = "SELL"
                        mech_reason = f"Supertrend bear + StochRSI overbought ({sk_v})"
                    elif rev and rev["action"] == "BUY" and rev["strength"] > 0.5:
                        mech_action = "BUY"
                        mech_reason = rev["reason"]

                print(f"   [MECH-FALLBACK] {mech_action} — {mech_reason}")
                action, confidence, reason = mech_action, 55 if mech_action != "HOLD" else 0, mech_reason
            else:
                print(f"   [SKIP] {e}")

        print()
        time.sleep(1)

    # ── Short put strategy (US) ───────────────────────────────────────────────
    print("── Short Put Strategy (US) ──")
    sp_trades = run_short_put_strategy(cfg, cfg.get("watchlist", []), macro,
                                       market_open, dry_run)
    trades_today.extend(sp_trades)
    print()

    # ── Short put strategy (India) ────────────────────────────────────────────
    india_market_open = is_india_market_open() or force
    print("── Short Put Strategy (India) ──")
    india_vix_val = cfg.get("_india_vix", {})
    if isinstance(india_vix_val, dict):
        india_vix_val = india_vix_val.get("vix")   # {"vix": 16.5, "level": "NORMAL"}
    in_sp_trades = run_india_short_put_strategy(
        cfg, cfg.get("watchlist", []), india_vix_val, india_market_open, dry_run)
    trades_today.extend(in_sp_trades)
    print()

    # ── Post-run ──────────────────────────────────────────────────────────────
    print(f"{'='*62}")
    print(f"  Trades today : {len(trades_today)}")
    print(f"  Open positions tracked : {len(load_positions())}")
    print(f"  Dashboard : https://ai4trade.ai/agent/10954")
    print(f"{'='*62}\n")

    if trades_today:
        append_log({"run_at": datetime.now(timezone.utc).isoformat(),
                    "dry_run": dry_run, "cash": cash,
                    "macro": macro, "trades": trades_today})

    try:
        n = export_excel(token, cash)
        print(f"Excel updated — {n} positions")
    except Exception as e:
        print(f"[warn] Excel: {e}")

    # Per-run Telegram heartbeat — always fires so user sees the bot is alive
    sp_open = len(sp_load_positions()) + len(india_sp_load_positions())
    send_run_status(cfg, cash, trades_today, len(load_positions()), sp_open)

    # Daily summary email at market close
    if is_near_close() and not dry_run:
        print("Near market close — sending daily summary...")
        send_daily_summary(cfg, cash, trades_today, len(load_positions()))
        _, port_val, dd_pct, _ = check_drawdown_circuit(token, cash)
        tg_daily_summary(cfg, trades_today, cash, port_val, dd_pct * 100)


if __name__ == "__main__":
    main()
