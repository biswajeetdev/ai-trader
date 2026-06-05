"""
debate_brain.py — Multi-Agent Debate Decision Engine

Three LLM agents run in parallel:
  BULL    — constructs the strongest possible bullish argument
  BEAR    — constructs the strongest possible bearish argument
  ARBITER — reads both, makes the final call with confidence score

Consensus → higher confidence → trade
No consensus → lower confidence → HOLD
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

BULL_SYSTEM = """\
You are an aggressive bull trader. Your job is to make the STRONGEST possible \
bullish case for this asset using ONLY the data provided. Cite specific numbers. \
Be concise. End with: VERDICT: BUY (confidence X%) or VERDICT: HOLD (X%).
Respond in under 80 words."""

BEAR_SYSTEM = """\
You are a disciplined short-seller. Your job is to make the STRONGEST possible \
bearish case for this asset using ONLY the data provided. Cite specific numbers. \
Be concise. End with: VERDICT: SELL (confidence X%) or VERDICT: HOLD (X%).
Respond in under 80 words."""

ARBITER_SYSTEM = """\
You are a neutral senior portfolio manager. Two analysts have debated this trade. \
Read both arguments and make the FINAL decision. Be disciplined — only act if the \
evidence clearly outweighs the risks.

SIGNAL PRIORITY ORDER (highest weight first):
1. OPTIONS FLOW — unusual call sweeps / OTM puts = smart money knows something. Highest weight.
2. WHALE/13D — activist hedge fund taking stake = near-certain catalyst. Very high weight.
3. GOVT INSIDER — Congress/Senate purchase = legal insider edge. High weight.
4. NEWS/M&A — merger target confirmed = immediate catalyst. High weight.
5. FEAR & GREED — extreme fear (<25) is contrarian BUY, extreme greed (>80) is SELL signal.
6. SOCIAL — Buffett/Musk mention. Medium weight (noise risk).
7. TECHNICALS — confirm direction but don't override strong catalyst signals.

Respond ONLY with valid JSON, no markdown:
{"action":"BUY"|"SELL"|"HOLD","confidence":<0-100>,"reason":"<one line citing top signal>","consensus":"STRONG"|"WEAK"|"SPLIT"}"""


def _call(client, model, system, user_msg, label):
    """Single LLM call — runs in thread."""
    resp = client.chat.completions.create(
        model=model, max_tokens=150, temperature=0.2,
        messages=[{"role":"system","content":system},
                  {"role":"user",  "content":user_msg}]
    )
    return label, resp.choices[0].message.content.strip()


def debate_decide(symbol, market, ind, fund, macro, cash, social_ctx, insider_ctx,
                  earn_str, cfg, news_ctx="", options_ctx="", whale_ctx="", fg_ctx="",
                  rag_ctx="", bb_pattern_ctx="", win_rate_summary="", rank_ctx="",
                  poly_ctx=""):
    """
    Run bull/bear debate in parallel, arbiter makes final call.
    Returns same format as llm_decide: {action, quantity, confidence, reason, reasoning}
    """
    client, model, _ = _get_client(cfg)

    # Build shared context block
    ctx = f"""Asset: {symbol} ({market})
Price: ${ind['price']} | RSI {ind['rsi14']} | MACD hist {ind['macd_hist']} | BB {int(ind['bb_position']*100)}%
SMA50: ${ind['sma50']} (above:{ind['above_sma50']}) | ATR ${ind['atr14']}
1d: {ind['pct_1d']}% | 5d: {ind['pct_5d']}%
"""
    if market == "us-stock":
        ctx += f"P/E: {fund.get('pe_ratio','N/A')} | EPS growth: {fund.get('eps_growth','N/A')} | Analyst: {fund.get('analyst_label','N/A')}\n"
    ctx += f"VIX: {macro.get('vix','N/A')} ({macro.get('vix_level','?')}) | QQQ 5d: {macro.get('qqq_5d_pct','N/A')}%\n"
    ctx += f"Social: {social_ctx[:200]}\nInsider: {insider_ctx[:200]}\nEarnings: {earn_str[:150]}"
    if news_ctx:
        ctx += f"\nNEWS/CATALYSTS (M&A, earnings, analyst): {news_ctx[:300]}"
    if options_ctx:
        ctx += f"\nOPTIONS FLOW (smart money): {options_ctx[:250]}"
    if whale_ctx:
        ctx += f"\nINSTITUTIONAL/WHALE (13F/13D): {whale_ctx[:200]}"
    if fg_ctx:
        ctx += f"\nFEAR & GREED: {fg_ctx[:150]}"
    if rag_ctx:
        ctx += f"\nPAST TRADE MEMORY (similar setups & outcomes): {rag_ctx[:400]}"
    if bb_pattern_ctx:
        ctx += f"\nBB PATTERN SIGNAL: {bb_pattern_ctx}"
    if rank_ctx:
        ctx += f"\nCROSS-SECTIONAL RANK (vs SPY + peers): {rank_ctx}"
    if poly_ctx:
        ctx += f"\nPREDICTION MARKETS (crowd probability): {poly_ctx[:400]}"

    # Run bull and bear in parallel
    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = {
            ex.submit(_call, client, model, BULL_SYSTEM, ctx, "bull"): "bull",
            ex.submit(_call, client, model, BEAR_SYSTEM, ctx, "bear"): "bear",
        }
        results = {}
        for fut in as_completed(futures):
            label, text = fut.result()
            results[label] = text

    bull_arg = results.get("bull", "No bullish case.")
    bear_arg = results.get("bear", "No bearish case.")

    # Arbiter reads both
    arbiter_msg = f"""BULL ANALYST says:\n{bull_arg}\n\nBEAR ANALYST says:\n{bear_arg}\n\nAsset context:\n{ctx[:400]}\nCash: ${cash:,.0f}"""

    arbiter_system = ARBITER_SYSTEM
    if win_rate_summary:
        arbiter_system = (
            f"PERFORMANCE CALIBRATION — your recent track record:\n{win_rate_summary}\n"
            f"If win rate < 45%, be MORE conservative (raise confidence threshold). "
            f"If win rate > 60%, maintain current threshold.\n\n"
            + ARBITER_SYSTEM
        )

    _, arbiter_raw = _call(client, model, arbiter_system, arbiter_msg, "arbiter")
    arbiter_raw = arbiter_raw.lstrip("```json").rstrip("```").strip()

    dec        = json.loads(arbiter_raw)
    action     = str(dec.get("action","HOLD")).upper().strip()
    confidence = int(dec.get("confidence", 0))
    reason     = str(dec.get("reason","")).strip()[:200]
    consensus  = dec.get("consensus","SPLIT")

    # Consensus modifier
    if consensus == "SPLIT":
        confidence = max(0, confidence - 20)
    elif consensus == "STRONG":
        confidence = min(100, confidence + 5)

    if action not in {"BUY","SELL","SHORT","COVER","HOLD"}:
        action = "HOLD"
    if confidence < 70:
        action = "HOLD"

    # ATR-based position sizing (same as main bot)
    from trader import size_position
    price = ind.get("price", 1)
    atr   = ind.get("atr14", price * 0.02)
    qty   = 0.0
    if action not in ("HOLD","COVER"):
        qty = size_position(price, atr, cash, confidence, market)
        max_usd = 5_000
        if qty * price > max_usd:
            qty = round(max_usd / price, 6)

    return {
        "action":     action,
        "quantity":   qty,
        "confidence": confidence,
        "reason":     reason,
        "consensus":  consensus,
        "bull_arg":   bull_arg[:200],
        "bear_arg":   bear_arg[:200],
        "reasoning":  {"technical": ctx[:200], "confidence": confidence},
    }


def _get_client(cfg):
    """Reuse backend detection from trader without circular import."""
    import os, subprocess
    from openai import OpenAI
    OLLAMA_URL = "http://localhost:11434/v1"
    try:
        import requests as _r
        r = _r.get(f"{OLLAMA_URL.replace('/v1','')}/api/tags", timeout=2)
        if r.ok:
            models = {m["name"].split(":")[0] for m in r.json().get("models",[])}
            for m in ["qwen2.5:32b","deepseek-r1:14b","llama3.1:8b"]:
                if m.split(":")[0] in models:
                    return OpenAI(base_url=OLLAMA_URL, api_key="ollama"), m, f"Ollama/{m}"
    except Exception:
        pass
    try:
        key = os.environ.get("GITHUB_TOKEN") or subprocess.check_output(["gh","auth","token"],text=True).strip()
        if key:
            # Use fast model for debate (3 calls/asset) — 50 req/min vs 50 req/day for gpt-4o
            fast = cfg.get("_fast_model", "gpt-4o-mini")
            return OpenAI(base_url="https://models.inference.ai.azure.com",api_key=key), fast, f"GitHub/{fast}"
    except Exception:
        pass
    ant = os.environ.get("ANTHROPIC_API_KEY") or cfg.get("anthropic_api_key","")
    if ant:
        return None, "claude-sonnet-4-6", "Anthropic"
    raise RuntimeError("No LLM backend")
