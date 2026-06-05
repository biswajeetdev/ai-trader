"""
fear_greed.py — Market sentiment index
Sources (all free, no API key):
  CNN Fear & Greed  — stocks composite (momentum, safe haven, junk bonds, VIX, put/call)
  Alternative.me    — crypto Fear & Greed index
  VIX derived       — fallback computed from live VIX

Extreme fear (<20)  → contrarian BUY signal (market oversold)
Fear (20-40)        → cautious bullish
Greed (60-80)       → cautious / reduce size
Extreme greed (>80) → SELL signal / take profits

Integrated as a macro overlay — adjusts confidence ±10 on LLM decisions.
"""

import json, time, requests
from datetime import datetime, timezone
from pathlib import Path

CACHE_FILE = Path(__file__).parent.parent / "signals_cache.json"
CACHE_TTL  = 30 * 60  # 30 min

CNN_URL    = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
CRYPTO_URL = "https://api.alternative.me/fng/?limit=1&format=json"


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


def _label(score):
    if score is None:
        return "UNKNOWN"
    if score <= 20:  return "EXTREME FEAR"
    if score <= 40:  return "FEAR"
    if score <= 60:  return "NEUTRAL"
    if score <= 80:  return "GREED"
    return "EXTREME GREED"


def _sentiment_from_score(score):
    """Convert F&G score to trading sentiment and confidence delta."""
    if score is None:
        return "NEUTRAL", 0.0, 0
    if score <= 20:   return "BULLISH",  0.65, +10   # extreme fear = buy
    if score <= 35:   return "BULLISH",  0.45, +5    # fear = mild buy
    if score <= 55:   return "NEUTRAL",  0.0,  0     # neutral = no adjustment
    if score <= 75:   return "BEARISH", -0.35, -5    # greed = reduce size
    return "BEARISH", -0.60, -10                      # extreme greed = sell


def get_stock_fear_greed():
    """Fetch CNN Fear & Greed index for US stocks. Falls back to VIX-derived score."""
    # Try CNN primary endpoint
    for url in [CNN_URL,
                "https://production.dataviz.cnn.io/index/fearandgreed/graphdata/past-year"]:
        try:
            r = requests.get(url, timeout=8,
                             headers={"User-Agent": "Mozilla/5.0",
                                      "Referer": "https://money.cnn.com"})
            if r.ok:
                data  = r.json()
                fg    = data.get("fear_and_greed", {})
                score = fg.get("score") or fg.get("current_score")
                if score is not None:
                    return round(float(score), 1)
        except Exception:
            pass

    # Fallback: derive from live VIX (inverse relationship)
    # VIX 10=extreme greed(90), 20=neutral(50), 30=fear(20), 40+=extreme fear(5)
    try:
        import yfinance as yf
        vix = yf.download("^VIX", period="2d", interval="1d",
                          progress=False, auto_adjust=True)
        if not vix.empty:
            v = float(vix["Close"].squeeze().iloc[-1])
            # Linear map: VIX 10→90, VIX 40→5
            score = max(5, min(95, round(90 - (v - 10) * (85 / 30), 1)))
            return score
    except Exception:
        pass
    return None


def get_crypto_fear_greed():
    """Fetch alternative.me crypto Fear & Greed index."""
    try:
        r = requests.get(CRYPTO_URL, timeout=8)
        if r.ok:
            data = r.json()
            val  = data.get("data", [{}])[0].get("value")
            if val is not None:
                return int(val)
    except Exception:
        pass
    return None


def get_fear_greed_signals(watchlist_tickers=None, use_cache=True):
    """
    Returns list of signal dicts for F&G — one for stocks, one for crypto.
    Also returns raw scores in each signal for the LLM to use as context.
    """
    cache = _load_cache() if use_cache else {}
    if cache.get("fg_signals"):
        return cache["fg_signals"]

    tickers = watchlist_tickers or []
    stock_tickers  = [t for t in tickers if t not in ("BTC", "ETH", "DOGE", "SOL")]
    crypto_tickers = [t for t in tickers if t in ("BTC", "ETH", "DOGE", "SOL")]

    signals = []
    now     = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")

    # ── Stock F&G ─────────────────────────────────────────────────────────────
    stock_score = get_stock_fear_greed()
    label       = _label(stock_score)
    direction, score, conf_delta = _sentiment_from_score(stock_score)

    if stock_tickers and stock_score is not None:
        signals.append({
            "source":      "CNN-FearGreed",
            "tickers":     stock_tickers,
            "event_type":  "FEAR_GREED",
            "direction":   direction,
            "sentiment":   score,
            "urgency":     "HIGH" if abs(score) >= 0.55 else "MEDIUM",
            "headline":    f"CNN Fear & Greed: {stock_score:.0f}/100 — {label}",
            "fg_score":    stock_score,
            "fg_label":    label,
            "conf_delta":  conf_delta,
            "pub":         now,
        })

    # ── Crypto F&G ────────────────────────────────────────────────────────────
    crypto_score = get_crypto_fear_greed()
    c_label      = _label(crypto_score)
    c_dir, c_sent, c_conf = _sentiment_from_score(crypto_score)

    if crypto_tickers and crypto_score is not None:
        signals.append({
            "source":      "AltMe-CryptoFearGreed",
            "tickers":     crypto_tickers,
            "event_type":  "CRYPTO_FEAR_GREED",
            "direction":   c_dir,
            "sentiment":   c_sent,
            "urgency":     "HIGH" if abs(c_sent) >= 0.55 else "MEDIUM",
            "headline":    f"Crypto Fear & Greed: {crypto_score}/100 — {c_label}",
            "fg_score":    crypto_score,
            "fg_label":    c_label,
            "conf_delta":  c_conf,
            "pub":         now,
        })

    cache["fg_signals"] = signals
    _save_cache(cache)
    return signals


def format_for_llm(signals, symbol):
    relevant = [s for s in signals if symbol in s.get("tickers", [])]
    if not relevant:
        return "Fear & Greed: N/A"
    s = relevant[0]
    delta_str = (f"+{s['conf_delta']}" if s['conf_delta'] >= 0 else str(s['conf_delta']))
    return (f"{s['source']}: {s['fg_score']:.0f}/100 — {s['fg_label']} "
            f"[{s['direction']}] | Confidence adjustment: {delta_str}")


def get_summary():
    """Quick summary string for printing — used in trader.py main header."""
    sigs = get_fear_greed_signals(use_cache=True)
    parts = []
    for s in sigs:
        parts.append(f"{s['source'].split('-')[0]}: {s.get('fg_score','?'):.0f} ({s['fg_label']})")
    return " | ".join(parts) if parts else "F&G: N/A"


if __name__ == "__main__":
    print("Fetching Fear & Greed indices...\n")
    stock  = get_stock_fear_greed()
    crypto = get_crypto_fear_greed()
    print(f"  Stock  F&G : {stock}  — {_label(stock)}")
    print(f"  Crypto F&G : {crypto} — {_label(crypto)}")

    sigs = get_fear_greed_signals(
        watchlist_tickers=["NVDA","AAPL","MSFT","BTC","ETH"], use_cache=False)
    for s in sigs:
        print(f"\n  {s['headline']}")
        print(f"  → {s['direction']}  sentiment={s['sentiment']:+.2f}  "
              f"conf_delta={s['conf_delta']:+d}  urgency={s['urgency']}")
