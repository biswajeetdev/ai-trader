"""
finbert_sentiment.py — Domain-tuned financial sentiment scoring.

Primary: ProsusAI/finbert (transformers) — if installed
Fallback: Loughran-McDonald financial keyword scoring — zero deps

Returns structured sentiment: {positive, negative, neutral, label, confidence}
Cached 4 hours to avoid re-running the model.
"""

import re, json, time, hashlib
from pathlib import Path

try:
    from transformers import pipeline as hf_pipeline
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

CACHE_FILE = Path(__file__).parent.parent / "signals_cache.json"
CACHE_TTL  = 4 * 3600

LM_POSITIVE = {
    "beat", "exceed", "strong", "growth", "profit", "surge", "upgrade",
    "bullish", "record", "outperform", "raise", "momentum",
    "acquisition", "expand", "partnership", "innovative", "robust",
    "rebound", "recovery", "optimistic", "dividend", "breakout",
    "accelerate", "milestone", "opportunity", "efficiency", "strength",
    "gain", "improve", "higher", "increase", "advance",
    "solid", "boost", "deliver", "achieve", "confidence",
}
LM_NEGATIVE = {
    "miss", "below", "weak", "loss", "decline", "downgrade", "bearish",
    "cut", "withdraw", "layoff", "investigation", "lawsuit",
    "recall", "bankruptcy", "fraud", "restructure", "headwinds",
    "shortfall", "disappoint", "concern", "risk", "uncertain",
    "selloff", "volatile", "pressure", "deteriorate", "debt",
    "violation", "penalty", "warning", "slowdown", "default",
    "deficit", "negative", "lower", "decrease", "reduce", "failure",
}

_model = None  # module-level singleton


def _get_model():
    global _model
    if _model is None and _HF_AVAILABLE:
        _model = hf_pipeline(
            "text-classification", model="ProsusAI/finbert", return_all_scores=True
        )
    return _model


def _cache_key(texts: list) -> str:
    return "finbert_" + hashlib.md5(str(frozenset(texts)).encode()).hexdigest()[:12]


def _load_cached(key: str):
    if not CACHE_FILE.exists():
        return None
    try:
        data = json.loads(CACHE_FILE.read_text())
        entry = data.get(key)
        if entry and time.time() - entry.get("_ts", 0) < CACHE_TTL:
            return entry["result"]
    except Exception:
        pass
    return None


def _save_cached(key: str, result: dict) -> None:
    try:
        data = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
    except Exception:
        data = {}
    data[key] = {"_ts": time.time(), "result": result}
    try:
        CACHE_FILE.write_text(json.dumps(data, indent=2))
    except Exception:
        pass


def _keyword_sentiment(texts: list) -> dict:
    """Loughran-McDonald keyword fallback."""
    pos = neg = 0
    for t in texts:
        words = set(re.findall(r"\b\w+\b", t.lower()))
        pos += len(words & LM_POSITIVE)
        neg += len(words & LM_NEGATIVE)
    total = pos + neg or 1
    p, n = pos / total, neg / total
    neu = max(0.0, round(1.0 - p - n, 3))
    label = "positive" if p > n else "negative" if n > p else "neutral"
    return {"positive": round(p, 3), "negative": round(n, 3),
            "neutral": neu, "label": label,
            "confidence": round(max(p, n, neu), 3), "_method": "keyword"}


def get_finbert_sentiment(texts: list) -> dict:
    """Run FinBERT (or keyword fallback) on up to 10 texts. Cached 4h."""
    texts = [t for t in texts if t.strip()][:10]
    if not texts:
        return {"positive": 0.0, "negative": 0.0, "neutral": 1.0,
                "label": "neutral", "confidence": 1.0, "_method": "empty"}

    key = _cache_key(texts)
    cached = _load_cached(key)
    if cached:
        return cached

    model = _get_model()
    if model:
        try:
            scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
            for out in model(texts, truncation=True, max_length=512):
                for item in out:
                    scores[item["label"].lower()] += item["score"]
            n = len(texts)
            pos, neg, neu = scores["positive"] / n, scores["negative"] / n, scores["neutral"] / n
            label = max(scores, key=scores.get)
            result = {"positive": round(pos, 3), "negative": round(neg, 3),
                      "neutral": round(neu, 3), "label": label,
                      "confidence": round(max(pos, neg, neu), 3), "_method": "finbert"}
            _save_cached(key, result)
            return result
        except Exception:
            pass

    result = _keyword_sentiment(texts)
    _save_cached(key, result)
    return result


def format_for_llm(sentiment: dict, symbol: str) -> str:
    """Format sentiment dict as a single LLM-context line."""
    label    = sentiment["label"].upper()
    conf_pct = int(sentiment["confidence"] * 100)
    if sentiment.get("_method") == "finbert":
        return f"[FINBERT] {symbol} → {label} ({conf_pct}% conf) — domain-tuned financial NLP"
    return f"[KEYWORD-SENT] {symbol} → {label} (keyword scoring)"


def score_news_headlines(news_ctx: str, symbol: str) -> str:
    """Split news_ctx into sentences, score, return formatted string."""
    sentences = [s.strip() for s in re.split(r"[.\n]+", news_ctx) if len(s.strip()) > 10]
    sentiment = get_finbert_sentiment(sentences)
    return format_for_llm(sentiment, symbol)
