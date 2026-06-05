"""
pattern_memory.py — Trade pattern store + outcome tracker.

Workflow:
  1. Before decision: retrieve similar past states → LLM context
  2. After BUY/SELL: store the decision with current indicators
  3. After 5 days: update outcome (price change %) for stored patterns

The bot accumulates memory over time. After 20+ trades it starts
making meaningfully better decisions based on what actually worked.
"""

import json
import time
import requests
import yfinance as yf
from datetime import datetime, timezone, timedelta
from pathlib import Path
from rag.vector_store import (
    store_pattern, update_outcome,
    retrieve_similar_patterns, format_patterns_for_llm,
    store_news
)

PENDING_FILE = Path(__file__).parent.parent / "rag" / "pending_outcomes.json"
OUTCOME_DAYS = 5   # check price change this many trading days after trade


# ── temporal decay helpers ────────────────────────────────────────────────────

def _age_days(date_str: str) -> float:
    try:
        dt = datetime.fromisoformat(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds() / 86400
    except Exception:
        return 999.0

def _get_tier(age: float) -> tuple:
    if age <= 7:   return "SHORT",  1.0
    if age <= 30:  return "MEDIUM", 0.6
    return "LONG", 0.3

def _tier_label(tier: str, age: float) -> str:
    d = int(age)
    if tier == "SHORT":  return f"[SHORT-TERM {d}d ago]"
    if tier == "MEDIUM": return f"[MED-TERM {d}d ago]"
    return f"[LONG-TERM {d}d ago]"


# ── store new trade ───────────────────────────────────────────────────────────

def record_trade_decision(symbol: str, market: str, ind: dict,
                          action: str, confidence: int, price: float) -> str:
    """
    Call this immediately after every BUY/SELL decision.
    Returns doc_id — saved to pending for outcome tracking.
    """
    doc_id = store_pattern(symbol, market, ind, action, confidence,
                           outcome_pct=None)

    # Queue for outcome tracking
    pending = _load_pending()
    pending[doc_id] = {
        "symbol":      symbol,
        "market":      market,
        "action":      action,
        "entry_price": price,
        "trade_date":  datetime.now(timezone.utc).isoformat(),
        "stored_at":   datetime.now(timezone.utc).isoformat(),
        "tier":        "SHORT",
        "outcome_due": (datetime.now(timezone.utc) +
                        timedelta(days=OUTCOME_DAYS * 1.5)).isoformat(),
        "resolved":    False,
    }
    _save_pending(pending)
    return doc_id


def record_hold_state(symbol: str, market: str, ind: dict, confidence: int):
    """
    Store HOLD decisions too — important for learning when NOT to trade.
    These get outcome-tracked to show if HOLD was the right call.
    """
    store_pattern(symbol, market, ind, "HOLD", confidence, outcome_pct=0.0)


# ── outcome resolution ────────────────────────────────────────────────────────

def resolve_pending_outcomes():
    """
    Check all pending trades whose outcome window has passed.
    Fetch actual price and update ChromaDB with outcome %.
    Call this once per run — it's cheap and fast.
    """
    pending = _load_pending()
    now     = datetime.now(timezone.utc)
    updated = 0

    for doc_id, trade in list(pending.items()):
        if trade["resolved"]:
            continue

        due = datetime.fromisoformat(trade["outcome_due"])
        if now < due:
            continue   # not ready yet

        # Fetch current price
        try:
            symbol = trade["symbol"]
            market = trade["market"]
            ticker = f"{symbol}-USD" if market == "crypto" else symbol
            df = yf.download(ticker, period="10d", interval="1d",
                             progress=False, auto_adjust=True)
            if df.empty:
                continue

            current_price = float(df["Close"].squeeze().iloc[-1])
            entry_price   = float(trade["entry_price"])
            outcome_pct   = round((current_price / entry_price - 1) * 100, 3)

            update_outcome(doc_id, outcome_pct)
            pending[doc_id]["resolved"]    = True
            pending[doc_id]["outcome_pct"] = outcome_pct
            pending[doc_id]["exit_price"]  = current_price
            updated += 1

        except Exception:
            continue

    if updated:
        _save_pending(pending)

    return updated


# ── news indexing ─────────────────────────────────────────────────────────────

def index_signals(social_signals: list, insider_signals: list):
    """
    Store today's signals in the news vector store.
    Over time this builds a searchable corpus of market-moving news.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    for s in social_signals:
        for ticker in s.get("tickers", []):
            store_news(s.get("headline",""), ticker,
                       s.get("source",""), s.get("direction","NEUTRAL"), today)
    for t in insider_signals:
        if t.get("ticker"):
            store_news(
                t.get("headline", f"{t.get('person','')} {t.get('action','')} {t.get('ticker','')}"),
                t.get("ticker",""), t.get("source",""),
                t.get("signal","NEUTRAL"), today
            )


# ── retrieval for LLM ─────────────────────────────────────────────────────────

def get_rag_context(symbol: str, market: str, ind: dict) -> str:
    """Single call that returns the full decay-weighted RAG context block for one asset."""
    patterns = retrieve_similar_patterns(symbol, ind, n=10)
    if not patterns:
        patterns = retrieve_similar_patterns(symbol, ind, n=5, same_symbol_only=False)
    if not patterns:
        return format_patterns_for_llm([], symbol)

    scored = []
    for p in patterns:
        ts = p.get("stored_at") or p.get("trade_date") or p.get("date", "")
        age = _age_days(ts)
        tier, decay = _get_tier(age)
        scored.append((p, tier, age, decay, p["similarity"] * decay))

    scored.sort(key=lambda x: x[4], reverse=True)
    top = scored[:7]

    wins  = sum(1 for p, *_ in top if p["outcome_pct"] > 0)
    wr    = round(wins / len(top) * 100) if top else 0
    lines = [f"HISTORICAL ANALOGUES for {symbol} (decay-weighted, most relevant first):"]
    for p, tier, age, decay, _ in top[:5]:
        icon   = "WIN" if p["outcome_pct"] > 0 else "LOSS"
        label  = _tier_label(tier, age)
        weight = f" (weight {decay})" if tier != "SHORT" else ""
        lines.append(
            f"  {label} {p['symbol']} {p['action']} {p['similarity']:.0%}"
            f" → {p['outcome_pct']:+.1f}% {icon}{weight}"
        )
    avg = sum(p["outcome_pct"] for p, *_ in top) / len(top)
    lines.append(f"  Base rate: {wr}% wins, avg outcome {avg:+.1f}% ({len(top)} analogues)")
    return "\n".join(lines)


# ── persistence helpers ───────────────────────────────────────────────────────

def _load_pending() -> dict:
    try:
        return json.loads(PENDING_FILE.read_text()) if PENDING_FILE.exists() else {}
    except Exception:
        return {}

def _save_pending(data: dict):
    PENDING_FILE.write_text(json.dumps(data, indent=2))


# ── tier stats ────────────────────────────────────────────────────────────────

def get_tier_summary(symbol: str) -> str:
    """Brief tier win-rate string for LLM calibration context."""
    pending = _load_pending()
    buckets = {"short": [], "medium": [], "long": []}
    for trade in pending.values():
        if not trade.get("resolved") or trade.get("symbol") != symbol:
            continue
        age = _age_days(trade.get("stored_at") or trade.get("trade_date", ""))
        tier, _ = _get_tier(age)
        buckets[tier.lower()].append(int(trade.get("outcome_pct", 0) > 0))

    parts = []
    labels = [("short", "Short-term", "7d"), ("medium", "Medium", "30d"), ("long", "Long", "90d")]
    for key, name, span in labels:
        wins = buckets[key]
        if wins:
            wr = round(sum(wins) / len(wins) * 100)
            suffix = " rate" if key == "short" else ""
            parts.append(f"{name} ({span}): {len(wins)} trades, {wr}% win{suffix}")
    return " | ".join(parts) if parts else f"No trade history for {symbol}"


def get_tier_stats() -> dict:
    """Raw tier stats across all symbols: {short/medium/long: {count, win_rate}}."""
    pending = _load_pending()
    buckets = {"short": [], "medium": [], "long": []}
    for trade in pending.values():
        if not trade.get("resolved"):
            continue
        age = _age_days(trade.get("stored_at") or trade.get("trade_date", ""))
        tier, _ = _get_tier(age)
        buckets[tier.lower()].append(int(trade.get("outcome_pct", 0) > 0))

    result = {}
    for key, wins in buckets.items():
        count = len(wins)
        result[key] = {"count": count, "win_rate": round(sum(wins) / count * 100) if count else 0}
    return result


# ── stats ─────────────────────────────────────────────────────────────────────

def memory_stats() -> dict:
    from rag.vector_store import db_stats
    pending = _load_pending()
    resolved = sum(1 for t in pending.values() if t.get("resolved"))
    return {
        **db_stats(),
        "pending_outcomes": len(pending) - resolved,
        "resolved_outcomes": resolved,
    }


if __name__ == "__main__":
    stats = memory_stats()
    print("RAG Memory Stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\nResolving pending outcomes...")
    n = resolve_pending_outcomes()
    print(f"  {n} outcomes resolved")
