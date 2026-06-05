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
    """
    Single call that returns the full RAG context block for one asset.
    Plugs directly into the LLM prompt.
    """
    patterns = retrieve_similar_patterns(symbol, ind, n=7)
    if not patterns:
        # Fall back to cross-symbol patterns (all assets)
        patterns = retrieve_similar_patterns(symbol, ind, n=5,
                                             same_symbol_only=False)
    return format_patterns_for_llm(patterns, symbol)


# ── persistence helpers ───────────────────────────────────────────────────────

def _load_pending() -> dict:
    try:
        return json.loads(PENDING_FILE.read_text()) if PENDING_FILE.exists() else {}
    except Exception:
        return {}

def _save_pending(data: dict):
    PENDING_FILE.write_text(json.dumps(data, indent=2))


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
