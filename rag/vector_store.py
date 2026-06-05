"""
vector_store.py — ChromaDB wrapper for the trading RAG pipeline.

Persists to ~/ai-trader/rag/db/ — survives restarts, no server needed.

Collections:
  trade_patterns  — historical market states + outcomes (numeric similarity)
  news_signals    — social/insider news items (text similarity)
  knowledge_base  — earnings transcripts, analyst notes (text similarity)
"""

import json
import numpy as np
import chromadb
from chromadb.config import Settings
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent / "db"
DB_PATH.mkdir(exist_ok=True)

_client = None

def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=str(DB_PATH),
            settings=Settings(anonymized_telemetry=False)
        )
    return _client


def _col(name, metadata=None):
    return _get_client().get_or_create_collection(
        name=name,
        metadata=metadata or {"hnsw:space": "cosine"}
    )


# ── numeric embedding for indicator vectors ───────────────────────────────────

def _indicator_embedding(ind: dict) -> list:
    """
    Convert market indicators to a normalized 12-dim vector.
    Consistent normalization makes cosine similarity meaningful.
    """
    rsi       = float(ind.get("rsi14", 50)) / 100.0        # 0–1
    bb        = float(ind.get("bb_position", 0.5))         # 0–1
    macd_norm = float(ind.get("macd_hist", 0))
    macd_norm = max(-1, min(1, macd_norm / (abs(macd_norm) + 1e-6) *
                   min(1, abs(macd_norm) / 5)))            # -1 to 1, compressed
    pct_1d    = max(-0.1, min(0.1, float(ind.get("pct_1d", 0)) / 100))
    pct_5d    = max(-0.2, min(0.2, float(ind.get("pct_5d", 0)) / 100))
    above20   = 1.0 if ind.get("above_sma20") else 0.0
    above50   = 1.0 if ind.get("above_sma50") else 0.0

    # ATR relative to price (volatility normalised)
    price     = float(ind.get("price", 1))
    atr       = float(ind.get("atr14", price * 0.02))
    atr_norm  = min(1.0, atr / (price + 1e-6))

    return [
        rsi, bb, macd_norm, pct_1d, pct_5d,
        above20, above50, atr_norm,
        0.0, 0.0, 0.0, 0.0   # padding to 12 dims
    ]


# ── trade pattern memory ──────────────────────────────────────────────────────

def store_pattern(symbol: str, market: str, ind: dict, action: str,
                  confidence: int, outcome_pct: float = None,
                  outcome_days: int = 5):
    """
    Store a market state → decision → outcome record.
    outcome_pct: price change % in outcome_days after the trade.
    Call this again with outcome_pct once outcome is known.
    """
    col = _col("trade_patterns")
    ts  = datetime.utcnow().isoformat()
    doc_id = f"{symbol}_{ts}".replace(":", "-")

    meta = {
        "symbol":       symbol,
        "market":       market,
        "action":       action,
        "confidence":   confidence,
        "rsi":          round(float(ind.get("rsi14", 50)), 2),
        "bb":           round(float(ind.get("bb_position", 0.5)), 3),
        "macd_hist":    round(float(ind.get("macd_hist", 0)), 4),
        "above_sma50":  str(ind.get("above_sma50", False)),
        "price":        round(float(ind.get("price", 0)), 2),
        "date":         ts[:10],
        "outcome_pct":  round(outcome_pct, 3) if outcome_pct is not None else 0.0,
        "outcome_days": outcome_days,
        "has_outcome":  str(outcome_pct is not None),
    }

    doc = (f"{symbol} RSI={meta['rsi']} BB={meta['bb']} "
           f"MACD={'bull' if meta['macd_hist']>0 else 'bear'} "
           f"above50={meta['above_sma50']} action={action}")

    emb = _indicator_embedding(ind)

    col.upsert(ids=[doc_id], documents=[doc],
               embeddings=[emb], metadatas=[meta])
    return doc_id


def update_outcome(doc_id: str, outcome_pct: float):
    """Update a stored pattern with its actual price outcome."""
    col = _col("trade_patterns")
    try:
        existing = col.get(ids=[doc_id])
        if existing["metadatas"]:
            meta = existing["metadatas"][0]
            meta["outcome_pct"]  = round(outcome_pct, 3)
            meta["has_outcome"]  = "True"
            col.update(ids=[doc_id], metadatas=[meta])
    except Exception:
        pass


def retrieve_similar_patterns(symbol: str, ind: dict, n: int = 5,
                               same_symbol_only: bool = False) -> list:
    """
    Find the N most similar historical market states for this asset.
    Returns patterns with outcomes — the LLM uses these as analogues.
    """
    col  = _col("trade_patterns")
    emb  = _indicator_embedding(ind)

    where = {"symbol": symbol} if same_symbol_only else None

    try:
        results = col.query(
            query_embeddings=[emb],
            n_results=min(n, col.count()),
            where=where,
            include=["documents", "metadatas", "distances"]
        )
    except Exception:
        return []

    out = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        dist = results["distances"][0][i]
        sim  = round(1 - dist, 3)   # cosine: 1=identical, 0=orthogonal
        if meta.get("has_outcome") == "True":
            out.append({
                "symbol":      meta["symbol"],
                "date":        meta["date"],
                "action":      meta["action"],
                "rsi":         meta["rsi"],
                "outcome_pct": meta["outcome_pct"],
                "similarity":  sim,
            })
    return out


def format_patterns_for_llm(patterns: list, symbol: str) -> str:
    """Format retrieved historical patterns as LLM context."""
    if not patterns:
        return f"No historical patterns for {symbol} yet."
    lines = [f"HISTORICAL ANALOGUES for {symbol} (most similar past setups):"]
    wins  = [p for p in patterns if p["outcome_pct"] > 0]
    wr    = round(len(wins)/len(patterns)*100) if patterns else 0
    for p in patterns[:5]:
        icon = "✓" if p["outcome_pct"] > 0 else "✗"
        lines.append(
            f"  {icon} {p['date']}: {p['action']} at RSI {p['rsi']} "
            f"→ {p['outcome_pct']:+.1f}% in 5d (similarity {p['similarity']:.2f})"
        )
    avg = sum(p["outcome_pct"] for p in patterns) / len(patterns)
    lines.append(f"  Base rate: {wr}% wins, avg outcome {avg:+.1f}% ({len(patterns)} analogues)")
    return "\n".join(lines)


# ── news / signal corpus ──────────────────────────────────────────────────────

def store_news(headline: str, ticker: str, source: str,
               direction: str, date: str):
    """Store a news/signal item for later retrieval."""
    col    = _col("news_signals")
    doc_id = f"{ticker}_{date}_{hash(headline) % 100000}"
    col.upsert(
        ids=[doc_id],
        documents=[f"{ticker} {direction} {headline}"],
        metadatas=[{"ticker": ticker, "source": source,
                    "direction": direction, "date": date}]
    )


def retrieve_news(query: str, ticker: str = None, n: int = 3) -> list:
    """Retrieve most relevant news for a query/ticker."""
    col = _col("news_signals")
    if col.count() == 0:
        return []
    where = {"ticker": ticker} if ticker else None
    try:
        r = col.query(query_texts=[query], n_results=min(n, col.count()),
                      where=where, include=["documents","metadatas"])
        return [{"text": r["documents"][0][i], **r["metadatas"][0][i]}
                for i in range(len(r["ids"][0]))]
    except Exception:
        return []


# ── knowledge base (earnings, analyst notes) ─────────────────────────────────

def store_knowledge(text: str, source: str, ticker: str, category: str, date: str):
    """Store a knowledge item (earnings transcript chunk, analyst note)."""
    col    = _col("knowledge_base")
    doc_id = f"{ticker}_{category}_{date}_{hash(text) % 100000}"
    col.upsert(
        ids=[doc_id],
        documents=[text],
        metadatas=[{"ticker": ticker, "source": source,
                    "category": category, "date": date}]
    )


def retrieve_knowledge(query: str, ticker: str = None, n: int = 3) -> list:
    """Retrieve relevant knowledge chunks."""
    col = _col("knowledge_base")
    if col.count() == 0:
        return []
    where = {"ticker": ticker} if ticker else None
    try:
        r = col.query(query_texts=[query], n_results=min(n, col.count()),
                      where=where, include=["documents","metadatas"])
        return [{"text": r["documents"][0][i], **r["metadatas"][0][i]}
                for i in range(len(r["ids"][0]))]
    except Exception:
        return []


# ── stats ─────────────────────────────────────────────────────────────────────

def db_stats():
    c = _get_client()
    cols = c.list_collections()
    stats = {}
    for col in cols:
        stats[col.name] = col.count()
    return stats


if __name__ == "__main__":
    print("ChromaDB RAG vector store")
    print(f"DB path: {DB_PATH}")
    print(f"Collections: {db_stats()}")
