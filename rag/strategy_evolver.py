"""strategy_evolver.py — Strategy-level self-evolution.

Two feedback loops:
1. Strategy bandit: Thompson sampling per strategy type → which strategies win?
2. Post-mortem: when a trade loses >2%, call LLM to diagnose WHY, store insight to RAG

Strategy types: LONG_MOMENTUM, SHORT_REVERSION, WHEEL_PUT, WHEEL_CALL, PAIRS_LONG,
PAIRS_SHORT, DIV_CAPTURE, MERGER_ARB, LEVERAGED_ETF_BULL, LEVERAGED_ETF_BEAR
"""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from openai import OpenAI

CALIB_FILE    = Path(__file__).parent / "strategy_calibration.json"
TRADE_HISTORY = Path(__file__).parent.parent / "trade_history.json"
STRATEGY_TYPES = [
    "LONG_MOMENTUM", "SHORT_REVERSION", "WHEEL_PUT", "WHEEL_CALL",
    "PAIRS_LONG", "PAIRS_SHORT", "DIV_CAPTURE", "MERGER_ARB",
    "LEVERAGED_ETF_BULL", "LEVERAGED_ETF_BEAR",
]
LLM_MODELS_DEFAULT = ["gpt-4o-mini", "gpt-4o", "o1-mini"]


def _default_entry() -> dict:
    return {"alpha": 1, "beta": 1, "wins": 0, "losses": 0, "total_pnl": 0.0}


def _load() -> dict:
    try:
        if CALIB_FILE.exists():
            data = json.loads(CALIB_FILE.read_text())
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {"strategies": {s: _default_entry() for s in STRATEGY_TYPES},
            "llm_bandit": {}, "post_mortems": [], "seen_trades": [], "last_updated": None}


def _save(data: dict) -> None:
    data["last_updated"] = datetime.now(timezone.utc).isoformat()
    CALIB_FILE.write_text(json.dumps(data, indent=2))


def _get_client():
    try:
        token = subprocess.check_output(["gh", "auth", "token"], text=True).strip()
        return OpenAI(base_url="https://models.inference.ai.azure.com", api_key=token)
    except Exception:
        return None


def get_thompson_score(strategy: str) -> float:
    # Expected value of Beta(alpha, beta) — higher = more historical wins
    s = _load().get("strategies", {}).get(strategy, _default_entry())
    return s["alpha"] / (s["alpha"] + s["beta"])


def _run_post_mortem(symbol: str, strategy: str, pnl_pct: float,
                     entry_context: dict, client) -> str:
    if client is None:
        return "Post-mortem unavailable"
    try:
        prompt = (
            f"You are a trading post-mortem analyst. "
            f"A {strategy} trade on {symbol} lost {pnl_pct:.1f}%.\n\n"
            f"Entry context:\n{json.dumps(entry_context, indent=2)}\n\n"
            "In 2 sentences: (1) What signal or condition was most likely wrong at entry? "
            "(2) What rule should the bot follow to avoid this in future?"
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120, temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return "Post-mortem unavailable"


def record_strategy_outcome(symbol: str, strategy: str, pnl_pct: float,
                             entry_context: dict = None) -> None:
    # Update bandit and trigger post-mortem on significant losses
    data = _load()
    s = data.setdefault("strategies", {}).setdefault(strategy, _default_entry())
    if pnl_pct > 0:
        s["alpha"] += 1; s["wins"] += 1
    else:
        s["beta"] += 1; s["losses"] += 1
    s["total_pnl"] = round(s.get("total_pnl", 0.0) + pnl_pct, 4)

    if pnl_pct < -2.0 and entry_context is not None:
        diagnosis = _run_post_mortem(symbol, strategy, pnl_pct, entry_context, _get_client())
        lesson    = diagnosis.split("(2)")[-1].strip() if "(2)" in diagnosis else diagnosis
        pms = data.setdefault("post_mortems", [])
        pms.append({"date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    "symbol": symbol, "strategy": strategy,
                    "pnl_pct": round(pnl_pct, 2), "diagnosis": diagnosis, "lesson": lesson})
        data["post_mortems"] = pms[-50:]

    _save(data)


def get_strategy_allocation_prompt() -> str:
    # ARBITER-style calibration block injected into LLM system prompts
    try:
        data       = _load()
        strategies = data.get("strategies", {})
        total      = sum(s.get("wins", 0) + s.get("losses", 0) for s in strategies.values())
        if total < 5:
            return ""
        rows = sorted(
            [(name, s.get("wins", 0), s.get("losses", 0),
              s["alpha"] / (s["alpha"] + s["beta"]))
             for name, s in strategies.items()],
            key=lambda x: -x[3]
        )
        lines = ["STRATEGY PERFORMANCE (Thompson sampling — use to weight your decision):"]
        for name, wins, losses, score in rows:
            label = ("TRUST (winning)" if score >= 0.60
                     else "REDUCE (losing)" if score < 0.40 else "NEUTRAL")
            lines.append(f"  {name}: {wins}W/{losses}L, score={score:.2f} → {label}")
        return "\n".join(lines)
    except Exception:
        return ""


def bootstrap_from_trade_history() -> int:
    # Seed bandit from trade_history.json once at startup; dedup via seen_trades
    if not TRADE_HISTORY.exists():
        return 0
    try:
        history = json.loads(TRADE_HISTORY.read_text())
    except Exception:
        return 0

    data = _load()
    seen = set(data.get("seen_trades", []))
    count = 0
    for trade in history:
        if not isinstance(trade, dict):
            continue
        symbol  = trade.get("symbol", "")
        pnl_pct = float(trade.get("pnl_pct", 0) or 0)
        date    = str(trade.get("exit_date", trade.get("date", "")))[:10]
        key     = f"{symbol}_{pnl_pct:.3f}_{date}_hist"
        if key in seen:
            continue
        strategy = trade.get("strategy", "LONG_MOMENTUM")
        s = data.setdefault("strategies", {}).setdefault(strategy, _default_entry())
        if pnl_pct > 0:
            s["alpha"] += 1; s["wins"] += 1
        else:
            s["beta"] += 1; s["losses"] += 1
        s["total_pnl"] = round(s.get("total_pnl", 0.0) + pnl_pct, 4)
        seen.add(key)
        count += 1

    data["seen_trades"] = list(seen)[-200:]
    if count:
        _save(data)
    return count


def get_recent_lessons(n: int = 5) -> str:
    # Last N post-mortem lessons formatted for LLM context injection
    try:
        pms = _load().get("post_mortems", [])
        if not pms:
            return ""
        return "\n".join(
            f"Lesson: [{pm['date']}] {pm['symbol']} — {pm.get('lesson', pm.get('diagnosis', ''))}"
            for pm in pms[-n:]
        )
    except Exception:
        return ""


def record_llm_outcome(model: str, won: bool) -> None:
    """Track which LLM model generates winning signals via Thompson sampling."""
    data = _load()
    bandit = data.setdefault("llm_bandit", {})
    entry  = bandit.setdefault(model, {"alpha": 1, "beta": 1, "wins": 0, "losses": 0})
    if won:
        entry["alpha"] += 1; entry["wins"] += 1
    else:
        entry["beta"]  += 1; entry["losses"] += 1
    _save(data)


def get_best_llm(candidates: list) -> str:
    """Return the candidate model with best Thompson sampling score.
    Falls back to first candidate if no data yet."""
    if not candidates:
        return "gpt-4o-mini"
    try:
        bandit = _load().get("llm_bandit", {})
        def _score(m):
            e = bandit.get(m, {"alpha": 1, "beta": 1})
            return e["alpha"] / (e["alpha"] + e["beta"])
        best = max(candidates, key=_score)
        scores = {m: f"{_score(m):.2f}" for m in candidates}
        return best
    except Exception:
        return candidates[0]
