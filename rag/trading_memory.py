"""rag/trading_memory.py — Cumulative decision memory log.

Every trade decision is appended to trading_memory.md.
The ARBITER reads recent entries at the start of each session so
mistakes don't repeat and winning patterns compound.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

MEMORY_FILE = Path(__file__).parent.parent / "trading_memory.md"
_MAX_ENTRIES = 200


def append_decision(symbol: str, action: str, confidence: int,
                    reason: str, outcome: str = "PENDING",
                    pnl_pct: float = 0.0, model: str = "") -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    line = (f"| {ts} | {symbol} | {action} | {confidence}% | "
            f"{outcome} | {pnl_pct:+.2f}% | {reason[:80]} | {model} |\n")
    if not MEMORY_FILE.exists():
        header = (
            "# AI-Trader Decision Memory\n\n"
            "| Date (UTC) | Symbol | Action | Conf | Outcome | P&L% | Reason | Model |\n"
            "|---|---|---|---|---|---|---|---|\n"
        )
        MEMORY_FILE.write_text(header + line)
        return
    lines = MEMORY_FILE.read_text().splitlines(keepends=True)
    # Keep header (first 3 lines) + last _MAX_ENTRIES data lines
    header = lines[:3]
    data   = [l for l in lines[3:] if l.strip()]
    data.append(line)
    data   = data[-_MAX_ENTRIES:]
    MEMORY_FILE.write_text("".join(header + data))


def update_outcome(symbol: str, pnl_pct: float) -> None:
    """Back-fill PENDING entries for a symbol with realized P&L."""
    if not MEMORY_FILE.exists():
        return
    text   = MEMORY_FILE.read_text()
    lines  = text.splitlines(keepends=True)
    result = []
    for line in lines:
        if f"| {symbol} |" in line and "| PENDING |" in line:
            outcome = "WIN" if pnl_pct > 0 else "LOSS"
            line    = line.replace("| PENDING | +0.00%", f"| {outcome} | {pnl_pct:+.2f}%")
        result.append(line)
    MEMORY_FILE.write_text("".join(result))


def get_memory_context(symbol: str = "", n: int = 8) -> str:
    """Return last N decisions (filtered by symbol if given) for ARBITER injection."""
    if not MEMORY_FILE.exists():
        return ""
    try:
        lines = [l for l in MEMORY_FILE.read_text().splitlines()
                 if l.startswith("|") and not l.startswith("| Date") and "---" not in l]
        if symbol:
            sym_lines = [l for l in lines if f"| {symbol} |" in l]
            lines     = (sym_lines[-4:] + [l for l in lines if f"| {symbol} |" not in l][-4:])
        else:
            lines = lines[-n:]
        if not lines:
            return ""
        return ("DECISION MEMORY (past calls — learn from outcomes, don't repeat mistakes):\n" +
                "\n".join(lines[-n:]))
    except Exception:
        return ""
