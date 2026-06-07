"""dashboard/terminal.py — Bloomberg-style live terminal for ai-trader.

Usage:
    python3 dashboard/terminal.py          # watch live (auto-refreshes)
    python3 dashboard/terminal.py --once   # render once and exit

Reads dashboard_state.json written by dashboard/state.py.
Requires: pip install rich
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from rich import box
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ImportError:
    print("rich not installed — run: pip install rich")
    sys.exit(1)

STATE_FILE  = Path(__file__).parent.parent / "dashboard_state.json"
REFRESH_SEC = 1.5

# Explicit style strings (theme names don't work in style= kwargs)
C_HDR    = "bold black on bright_yellow"
C_LBL    = "bold cyan"
C_VAL    = "white"
C_GAIN   = "bold green"
C_LOSS   = "bold red"
C_DIM    = "dim white"
C_BUY    = "bold bright_green"
C_SELL   = "bold red"
C_HOLD   = "dim yellow"
C_STRONG = "bold green"
C_WEAK   = "yellow"
C_SPLIT  = "dim red"
C_RUN    = "bold bright_green"
C_IDLE   = "dim yellow"
C_BDR    = "bright_yellow"
C_SAFE   = "bold green"
C_DANGER = "bold red"
C_WARN   = "bold yellow"

console = Console(force_terminal=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _state() -> dict:
    try:
        if STATE_FILE.exists():
            return json.loads(STATE_FILE.read_text())
    except Exception:
        pass
    return {}


def _pnl_text(val: float) -> Text:
    return Text(f"{val:+.2f}%", style=C_GAIN if val >= 0 else C_LOSS)


def _action_style(action: str) -> str:
    return {"BUY": C_BUY, "COVER": C_BUY, "SELL": C_SELL, "SHORT": C_SELL}.get(action, C_HOLD)


def _bar(score: float, width: int = 5) -> str:
    filled = max(0, min(width, round(score * width)))
    return "█" * filled + "░" * (width - filled)


# ── Panel builders ────────────────────────────────────────────────────────────

def _header(s: dict) -> Text:
    acc  = s.get("account", {})
    mac  = s.get("macro", {})
    stat = s.get("status", "IDLE")
    sym  = s.get("current_symbol", "—")
    upd  = s.get("last_updated", "")

    eq      = acc.get("equity", 0)
    pnl_d   = acc.get("pnl_today", 0)
    pnl_pct = acc.get("pnl_pct", 0)
    vix     = mac.get("vix", "—")
    spy     = mac.get("spy_5d", "—")
    fg      = mac.get("fg_score", "—")
    bscore  = mac.get("bot_score", "—")
    regime  = mac.get("regime", "")

    t = Text()
    t.append("  AI-TRADER  ", style=C_HDR)
    t.append(f"  {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC", style=C_DIM)
    t.append("  ║  ")
    t.append(f"PAPER  ${eq:>12,.2f}", style="bold white")
    t.append("  ")
    t.append(f"{pnl_d:+,.2f}  ({pnl_pct:+.2f}%)",
             style=C_GAIN if pnl_d >= 0 else C_LOSS)
    t.append("  ║  VIX ", style=C_DIM)
    t.append(str(vix), style=C_WARN if isinstance(vix, (int, float)) and vix > 25 else C_VAL)
    t.append("  SPY 5d ", style=C_DIM)
    t.append(f"{spy}%",
             style=C_GAIN if isinstance(spy, (int, float)) and spy >= 0 else C_LOSS)
    t.append("  F&G ", style=C_DIM)
    t.append(str(fg), style=C_VAL)
    t.append("  BotScore ", style=C_DIM)
    t.append(str(bscore), style=C_VAL)
    if regime:
        t.append(f"  [{regime}]", style=C_DIM)
    t.append("  ║  SCANNING ", style=C_DIM)
    t.append(sym or "—", style="bold cyan")
    t.append("  ║  ")
    t.append(f"● {stat}", style=C_RUN if stat == "RUNNING" else C_IDLE)
    if upd:
        t.append(f"  upd {upd}", style=C_DIM)
    return t


def _positions_table(s: dict) -> Table:
    tbl = Table(box=box.SIMPLE_HEAD, header_style=C_LBL, expand=True,
                show_edge=False, border_style=C_BDR, pad_edge=False)
    tbl.add_column("SYMBOL",  style="bold white", width=7)
    tbl.add_column("QTY",     justify="right",    width=8)
    tbl.add_column("ENTRY",   justify="right",    width=8)
    tbl.add_column("CURR",    justify="right",    width=8)
    tbl.add_column("P&L",     justify="right",    width=8)
    tbl.add_column("MKT",     style=C_DIM,        width=8)

    rows = s.get("positions", [])
    if not rows:
        tbl.add_row("—", "—", "—", "—", "—", "—")
    for p in rows[:14]:
        tbl.add_row(
            p.get("symbol", ""),
            str(p.get("qty", "")),
            f"${p.get('entry', 0):.2f}",
            f"${p.get('current', 0):.2f}",
            _pnl_text(p.get("pnl_pct", 0)),
            p.get("market", ""),
        )
    return tbl


def _signal_feed_table(s: dict) -> Table:
    tbl = Table(box=box.SIMPLE, header_style=C_LBL, expand=True,
                show_edge=False, border_style=C_BDR, pad_edge=False)
    tbl.add_column("TIME",      style=C_DIM,       width=9)
    tbl.add_column("SYMBOL",    style="bold white", width=6)
    tbl.add_column("ACTION",                       width=7)
    tbl.add_column("CONF",      justify="right",   width=5)
    tbl.add_column("CONSENSUS",                    width=9)
    tbl.add_column("REASON",    style=C_DIM,       min_width=20, no_wrap=True)

    feed = list(reversed(s.get("signal_feed", [])))[:16]
    if not feed:
        tbl.add_row("—", "—", "—", "—", "—", "waiting for signals…")
    for ev in feed:
        action    = ev.get("action", "HOLD")
        consensus = ev.get("consensus", "SPLIT")
        con_sty   = {"STRONG": C_STRONG, "WEAK": C_WEAK, "SPLIT": C_SPLIT}.get(consensus, C_DIM)
        tbl.add_row(
            ev.get("time", ""),
            ev.get("symbol", ""),
            Text(action, style=_action_style(action)),
            f"{ev.get('conf', 0)}%",
            Text(consensus, style=con_sty),
            ev.get("reason", "")[:80],
        )
    return tbl


def _strategy_table(s: dict) -> Table:
    tbl = Table(box=box.SIMPLE, header_style=C_LBL, expand=True,
                show_edge=False, border_style=C_BDR, pad_edge=False)
    tbl.add_column("STRATEGY", style="bold white", min_width=15)
    tbl.add_column("W/L",      justify="right",    width=7)
    tbl.add_column("SCORE",    justify="right",    width=6)
    tbl.add_column("EDGE",                        width=14)

    strats = s.get("strategies", {})
    if not strats:
        tbl.add_row("—", "—", "—", "(no data yet)")
        return tbl

    for name, v in sorted(strats.items(), key=lambda x: -x[1].get("score", 0)):
        score  = v.get("score", 0.5)
        wins   = v.get("wins", 0)
        losses = v.get("losses", 0)
        if score >= 0.60:
            label, sty = "TRUST",   C_GAIN
        elif score < 0.40:
            label, sty = "REDUCE",  C_LOSS
        else:
            label, sty = "NEUTRAL", C_WARN
        tbl.add_row(
            name.replace("_", " ")[:15],
            f"{wins}W/{losses}L",
            f"{score:.2f}",
            Text(f"{_bar(score)} {label}", style=sty),
        )
    return tbl


def _lessons_text(s: dict) -> Text:
    lessons = s.get("lessons", [])
    t = Text()
    if not lessons:
        t.append("No post-mortems yet.\n", style=C_DIM)
        t.append("Losses >2% trigger LLM diagnosis.", style=C_DIM)
        return t
    for l in lessons[-6:]:
        t.append("▶ ", style="bold yellow")
        t.append(l + "\n", style=C_DIM)
    return t


def _pipeline_panel(s: dict) -> Text:
    pipe   = s.get("pipeline", {})
    stages = ["yfinance", "debate", "arbiter", "alpaca"]
    ICONS  = {"OK": "✓", "RUNNING": "⟳", "ERROR": "✗", "IDLE": "·"}
    STYS   = {"OK": C_GAIN, "RUNNING": "bold cyan", "ERROR": C_LOSS, "IDLE": C_DIM}
    t = Text()
    for i, stage in enumerate(stages):
        info   = pipe.get(stage, {})
        status = info.get("status", "IDLE")
        icon   = ICONS.get(status, "·")
        sty    = STYS.get(status, C_DIM)
        ms     = info.get("latency_ms", 0)
        t.append(f" {icon} ", style=sty)
        t.append(stage, style="bold white" if status != "IDLE" else C_DIM)
        if ms:
            t.append(f"({ms}ms)", style=C_DIM)
        if i < len(stages) - 1:
            t.append(" → ", style=C_DIM)
    return t


def _footer(s: dict) -> Text:
    log   = s.get("trade_log", [])
    radar = s.get("radar", {})
    wr    = s.get("win_rate", {})

    t = Text()
    t.append(" TRADES ", style="bold bright_yellow")
    if not log:
        t.append("none yet   ", style=C_DIM)
    for entry in list(reversed(log))[:6]:
        action = entry.get("action", "?")
        t.append(f"[{entry.get('time','')}] ", style=C_DIM)
        t.append(f"{action} ", style=_action_style(action))
        t.append(f"{entry.get('symbol','')} ", style="bold white")
        t.append(f"{entry.get('qty','')}@${entry.get('price',0):.2f}  ", style=C_VAL)

    score = radar.get("score", "—")
    level = radar.get("level", "—")
    lvl_sty = {"SAFE": C_SAFE, "DANGER": C_DANGER, "WARNING": C_WARN}.get(str(level), C_DIM)
    t.append("║ RADAR ", style="bold bright_yellow")
    t.append(f"{score}/100  ", style=C_VAL)
    t.append(str(level), style=lvl_sty)

    t.append("  ║ WIN RATE ", style="bold bright_yellow")
    t.append(f"{wr.get('rate','—')}%  ({wr.get('trades','—')} trades)", style=C_VAL)
    t.append("\n")
    t.append(" PIPELINE  ", style="bold bright_yellow")
    t.append_text(_pipeline_panel(s))
    return t


# ── Layout assembly ───────────────────────────────────────────────────────────

def _build(s: dict) -> Layout:
    root = Layout(name="root")
    root.split_column(
        Layout(name="hdr",    size=1),
        Layout(name="body",   ratio=1),
        Layout(name="footer", size=3),
    )
    root["body"].split_row(
        Layout(name="left",   ratio=28),
        Layout(name="center", ratio=44),
        Layout(name="right",  ratio=28),
    )
    root["right"].split_column(
        Layout(name="strategies", ratio=55),
        Layout(name="lessons",    ratio=45),
    )

    root["hdr"].update(_header(s))
    root["left"].update(
        Panel(_positions_table(s), title="[bold cyan]POSITIONS[/bold cyan]",
              border_style=C_BDR, expand=True)
    )
    root["center"].update(
        Panel(_signal_feed_table(s), title="[bold cyan]LIVE SIGNAL FEED[/bold cyan]",
              border_style=C_BDR, expand=True)
    )
    root["strategies"].update(
        Panel(_strategy_table(s), title="[bold cyan]STRATEGY SCORES[/bold cyan]",
              border_style=C_BDR, expand=True)
    )
    root["lessons"].update(
        Panel(_lessons_text(s), title="[bold cyan]LEARNED LESSONS[/bold cyan]",
              border_style=C_BDR, expand=True)
    )
    root["footer"].update(
        Panel(_footer(s), border_style="dim bright_yellow",
              title="[bold cyan]TRADE LOG[/bold cyan]", expand=True)
    )
    return root


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    once = "--once" in sys.argv
    if once:
        console.print(_build(_state()))
        return

    console.clear()
    with Live(console=console, refresh_per_second=int(1 / REFRESH_SEC) + 1,
              screen=True) as live:
        while True:
            live.update(_build(_state()))
            time.sleep(REFRESH_SEC)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[dim]Dashboard closed.[/dim]")
