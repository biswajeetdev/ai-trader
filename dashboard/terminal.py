"""dashboard/terminal.py — Bloomberg-style live terminal for ai-trader.

Usage:
    python3 dashboard/terminal.py          # live (auto-refreshes)
    python3 dashboard/terminal.py --once   # render once and exit

Reads dashboard_state.json written by dashboard/state.py.
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
    from rich.columns import Columns
except ImportError:
    print("rich not installed — run: pip install rich")
    sys.exit(1)

STATE_FILE  = Path(__file__).parent.parent / "dashboard_state.json"
REFRESH_SEC = 1.5

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

console = Console(force_terminal=True, highlight=False)


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


def _bar(score: float, width: int = 4) -> str:
    filled = max(0, min(width, round(score * width)))
    return "█" * filled + "░" * (width - filled)


def _ell(s: str, n: int) -> str:
    """Truncate string with ellipsis at n chars."""
    return s if len(s) <= n else s[: n - 1] + "…"


def _price(v: float) -> str:
    """Compact price: $60.7K for large values, $187.2 for normal."""
    if v >= 10_000:
        return f"${v/1000:.1f}K"
    if v >= 1_000:
        return f"${v:.0f}"
    return f"${v:.2f}"


# ── Panel builders ────────────────────────────────────────────────────────────

def _header(s: dict) -> Table:
    """Two-row header table: branding row + market data row."""
    acc  = s.get("account", {})
    mac  = s.get("macro", {})
    stat = s.get("status", "IDLE")
    sym  = s.get("current_symbol", "—")

    eq      = acc.get("equity", 0)
    pnl_d   = acc.get("pnl_today", 0)
    pnl_pct = acc.get("pnl_pct", 0)
    vix     = mac.get("vix", "—")
    spy     = mac.get("spy_5d", "—")
    fg      = mac.get("fg_score", "—")
    bscore  = mac.get("bot_score", "—")
    regime  = mac.get("regime", "NORMAL")
    upd     = s.get("last_updated", "")

    tbl = Table(box=None, show_header=False, expand=True,
                padding=(0, 1), show_edge=False)
    tbl.add_column("a", ratio=1)
    tbl.add_column("b", ratio=2)
    tbl.add_column("c", ratio=1, justify="right")

    # Row 1: brand + equity + status
    brand = Text("  ⬛ AI-TRADER  ", style=C_HDR)
    brand.append(f" {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC ", style=C_DIM)

    eq_t = Text(f"PAPER  ${eq:>12,.2f} ", style="bold white")
    pnl_style = C_GAIN if pnl_d >= 0 else C_LOSS
    eq_t.append(f"{pnl_d:+,.2f}  ({pnl_pct:+.2f}%)", style=pnl_style)

    stat_t = Text()
    stat_t.append(f"● {stat}", style=C_RUN if stat == "RUNNING" else C_IDLE)
    if sym and sym != "—":
        stat_t.append(f"  SCANNING ", style=C_DIM)
        stat_t.append(sym, style="bold cyan")
    if upd:
        stat_t.append(f"  [{upd}]", style=C_DIM)

    # Row 2: macro data
    vix_sty = C_WARN if isinstance(vix, (int, float)) and vix > 25 else C_VAL
    spy_sty = C_GAIN if isinstance(spy, (int, float)) and spy >= 0 else C_LOSS

    mac_t = Text()
    mac_t.append("VIX ", style=C_DIM)
    mac_t.append(str(vix), style=vix_sty)
    mac_t.append("  SPY5d ", style=C_DIM)
    mac_t.append(f"{spy}%", style=spy_sty)
    mac_t.append("  F&G ", style=C_DIM)
    mac_t.append(str(fg), style=C_VAL)

    regime_t = Text()
    regime_t.append("BotScore ", style=C_DIM)
    regime_t.append(str(bscore), style=C_VAL)
    regime_t.append(f"  [{regime}]", style=C_DIM)

    tbl.add_row(brand, eq_t, stat_t)
    tbl.add_row(mac_t, regime_t, Text(""))
    return tbl


def _positions_table(s: dict) -> Table:
    tbl = Table(box=box.SIMPLE_HEAD, header_style=C_LBL, expand=True,
                show_edge=False, border_style=C_BDR, pad_edge=False)
    tbl.add_column("SYM",   style="bold white", no_wrap=True, overflow="ellipsis", width=6)
    tbl.add_column("QTY",   justify="right",    no_wrap=True, width=7)
    tbl.add_column("ENTRY", justify="right",    no_wrap=True, width=7)
    tbl.add_column("CURR",  justify="right",    no_wrap=True, width=7)
    tbl.add_column("P&L",   justify="right",    no_wrap=True, width=7)
    tbl.add_column("MKT",   style=C_DIM,        no_wrap=True, overflow="ellipsis", width=5)

    rows = s.get("positions", [])
    if not rows:
        tbl.add_row("—", "—", "—", "—", "—", "—")
    for p in rows[:12]:
        qty = p.get("qty", 0)
        qty_s = f"{qty:.3f}" if isinstance(qty, float) and qty < 1 else str(int(qty)) if isinstance(qty, float) else str(qty)
        tbl.add_row(
            _ell(p.get("symbol", ""), 6),
            _ell(qty_s, 7),
            _price(p.get('entry', 0)),
            _price(p.get('current', 0)),
            _pnl_text(p.get("pnl_pct", 0)),
            _ell(p.get("market", ""), 5),
        )
    return tbl


def _signal_feed_table(s: dict, reason_width: int = 35) -> Table:
    tbl = Table(box=box.SIMPLE_HEAD, header_style=C_LBL, expand=True,
                show_edge=False, border_style=C_BDR, pad_edge=False)
    tbl.add_column("TIME",   style=C_DIM,       no_wrap=True, width=8)
    tbl.add_column("SYM",    style="bold white", no_wrap=True, width=5)
    tbl.add_column("ACT",                        no_wrap=True, width=5)
    tbl.add_column("CF",     justify="right",    no_wrap=True, width=4)
    tbl.add_column("CONS",                       no_wrap=True, width=7)
    # REASON: ratio=1 fills remaining space, ellipsis truncates cleanly
    tbl.add_column("REASON", style=C_DIM, ratio=1, no_wrap=True, overflow="ellipsis")

    feed = list(reversed(s.get("signal_feed", [])))[:18]
    if not feed:
        tbl.add_row("—", "—", "—", "—", "—", "waiting for signals…")
    for ev in feed:
        action    = ev.get("action", "HOLD")
        consensus = ev.get("consensus", "SPLIT")
        con_sty   = {"STRONG": C_STRONG, "WEAK": C_WEAK, "SPLIT": C_SPLIT}.get(consensus, C_DIM)
        tbl.add_row(
            ev.get("time", ""),
            _ell(ev.get("symbol", ""), 5),
            Text(_ell(action, 5), style=_action_style(action)),
            f"{ev.get('conf', 0)}%",
            Text(_ell(consensus, 7), style=con_sty),
            _ell(ev.get("reason", ""), reason_width),
        )
    return tbl


def _strategy_table(s: dict) -> Table:
    tbl = Table(box=box.SIMPLE_HEAD, header_style=C_LBL, expand=True,
                show_edge=False, border_style=C_BDR, pad_edge=False)
    tbl.add_column("STRATEGY", style="bold white", ratio=1, no_wrap=True, overflow="ellipsis")
    tbl.add_column("W/L",      justify="right",    no_wrap=True, width=6)
    tbl.add_column("SCR",      justify="right",    no_wrap=True, width=4)
    tbl.add_column("EDGE",     no_wrap=True,       width=12)

    strats = s.get("strategies", {})
    if not strats:
        tbl.add_row("—", "—", "—", "(no data)")
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
        display = name.replace("_", " ").title()
        tbl.add_row(
            display,
            f"{wins}W/{losses}L",
            f"{score:.2f}",
            Text(f"{_bar(score)} {label[:6]}", style=sty),
        )
    return tbl


def _lessons_text(s: dict, width: int = 40) -> Text:
    lessons = s.get("lessons", [])
    t = Text(overflow="ellipsis")
    if not lessons:
        t.append("No post-mortems yet.\n", style=C_DIM)
        t.append("Losses >2% trigger auto-diagnosis.", style=C_DIM)
        return t
    for lesson in lessons[-6:]:
        t.append("▸ ", style="bold yellow")
        t.append(_ell(lesson, width) + "\n", style=C_DIM)
    return t


def _pipeline_row(s: dict) -> Text:
    pipe   = s.get("pipeline", {})
    stages = [("yfinance", "DATA"), ("sentiment", "SENT"),
              ("debate", "DEBATE"), ("arbiter", "ARBITER"), ("alpaca", "EXEC")]
    ICONS  = {"OK": "✓", "RUNNING": "⟳", "ERROR": "✗", "IDLE": "·"}
    STYS   = {"OK": C_GAIN, "RUNNING": "bold cyan", "ERROR": C_LOSS, "IDLE": C_DIM}
    t = Text()
    t.append(" PIPELINE ", style="bold bright_yellow")
    for i, (key, label) in enumerate(stages):
        info   = pipe.get(key, {})
        status = info.get("status", "IDLE")
        icon   = ICONS.get(status, "·")
        sty    = STYS.get(status, C_DIM)
        t.append(f"{icon}", style=sty)
        t.append(label, style="bold white" if status not in ("IDLE", "") else C_DIM)
        if i < len(stages) - 1:
            t.append("→", style=C_DIM)
    return t


def _footer(s: dict) -> Text:
    log   = s.get("trade_log", [])
    radar = s.get("radar", {})
    wr    = s.get("win_rate", {})

    t = Text()
    # Line 1: trades + radar + win rate
    t.append(" TRADES ", style="bold bright_yellow")
    if not log:
        t.append("none yet  ", style=C_DIM)
    for entry in list(reversed(log))[:5]:
        action = entry.get("action", "?")
        t.append(f"[{entry.get('time','')}] ", style=C_DIM)
        t.append(f"{action} ", style=_action_style(action))
        t.append(f"{_ell(entry.get('symbol',''), 5)} ", style="bold white")
        t.append(f"{entry.get('qty','')}@${entry.get('price',0):.2f}  ", style=C_VAL)

    score   = radar.get("score", "—")
    level   = radar.get("level", "—")
    lvl_sty = {"SAFE": C_SAFE, "DANGER": C_DANGER, "WARNING": C_WARN}.get(str(level), C_DIM)
    t.append(" ║ RADAR ", style="bold bright_yellow")
    t.append(f"{score}/100 ", style=C_VAL)
    t.append(str(level), style=lvl_sty)
    t.append("  ║ WIN ", style="bold bright_yellow")
    t.append(f"{wr.get('rate','—')}% ({wr.get('trades','—')} tr)", style=C_VAL)

    # Line 2: pipeline
    t.append("\n")
    t.append_text(_pipeline_row(s))
    return t


# ── Layout assembly ───────────────────────────────────────────────────────────

def _build(s: dict) -> Layout:
    w = console.width or 180

    root = Layout(name="root")
    root.split_column(
        Layout(name="hdr",    size=3),   # 2 content rows + 1 padding
        Layout(name="body",   ratio=1),
        Layout(name="footer", size=5),   # border + 2 content + border + spare
    )

    if w >= 160:
        # Wide: 3-column layout
        root["body"].split_row(
            Layout(name="left",   ratio=25),
            Layout(name="center", ratio=45),
            Layout(name="right",  ratio=30),
        )
        root["right"].split_column(
            Layout(name="strategies", ratio=55),
            Layout(name="lessons",    ratio=45),
        )
        right_wide = True
    else:
        # Narrow: 2-column, stack right panels below center
        root["body"].split_row(
            Layout(name="left",   ratio=30),
            Layout(name="center", ratio=70),
        )
        right_wide = False

    reason_w = max(20, (w * 44 // 100) - 45)

    root["hdr"].update(_header(s))

    root["left"].update(
        Panel(_positions_table(s),
              title="[bold cyan]POSITIONS[/bold cyan]",
              border_style=C_BDR, expand=True)
    )

    if right_wide:
        root["center"].update(
            Panel(_signal_feed_table(s, reason_w),
                  title="[bold cyan]LIVE SIGNAL FEED[/bold cyan]",
                  border_style=C_BDR, expand=True)
        )
        root["strategies"].update(
            Panel(_strategy_table(s),
                  title="[bold cyan]STRATEGY SCORES[/bold cyan]",
                  border_style=C_BDR, expand=True)
        )
        root["lessons"].update(
            Panel(_lessons_text(s, width=w * 30 // 100 - 6),
                  title="[bold cyan]LEARNED LESSONS[/bold cyan]",
                  border_style=C_BDR, expand=True)
        )
    else:
        # Narrow: center gets signal feed only
        root["center"].update(
            Panel(_signal_feed_table(s, max(20, w * 60 // 100 - 45)),
                  title="[bold cyan]LIVE SIGNAL FEED[/bold cyan]",
                  border_style=C_BDR, expand=True)
        )

    root["footer"].update(
        Panel(_footer(s),
              border_style="dim bright_yellow",
              title="[bold cyan]TRADE LOG  ║  PIPELINE[/bold cyan]",
              expand=True)
    )
    return root


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    once = "--once" in sys.argv
    if once:
        console.print(_build(_state()))
        return

    console.clear()
    with Live(console=console,
              refresh_per_second=int(1 / REFRESH_SEC) + 1,
              screen=True) as live:
        while True:
            live.update(_build(_state()))
            time.sleep(REFRESH_SEC)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[dim]Dashboard closed.[/dim]")
