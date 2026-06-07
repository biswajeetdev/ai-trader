"""dashboard/terminal.py — Bloomberg-style live terminal for ai-trader.

Usage:
    python3 dashboard/terminal.py          # live (auto-refreshes)
    python3 dashboard/terminal.py --once   # render once and exit
"""

import json
import sys
import time
from datetime import datetime, timezone
from itertools import cycle
from pathlib import Path

try:
    from rich import box
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.rule import Rule
except ImportError:
    print("rich not installed — run: pip install rich")
    sys.exit(1)

STATE_FILE  = Path(__file__).parent.parent / "dashboard_state.json"
REFRESH_SEC = 1.5

# ── Cobalt-inspired Bloomberg palette ─────────────────────────────────────────
C_HDR    = "bold black on color(220)"      # amber header bar
C_ACCENT = "color(75)"                      # cobalt blue
C_VAL    = "bright_white"
C_GAIN   = "color(82)"                      # vivid green
C_LOSS   = "color(196)"                     # vivid red
C_DIM    = "color(244)"                     # mid-grey
C_BUY    = "bold color(82)"
C_SELL   = "bold color(196)"
C_HOLD   = "color(136)"                     # muted amber
C_STRONG = "bold color(82)"
C_WEAK   = "color(136)"
C_SPLIT  = "color(244)"
C_RUN    = "bold color(82)"
C_IDLE   = "color(244)"
C_BDR    = "color(220)"                     # amber border
C_BDR2   = "color(240)"                     # dim border for footer
C_SAFE   = "bold color(82)"
C_DANGER = "bold color(196)"
C_WARN   = "bold color(220)"
C_SPARK  = "color(75)"                      # sparkline color
C_TAB    = "bold black on color(240)"       # inactive tab
C_TAB_A  = "bold black on color(220)"       # active tab

# Spinner frames for RUNNING state
_SPIN = cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])

import os as _os
try:
    _TERM_WIDTH = _os.get_terminal_size().columns
except OSError:
    _TERM_WIDTH = 200  # fallback for non-TTY (background jobs, CI, piped output)

console = Console(force_terminal=True, highlight=False, width=_TERM_WIDTH)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _state() -> dict:
    try:
        if STATE_FILE.exists():
            return json.loads(STATE_FILE.read_text())
    except Exception:
        pass
    return {}


def _pnl_text(val: float) -> Text:
    arrow = "▲" if val > 0 else "▼" if val < 0 else "─"
    return Text(f"{arrow}{abs(val):.2f}%", style=C_GAIN if val > 0 else C_LOSS if val < 0 else C_DIM)


def _action_style(action: str) -> str:
    return {"BUY": C_BUY, "COVER": C_BUY, "SELL": C_SELL, "SHORT": C_SELL}.get(action, C_HOLD)


def _conf_bar(conf: int, width: int = 8) -> Text:
    """Visual confidence bar: ████░░░░ 82%"""
    filled = max(0, min(width, round(conf / 100 * width)))
    bar    = "█" * filled + "░" * (width - filled)
    style  = C_GAIN if conf >= 70 else C_WARN if conf >= 50 else C_LOSS
    t = Text()
    t.append(bar, style=style)
    t.append(f" {conf:2d}%", style=C_DIM)
    return t


def _spark(prices: list, width: int = 8) -> Text:
    """Unicode sparkline from a list of prices."""
    blocks = " ▁▂▃▄▅▆▇█"
    if not prices or len(prices) < 2:
        return Text("─" * width, style=C_DIM)
    lo, hi = min(prices), max(prices)
    rng = hi - lo or 1
    sample = prices[-(width):]
    chars  = [blocks[min(8, int((v - lo) / rng * 8))] for v in sample]
    t = Text()
    last_dir = prices[-1] >= prices[-2] if len(prices) >= 2 else True
    t.append("".join(chars), style=C_GAIN if last_dir else C_LOSS)
    return t


def _ell(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def _price(v: float) -> str:
    if v >= 10_000: return f"${v/1000:.1f}K"
    if v >= 1_000:  return f"${v:.0f}"
    return f"${v:.2f}"


def _spin() -> str:
    return next(_SPIN)


# ── Panel builders ────────────────────────────────────────────────────────────

def _header(s: dict) -> Table:
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
    tbl.add_column("brand",  ratio=2, no_wrap=True, overflow="ellipsis")
    tbl.add_column("equity", ratio=3, no_wrap=True, overflow="ellipsis")
    tbl.add_column("macro",  ratio=3, no_wrap=True, overflow="ellipsis")
    tbl.add_column("status", ratio=2, justify="right", no_wrap=True, overflow="ellipsis")

    # Row 1: brand tabs + equity + macro + status
    brand = Text()
    brand.append("  AI-TRADER ", style=C_HDR)
    brand.append("  ")
    brand.append(" MARKET ", style=C_TAB_A)
    brand.append(" SIGNALS ", style=C_TAB)
    brand.append(" PORTFOLIO ", style=C_TAB)

    eq_t = Text()
    pnl_sty = C_GAIN if pnl_d >= 0 else C_LOSS
    arrow   = "▲" if pnl_d > 0 else "▼" if pnl_d < 0 else "─"
    eq_t.append("PAPER  ", style=C_DIM)
    eq_t.append(f"${eq:>12,.2f}", style="bold bright_white")
    eq_t.append(f"  {arrow}{abs(pnl_d):,.2f}  ({pnl_pct:+.2f}%)", style=pnl_sty)

    vix_sty = C_WARN if isinstance(vix, (int, float)) and vix > 25 else C_VAL
    spy_sty = C_GAIN if isinstance(spy, (int, float)) and spy >= 0 else C_LOSS
    mac_t = Text()
    mac_t.append("VIX ", style=C_DIM)
    mac_t.append(str(vix), style=vix_sty)
    mac_t.append("  SPY ", style=C_DIM)
    mac_t.append(f"{spy}%", style=spy_sty)
    mac_t.append("  F&G ", style=C_DIM)
    mac_t.append(str(fg), style=C_VAL)
    mac_t.append("  Bot ", style=C_DIM)
    mac_t.append(str(bscore), style=C_VAL)

    stat_t = Text(justify="right")
    stat_icon = "⬤" if stat == "RUNNING" else "○"
    stat_t.append(f"{stat_icon} {stat}", style=C_RUN if stat == "RUNNING" else C_IDLE)
    if sym and sym != "—":
        stat_t.append("  ", style=C_DIM)
        stat_t.append(sym, style=C_ACCENT)
    stat_t.append(f"  {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC  ", style=C_DIM)

    # Row 2: regime + divider
    regime_t = Text()
    regime_t.append(f"  ◈ REGIME  ", style=C_DIM)
    regime_t.append(regime, style=C_ACCENT)
    regime_t.append(f"  [{upd}]" if upd else "", style=C_DIM)

    tbl.add_row(brand, eq_t, mac_t, stat_t)
    tbl.add_row(Text("  " + "─" * 20, style=C_DIM), regime_t, Text(""), Text(""))
    return tbl


def _positions_table(s: dict) -> Table:
    tbl = Table(box=None, header_style=C_ACCENT, expand=True,
                show_edge=False, pad_edge=False, padding=(0, 1))
    tbl.add_column("SYM",  style="bold bright_white", no_wrap=True, width=5)
    tbl.add_column("QTY",  justify="right",           no_wrap=True, width=4)
    tbl.add_column("$",    justify="right",           no_wrap=True, width=7)
    tbl.add_column("P&L",  justify="right",           no_wrap=True, width=7)
    tbl.add_column("SPARK",                           no_wrap=True, width=8)

    rows = s.get("positions", [])
    if not rows:
        tbl.add_row(Text("—", style=C_DIM), "—", "—", "—", "—")
        return tbl
    for p in rows[:14]:
        qty = p.get("qty", 0)
        if isinstance(qty, float):
            qty_s = f"{qty:.2f}" if qty < 10 else str(int(qty))
        else:
            qty_s = str(qty)
        history = p.get("price_history", [])
        curr    = p.get("current", 0)
        entry   = p.get("entry", 0)
        curr_t  = Text(_price(curr), style=C_GAIN if curr >= entry else C_LOSS)
        tbl.add_row(
            _ell(p.get("symbol", ""), 5),
            _ell(qty_s, 4),
            curr_t,
            _pnl_text(p.get("pnl_pct", 0)),
            _spark(history),
        )
    return tbl


def _signal_feed_table(s: dict) -> Table:
    tbl = Table(box=None, header_style=C_ACCENT, expand=True,
                show_edge=False, pad_edge=False, padding=(0, 1))
    tbl.add_column("TIME",   style=C_DIM,            no_wrap=True, width=5)
    tbl.add_column("SYM",    style="bold bright_white", no_wrap=True, width=5)
    tbl.add_column("SIGNAL",                          no_wrap=True, width=9)
    tbl.add_column("CONF",                            no_wrap=True, width=12)
    tbl.add_column("REASON", style=C_DIM,             no_wrap=True, overflow="ellipsis", ratio=1)

    feed = list(reversed(s.get("signal_feed", [])))[:20]
    if not feed:
        tbl.add_row("—", "—", "—", Text("awaiting…", style=C_DIM), "—")
        return tbl
    for ev in feed:
        action    = ev.get("action", "HOLD")
        conf      = ev.get("conf", 0)
        consensus = ev.get("consensus", "SPLIT")
        sig_sty   = (C_BUY if action in ("BUY","COVER") and consensus == "STRONG"
                     else C_SELL if action in ("SELL","SHORT") and consensus == "STRONG"
                     else C_WEAK if consensus == "WEAK"
                     else C_DIM)
        t = ev.get("time", "")[:5]
        tbl.add_row(
            t,
            _ell(ev.get("symbol", ""), 5),
            Text(f"{_ell(action,4)} ", style=_action_style(action)),
            _conf_bar(conf),
            ev.get("reason", ""),
        )
    return tbl


def _strategy_table(s: dict) -> Table:
    tbl = Table(box=None, header_style=C_ACCENT, expand=True,
                show_edge=False, pad_edge=False, padding=(0, 1))
    tbl.add_column("STRATEGY", style="bold bright_white", ratio=1, no_wrap=True, overflow="ellipsis")
    tbl.add_column("W/L",      justify="right", no_wrap=True, width=6)
    tbl.add_column("SCORE",    justify="right", no_wrap=True, width=5)
    tbl.add_column("BAR",      no_wrap=True,    width=16)

    strats = s.get("strategies", {})
    if not strats:
        tbl.add_row(Text("No data yet", style=C_DIM), "—", "—", "—")
        return tbl

    for name, v in sorted(strats.items(), key=lambda x: -x[1].get("score", 0)):
        score  = v.get("score", 0.5)
        wins   = v.get("wins", 0)
        losses = v.get("losses", 0)
        filled = max(0, min(8, round(score * 8)))
        bar    = "█" * filled + "░" * (8 - filled)
        if score >= 0.60:
            label, sty = "▲TRUST",  C_GAIN
        elif score < 0.40:
            label, sty = "▼REDUCE", C_LOSS
        else:
            label, sty = "─NEUT",   C_WARN
        tbl.add_row(
            name.replace("_", " ").title(),
            f"{wins}W/{losses}L",
            f"{score:.2f}",
            Text(f"{bar} {label}", style=sty),
        )
    return tbl


def _lessons_text(s: dict) -> Text:
    lessons = s.get("lessons", [])
    t = Text(overflow="ellipsis")
    if not lessons:
        t.append("No post-mortems yet.\n", style=C_DIM)
        t.append("Losses >2% trigger auto-diagnosis.", style=C_DIM)
        return t
    for lesson in lessons[-5:]:
        t.append("◆ ", style=C_ACCENT)
        t.append(_ell(lesson, 52) + "\n", style=C_DIM)
    return t


def _pipeline_row(s: dict) -> Text:
    pipe   = s.get("pipeline", {})
    stages = [("yfinance","DATA"), ("sentiment","SENT"),
              ("debate","DEBATE"), ("arbiter","ARBIT"), ("alpaca","EXEC")]
    t = Text()
    t.append("  PIPELINE  ", style=f"bold {C_HDR}")
    for i, (key, label) in enumerate(stages):
        info   = pipe.get(key, {})
        status = info.get("status", "IDLE")
        if status == "RUNNING":
            icon, sty = _spin(), "bold color(75)"
        elif status == "OK":
            icon, sty = "✓", C_GAIN
        elif status == "ERROR":
            icon, sty = "✗", C_LOSS
        else:
            icon, sty = "·", C_DIM
        t.append(f"{icon}", style=sty)
        t.append(label, style="bold bright_white" if status not in ("IDLE","") else C_DIM)
        if i < len(stages) - 1:
            t.append("→", style=C_DIM)
    return t


def _footer(s: dict) -> Table:
    log   = s.get("trade_log", [])
    radar = s.get("radar", {})
    wr    = s.get("win_rate", {})

    tbl = Table(box=None, show_header=False, expand=True,
                padding=(0, 1), show_edge=False)
    tbl.add_column("trades", ratio=3)
    tbl.add_column("stats",  ratio=1, justify="right")

    # Trades row
    trades_t = Text()
    trades_t.append("TRADES  ", style=f"bold {C_BDR}")
    if not log:
        trades_t.append("none yet", style=C_DIM)
    for entry in list(reversed(log))[:5]:
        action = entry.get("action", "?")
        trades_t.append(f"[{entry.get('time','')}] ", style=C_DIM)
        trades_t.append(f"{action} ", style=_action_style(action))
        trades_t.append(f"{_ell(entry.get('symbol',''), 5)} ", style="bold bright_white")
        trades_t.append(f"@{_price(entry.get('price',0))}  ", style=C_DIM)

    score   = radar.get("score", "—")
    level   = radar.get("level", "—")
    lvl_sty = {"SAFE": C_SAFE, "DANGER": C_DANGER, "WARNING": C_WARN}.get(str(level), C_DIM)
    stats_t = Text(justify="right")
    stats_t.append("RADAR ", style=C_DIM)
    stats_t.append(f"{score}/100 ", style=C_VAL)
    stats_t.append(str(level), style=lvl_sty)
    stats_t.append("  WIN ", style=C_DIM)
    wr_rate = wr.get('rate', '—')
    wr_sty  = C_GAIN if isinstance(wr_rate, (int,float)) and wr_rate >= 55 else C_LOSS if isinstance(wr_rate, (int,float)) and wr_rate < 45 else C_VAL
    stats_t.append(f"{wr_rate}%", style=wr_sty)
    stats_t.append(f" ({wr.get('trades','—')} tr)  ", style=C_DIM)

    tbl.add_row(trades_t, stats_t)
    tbl.add_row(_pipeline_row(s), Text(""))
    return tbl


# ── Layout assembly ───────────────────────────────────────────────────────────

def _build(s: dict) -> Layout:
    w = console.width or 180

    root = Layout(name="root")
    root.split_column(
        Layout(name="hdr",    size=3),
        Layout(name="body",   ratio=1),
        Layout(name="footer", size=5),
    )

    if w >= 160:
        root["body"].split_row(
            Layout(name="left",   ratio=24),
            Layout(name="center", ratio=46),
            Layout(name="right",  ratio=30),
        )
        root["right"].split_column(
            Layout(name="strategies", ratio=58),
            Layout(name="lessons",    ratio=42),
        )
        wide = True
    else:
        root["body"].split_row(
            Layout(name="left",   ratio=32),
            Layout(name="center", ratio=68),
        )
        wide = False

    root["hdr"].update(_header(s))

    root["left"].update(
        Panel(_positions_table(s),
              title=f"[{C_ACCENT}]◈ POSITIONS[/{C_ACCENT}]",
              border_style=C_BDR, expand=True)
    )
    root["center"].update(
        Panel(_signal_feed_table(s),
              title=f"[{C_ACCENT}]◈ LIVE SIGNAL FEED[/{C_ACCENT}]",
              border_style=C_BDR, expand=True)
    )
    if wide:
        root["strategies"].update(
            Panel(_strategy_table(s),
                  title=f"[{C_ACCENT}]◈ STRATEGY SCORES[/{C_ACCENT}]",
                  border_style=C_BDR, expand=True)
        )
        root["lessons"].update(
            Panel(_lessons_text(s),
                  title=f"[{C_ACCENT}]◈ LEARNED LESSONS[/{C_ACCENT}]",
                  border_style=C_BDR, expand=True)
        )
    root["footer"].update(
        Panel(_footer(s),
              border_style=C_BDR2,
              title=f"[{C_ACCENT}]◈ TRADE LOG[/{C_ACCENT}]",
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
