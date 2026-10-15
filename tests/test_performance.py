import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from broker import performance as perf  # noqa: E402


def test_breakeven_trades_are_not_losses():
    trades = [{"pnl": 100}, {"pnl": -50}, {"pnl": 0}, {"pnl": 0}]
    r = perf.win_rate_and_profit_factor(trades)
    assert (r["winning_trades"], r["losing_trades"], r["breakeven_trades"]) == (1, 1, 2)
    assert r["avg_loss"] == -50  # zeros no longer drag the average loss towards 0
    assert r["profit_factor"] == 2.0


def test_sortino_uses_downside_deviation_over_all_periods():
    rets = [0.01, -0.02, 0.015, -0.005] * 6
    equity = [100.0]
    for x in rets:
        equity.append(equity[-1] * (1 + x))
    excess = np.diff(equity) / np.array(equity[:-1]) - perf.RISK_FREE_ANNUAL / perf.TRADING_DAYS
    expected = np.mean(excess) / math.sqrt(np.mean(np.minimum(excess, 0) ** 2)) * math.sqrt(perf.TRADING_DAYS)
    assert perf.sortino_ratio(equity) == round(float(expected), 3)


def test_report_hides_equity_ratios_on_intraday_history(tmp_path, monkeypatch):
    hwm = tmp_path / "portfolio_hwm.json"
    hwm.write_text(json.dumps({"equity_history": [100000 + i * 10 for i in range(30)]}))
    trades = tmp_path / "trade_history.json"
    trades.write_text(json.dumps([{"pnl": 25, "entry_date": "2026-06-01", "exit_date": "2026-06-03"}]))
    monkeypatch.setattr(perf, "HWM_FILE", hwm)
    monkeypatch.setattr(perf, "TRADE_HIST", trades)
    report = perf.get_performance_report()
    assert report["sharpe_ratio"] is None and report["sortino_ratio"] is None and report["max_drawdown_pct"] is None
    assert report["equity_metrics_note"] == perf.NEEDS_DAILY
    text = perf.format_for_summary(report)
    assert "Sharpe    : needs daily equity history" in text
    assert "Win Rate  : 100.0%" in text and "Avg Hold  : 2.0d" in text


def test_max_drawdown_fraction():
    assert perf.max_drawdown([100, 120, 90, 130]) == 0.25
