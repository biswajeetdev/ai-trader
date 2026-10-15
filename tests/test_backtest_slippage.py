import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import backtest  # noqa: E402


def _synthetic_prices(days=400, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=days)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, days)))
    open_ = close * (1 + rng.normal(0, 0.004, days))
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close,
                         "Volume": rng.integers(1_000_000, 2_000_000, days)}, index=idx)


def _run(monkeypatch, slippage):
    df = _synthetic_prices()
    monkeypatch.setattr(backtest.yf, "download", lambda *a, **k: df.copy())
    params = {**backtest.load_params(), "slippage": slippage} if hasattr(backtest, "load_params") else {"slippage": slippage}
    return backtest.run_asset("TEST", "stock", "2024-01-01", "2025-12-31", 100_000, params)


def test_slippage_fills_adversely_and_costs_money(monkeypatch):
    clean = _run(monkeypatch, 0.0)
    slipped = _run(monkeypatch, 0.002)
    assert clean and slipped and clean["trades"] > 0, "synthetic series must produce trades"
    assert slipped["final_value"] < clean["final_value"]
    buys_clean = [t for t in clean["trade_log"] if t["action"] == "BUY"]
    buys_slip = [t for t in slipped["trade_log"] if t["action"] == "BUY"]
    assert buys_slip[0]["price"] > buys_clean[0]["price"]  # buys fill above the reference price
    exits_clean = [t for t in clean["trade_log"] if t["action"] != "BUY"]
    exits_slip = [t for t in slipped["trade_log"] if t["action"] != "BUY"]
    if exits_clean and exits_slip and exits_clean[0]["date"] == exits_slip[0]["date"]:
        assert exits_slip[0]["price"] < exits_clean[0]["price"]  # sells fill below it
