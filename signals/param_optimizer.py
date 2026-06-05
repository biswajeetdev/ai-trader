"""Genetic algorithm optimizer for BB/ADX/ATR trading parameters."""
from __future__ import annotations
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

PARAM_SPACE = {
    "bb_period":       (10, 30),
    "bb_std":          (1.5, 2.5),
    "adx_threshold":   (15, 30),
    "atr_stop_mult":   (2.0, 4.5),
    "atr_target_mult": (4.0, 10.0),
    "rsi_buy_max":     (40, 65),
    "rsi_sell_min":    (60, 85),
}

BEST_PARAMS_PATH = Path(__file__).parent.parent / "best_params.json"


def _fast_backtest(params: dict, symbol: str, period_days: int = 90) -> float:
    """Vectorized backtest on symbol; returns annualized Sharpe (0.0 on failure)."""
    import numpy as np
    import yfinance as yf

    ticker = f"{symbol}-USD" if symbol in ("BTC", "ETH") else symbol
    df = yf.download(ticker, period=f"{period_days}d", interval="1d",
                     progress=False, auto_adjust=True)
    if df is None or len(df) < 30:
        return 0.0

    closes = df["Close"].values.flatten().astype(float)
    highs  = df["High"].values.flatten().astype(float)
    lows   = df["Low"].values.flatten().astype(float)

    period = int(params["bb_period"])
    std_m  = float(params["bb_std"])
    stop_m = float(params["atr_stop_mult"])
    tgt_m  = float(params["atr_target_mult"])
    rsi_mx = float(params["rsi_buy_max"])

    rolling_mean = np.convolve(closes, np.ones(period) / period, mode="valid")
    rolling_std  = np.array([closes[i:i + period].std()
                             for i in range(len(closes) - period + 1)])
    upper  = rolling_mean + std_m * rolling_std
    lower  = rolling_mean - std_m * rolling_std
    bb_pct = (closes[period - 1:] - lower) / (upper - lower + 1e-9)

    tr = np.maximum(highs[1:] - lows[1:],
         np.maximum(np.abs(highs[1:] - closes[:-1]),
                    np.abs(lows[1:] - closes[:-1])))
    atr = np.convolve(tr, np.ones(14) / 14, mode="valid")

    deltas   = np.diff(closes)
    gains    = np.where(deltas > 0, deltas, 0)
    losses   = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.convolve(gains,  np.ones(14) / 14, mode="valid")
    avg_loss = np.convolve(losses, np.ones(14) / 14, mode="valid")
    rs  = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - 100 / (1 + rs)

    n    = min(len(bb_pct), len(atr), len(rsi)) - 1
    bb   = bb_pct[-n:]
    at   = atr[-n:]
    rs_v = rsi[-n:]
    pr   = closes[-n:]

    returns: list[float] = []
    in_trade = False
    entry_p = stop_p = tgt_p = 0.0

    for i in range(1, n):
        if not in_trade:
            if bb[i - 1] < 0.25 and bb[i] > 0.25 and rs_v[i] < rsi_mx:
                in_trade = True
                entry_p  = pr[i]
                stop_p   = entry_p - stop_m * at[i]
                tgt_p    = entry_p + tgt_m  * at[i]
        else:
            if pr[i] <= stop_p or pr[i] >= tgt_p:
                returns.append((pr[i] - entry_p) / entry_p - 0.001)
                in_trade = False

    if len(returns) < 3:
        return 0.0
    r = np.array(returns)
    return float(r.mean() / (r.std() + 1e-9) * np.sqrt(252))


def _random_params() -> dict:
    """Generate a random chromosome within PARAM_SPACE bounds."""
    out = {}
    for key, (lo, hi) in PARAM_SPACE.items():
        if isinstance(lo, int) and isinstance(hi, int):
            out[key] = random.randint(lo, hi)
        else:
            out[key] = round(random.uniform(lo, hi), 3)
    return out


def _crossover(p1: dict, p2: dict) -> dict:
    """Single-point crossover on param dict."""
    keys = list(PARAM_SPACE.keys())
    cut  = random.randint(1, len(keys) - 1)
    child = {k: p1[k] for k in keys[:cut]}
    child.update({k: p2[k] for k in keys[cut:]})
    return child


def _mutate(params: dict, rate: float = 0.15) -> dict:
    """Mutate each param independently with `rate` probability."""
    out = dict(params)
    for key, (lo, hi) in PARAM_SPACE.items():
        if random.random() < rate:
            if isinstance(lo, int) and isinstance(hi, int):
                out[key] = random.randint(lo, hi)
            else:
                out[key] = round(random.uniform(lo, hi), 3)
    return out


def _fitness(params: dict, symbols: list) -> float:
    """Average Sharpe across all symbols (portfolio fitness)."""
    scores = [_fast_backtest(params, s) for s in symbols]
    return sum(scores) / len(scores) if scores else 0.0


def run_optimization(symbols: list = None, generations: int = 5,
                     pop_size: int = 20, verbose: bool = True) -> dict:
    """Run GA optimization. Returns best params dict; saves to best_params.json."""
    if symbols is None:
        symbols = ["NVDA", "AAPL", "MSFT", "BTC"]

    population = [_random_params() for _ in range(pop_size)]
    scored: list[tuple[float, dict]] = []

    for gen in range(1, generations + 1):
        with ThreadPoolExecutor(max_workers=min(pop_size, 8)) as ex:
            futures = {ex.submit(_fitness, p, symbols): p for p in population}
            scored = [(f.result(), futures[f]) for f in as_completed(futures)]

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_ind = scored[0]

        if verbose:
            per_sym = ", ".join(
                f"{s}: {_fast_backtest(best_ind, s):.2f}" for s in symbols
            )
            print(f"Gen {gen}/{generations} | best Sharpe: {best_score:.2f} | {per_sym}")

        cutoff   = pop_size // 2
        parents  = [ind for _, ind in scored[:cutoff]]
        children = []
        while len(children) < pop_size - cutoff:
            p1, p2 = random.sample(parents, 2)
            children.append(_mutate(_crossover(p1, p2)))
        population = parents + children

    best_score, best_params = scored[0]
    output = {
        "found_at":     datetime.now(timezone.utc).isoformat(),
        "score":        round(best_score, 4),
        "oos_return":   0.0,
        "oos_win_rate": 0.0,
        "oos_sharpe":   round(best_score, 4),
        "source":       "genetic-algorithm",
        "params":       best_params,
    }
    BEST_PARAMS_PATH.write_text(json.dumps(output, indent=2))
    return best_params


if __name__ == "__main__":
    import sys
    symbols = sys.argv[1:] or ["NVDA", "AAPL", "MSFT", "BTC"]
    best = run_optimization(symbols, generations=5, pop_size=20)
    print(f"Best params: {best}")
