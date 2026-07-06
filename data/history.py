"""data/history.py — resilient daily OHLCV fetch for the eval/backtest loops.

`get_daily(symbol, start, end)` returns a yfinance-shaped DataFrame
(DatetimeIndex, columns Open/High/Low/Close/Volume). It tries, in order:

  1. on-disk CSV cache  (data/cache/<symbol>_<start>_<end>_1d.csv)
  2. yfinance           (retry + exponential backoff)
  3. Alpaca daily bars  (split/dividend-adjusted) — hard fallback

Why this exists: eval_llm.py / backtest.py loop `yf.download` per symbol.
Yahoo rate-limits bursts to empty responses that yfinance mislabels as
"possibly delisted; no price data found", so the scheduled look-ahead-bias
eval collected ZERO decisions. Alpaca paper data has no per-call throttle,
so it guarantees the loop yields real numbers even when Yahoo throttles.

Output is normalised to flat, capitalised OHLCV columns on every path, so a
cache round-trip and the yfinance/Alpaca paths all feed compute_all() the
same shape.
"""

import json
import time
from pathlib import Path

import pandas as pd

_ROOT     = Path(__file__).resolve().parent.parent
CACHE_DIR = _ROOT / "data" / "cache"
_OHLCV    = ["Open", "High", "Low", "Close", "Volume"]


def _load_cfg() -> dict:
    try:
        return json.loads((_ROOT / "config.json").read_text())
    except Exception:
        return {}


def _normalize(df):
    """Flatten to single-level, capitalised OHLCV with a tz-naive DatetimeIndex."""
    if df is None or df.empty:
        return None
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):          # yf single-ticker download
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={c: str(c).capitalize() for c in df.columns})
    df.index = pd.to_datetime(df.index)
    try:
        df.index = df.index.tz_localize(None)          # tz-aware -> naive
    except (TypeError, AttributeError):
        pass                                           # already tz-naive
    keep = [c for c in _OHLCV if c in df.columns]
    if "Close" not in keep:
        return None
    return df[keep]


def _cache_path(symbol: str, start: str, end: str) -> Path:
    safe = symbol.replace("/", "_").replace("-", "_")
    return CACHE_DIR / f"{safe}_{start}_{end}_1d.csv"


def _from_cache(path: Path):
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df if not df.empty else None
    except Exception:
        return None


def _to_cache(path: Path, df) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)
    except Exception:
        pass                                           # cache is best-effort


def _from_yfinance(symbol, start, end, retries: int = 3):
    try:
        import yfinance as yf
    except ImportError:
        return None
    for attempt in range(retries):
        try:
            df = yf.download(symbol, start=start, end=end, interval="1d",
                             progress=False, auto_adjust=True)
            df = _normalize(df)
            if df is not None:
                return df
        except Exception:
            pass
        time.sleep(1.5 * (2 ** attempt))               # 1.5s, 3s, 6s backoff
    return None


def _from_alpaca(symbol, start, end):
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests   import StockBarsRequest
        from alpaca.data.timeframe  import TimeFrame
        from alpaca.data.enums      import Adjustment
    except ImportError:
        return None
    cfg = _load_cfg()
    k, s = cfg.get("alpaca_api_key", ""), cfg.get("alpaca_secret_key", "")
    if not k or not s:
        return None
    try:
        client = StockHistoricalDataClient(k, s)
        req = StockBarsRequest(
            symbol_or_symbols=symbol, timeframe=TimeFrame.Day,
            start=pd.Timestamp(start), end=pd.Timestamp(end),
            adjustment=Adjustment.ALL,
        )
        df = client.get_stock_bars(req).df
        if df is None or df.empty:
            return None
        if isinstance(df.index, pd.MultiIndex):        # (symbol, timestamp) -> timestamp
            df = df.reset_index(level=0, drop=True)
        return _normalize(df)
    except Exception:
        return None


def get_daily(symbol: str, start, end):
    """Return daily OHLCV for `symbol` in [start, end], or an empty DataFrame.

    Drop-in for the `yf.download(..., interval="1d", auto_adjust=True)` calls
    in eval_llm.py / backtest.py. Never raises for a missing/throttled symbol —
    an empty DataFrame lets callers skip it via their existing `df.empty` check.
    """
    start_s, end_s = str(start), str(end)
    path = _cache_path(symbol, start_s, end_s)

    cached = _from_cache(path)
    if cached is not None:
        return cached

    df = _from_yfinance(symbol, start, end)
    if df is None:
        df = _from_alpaca(symbol, start, end)
    if df is None or df.empty:
        return pd.DataFrame()

    _to_cache(path, df)
    return df
