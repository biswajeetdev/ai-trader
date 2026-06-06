"""short_exec.py — Short sell execution via Alpaca paper trading.
Alpaca paper: SELL on non-owned symbol = short position. COVER = BUY to close.
"""

import json
from datetime import datetime
from pathlib import Path

try:
    from alpaca.trading.client   import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
    from alpaca.trading.enums    import OrderSide, TimeInForce
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

SHORT_POS = Path(__file__).parent / "short_positions.json"


def _get_client(cfg):
    if not ALPACA_AVAILABLE:
        return None
    key    = cfg.get("alpaca_api_key", "")
    secret = cfg.get("alpaca_secret_key", "")
    paper  = cfg.get("alpaca_paper", True)
    if not key or not secret:
        return None
    return TradingClient(key, secret, paper=paper)


def _load_shorts() -> dict:
    return json.loads(SHORT_POS.read_text()) if SHORT_POS.exists() else {}


def _save_shorts(pos: dict) -> None:
    SHORT_POS.write_text(json.dumps(pos, indent=2))


def execute_short_sell(cfg, symbol, qty, limit_price=None, dry_run=False) -> dict:
    # Submit SELL order; Alpaca paper auto-creates short if symbol not held long
    if dry_run:
        return {"order_id": "dry-run", "symbol": symbol, "qty": qty,
                "price": limit_price, "status": "dry_run"}

    client = _get_client(cfg)
    if not client:
        return {"error": "Alpaca not configured — add alpaca_api_key + alpaca_secret_key to config.json"}

    try:
        if limit_price:
            req = LimitOrderRequest(
                symbol=symbol, qty=qty, side=OrderSide.SELL,
                limit_price=limit_price, time_in_force=TimeInForce.DAY)
        else:
            req = MarketOrderRequest(
                symbol=symbol, qty=qty, side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY)

        order = client.submit_order(req)
        result = {
            "order_id":  str(order.id),
            "symbol":    symbol,
            "qty":       float(order.qty or qty),
            "price":     limit_price,
            "status":    str(order.status),
            "submitted": datetime.now().isoformat(),
        }
        pos = _load_shorts()
        pos[symbol] = {"entry_price": limit_price, "qty": qty,
                       "entered": datetime.now().isoformat()}
        _save_shorts(pos)
        return result
    except Exception as e:
        return {"error": str(e)}


def cover_short(cfg, symbol, qty, limit_price=None, dry_run=False) -> dict:
    # BUY to close (cover) an existing short position
    if dry_run:
        return {"order_id": "dry-run", "symbol": symbol, "qty": qty,
                "price": limit_price, "status": "dry_run"}

    client = _get_client(cfg)
    if not client:
        return {"error": "Alpaca not configured — add alpaca_api_key + alpaca_secret_key to config.json"}

    try:
        if limit_price:
            req = LimitOrderRequest(
                symbol=symbol, qty=qty, side=OrderSide.BUY,
                limit_price=limit_price, time_in_force=TimeInForce.DAY)
        else:
            req = MarketOrderRequest(
                symbol=symbol, qty=qty, side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY)

        order = client.submit_order(req)
        pos = _load_shorts()
        pos.pop(symbol, None)  # remove from tracking on cover
        _save_shorts(pos)
        return {
            "order_id":  str(order.id),
            "symbol":    symbol,
            "qty":       float(order.qty or qty),
            "price":     limit_price,
            "status":    str(order.status),
            "submitted": datetime.now().isoformat(),
        }
    except Exception as e:
        return {"error": str(e)}
