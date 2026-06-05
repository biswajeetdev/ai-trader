"""
alpaca_exec.py — Alpaca Paper Trading Integration

Mirrors every trade to Alpaca's paper account in parallel with ai4trade.ai.
Paper trading = real market prices, real fills, real slippage — most realistic sim.

Setup (one-time):
  1. Sign up at alpaca.markets (free)
  2. Go to Paper Trading → API Keys → Generate
  3. Add to config.json:
       "alpaca_api_key":    "PKXXXXXXXXXXXXXXXX",
       "alpaca_secret_key": "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

Switch to live:
  Set "alpaca_paper": false in config.json
  Add live API keys (requires funded account)
"""

import json, os
from pathlib import Path
from datetime import datetime

try:
    from alpaca.trading.client   import TradingClient
    from alpaca.trading.requests import (MarketOrderRequest, LimitOrderRequest,
                                         StopLossRequest, TakeProfitRequest)
    from alpaca.trading.enums    import OrderSide, TimeInForce, OrderType, OrderClass
    from alpaca.data.historical  import StockHistoricalDataClient
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

from broker.risk import STOP_LOSS_ATR, PROFIT_TARGET_ATR

DIR = Path(__file__).parent.parent


def _get_client(cfg):
    if not ALPACA_AVAILABLE:
        return None
    key    = cfg.get("alpaca_api_key","")
    secret = cfg.get("alpaca_secret_key","")
    paper  = cfg.get("alpaca_paper", True)   # always True until user explicitly sets False
    if not key or not secret:
        return None
    return TradingClient(key, secret, paper=paper)


def get_alpaca_portfolio(cfg):
    """Returns Alpaca account state: cash, positions, P&L."""
    client = _get_client(cfg)
    if not client:
        return None
    try:
        acct   = client.get_account()
        pos    = client.get_all_positions()
        return {
            "cash":         float(acct.cash),
            "equity":       float(acct.equity),
            "buying_power": float(acct.buying_power),
            "pnl_today":    float(acct.equity) - float(acct.last_equity),
            "positions": [{
                "symbol":     p.symbol,
                "qty":        float(p.qty),
                "avg_entry":  float(p.avg_entry_price),
                "current":    float(p.current_price or 0),
                "pnl":        float(p.unrealized_pl or 0),
                "pnl_pct":    float(p.unrealized_plpc or 0) * 100,
            } for p in pos]
        }
    except Exception as e:
        return {"error": str(e)}


def execute_alpaca_trade(cfg, symbol, market, action, quantity, reason,
                         dry_run=False, price=None, atr=None):
    """
    Submit order to Alpaca.
    BUY on US stocks with price+atr → bracket order (native stop-loss + take-profit).
    SELL, COVER, or crypto → plain market order (brackets unsupported for crypto).
    Returns order details dict.
    """
    client = _get_client(cfg)
    if not client:
        return {"skipped": "Alpaca not configured"}

    alpaca_symbol = f"{symbol}/USD" if market == "crypto" else symbol
    side = OrderSide.BUY if action.upper() in ("BUY", "COVER") else OrderSide.SELL

    if dry_run:
        stop  = round(price - STOP_LOSS_ATR * atr, 2) if price and atr else None
        tgt   = round(price + PROFIT_TARGET_ATR * atr, 2) if price and atr else None
        return {"dry_run": True, "symbol": alpaca_symbol, "action": action,
                "qty": quantity, "stop": stop, "target": tgt}

    use_bracket = (
        side == OrderSide.BUY
        and market == "us-stock"   # Alpaca bracket not supported for crypto
        and price is not None
        and atr is not None
        and atr > 0
    )

    try:
        if use_bracket:
            stop_price = round(price - STOP_LOSS_ATR * atr, 2)
            take_price = round(price + PROFIT_TARGET_ATR * atr, 2)
            req = MarketOrderRequest(
                symbol        = alpaca_symbol,
                qty           = quantity,
                side          = side,
                time_in_force = TimeInForce.GTC,   # bracket orders require GTC
                order_class   = OrderClass.BRACKET,
                stop_loss     = StopLossRequest(stop_price=stop_price),
                take_profit   = TakeProfitRequest(limit_price=take_price),
            )
        else:
            stop_price = take_price = None
            req = MarketOrderRequest(
                symbol        = alpaca_symbol,
                qty           = quantity,
                side          = side,
                time_in_force = TimeInForce.DAY,
                extended_hours = False,
            )

        order = client.submit_order(req)
        result = {
            "alpaca_order_id": str(order.id),
            "symbol":   alpaca_symbol,
            "side":     str(side),
            "qty":      float(order.qty or quantity),
            "status":   str(order.status),
            "submitted": datetime.now().isoformat(),
        }
        if use_bracket:
            result["stop"]   = stop_price
            result["target"] = take_price
        return result
    except Exception as e:
        return {"error": str(e)}


def get_alpaca_status(cfg):
    """Quick status check — returns paper/live mode and account health."""
    client = _get_client(cfg)
    if not client:
        return "Alpaca not configured. Add alpaca_api_key + alpaca_secret_key to config.json"
    try:
        acct = client.get_account()
        mode = "PAPER" if cfg.get("alpaca_paper", True) else "LIVE"
        return (f"Alpaca {mode} | Cash: ${float(acct.cash):,.2f} | "
                f"Equity: ${float(acct.equity):,.2f} | Status: {acct.status}")
    except Exception as e:
        return f"Alpaca error: {e}"


def export_alpaca_to_excel(cfg, writer):
    """Add Alpaca sheet to existing Excel writer."""
    port = get_alpaca_portfolio(cfg)
    if not port or "error" in port:
        return

    import pandas as pd
    rows = port.get("positions", [])
    df   = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["symbol","qty","avg_entry","current","pnl","pnl_pct"])
    df_sum = pd.DataFrame([{
        "Source":   "Alpaca Paper",
        "Cash ($)": round(port["cash"],2),
        "Equity ($)":round(port["equity"],2),
        "Today P&L":round(port["pnl_today"],2),
    }])
    df_sum.to_excel(writer, sheet_name="Alpaca_Summary", index=False)
    df.to_excel(writer,     sheet_name="Alpaca_Positions", index=False)


if __name__ == "__main__":
    cfg = json.loads((DIR/"config.json").read_text())
    print(get_alpaca_status(cfg))
    port = get_alpaca_portfolio(cfg)
    if port and "error" not in port:
        print(f"Cash: ${port['cash']:,.2f} | Positions: {len(port['positions'])}")
        for p in port["positions"]:
            print(f"  {p['symbol']:6} qty={p['qty']} pnl=${p['pnl']:+.2f} ({p['pnl_pct']:+.1f}%)")
