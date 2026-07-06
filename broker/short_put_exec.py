"""Short put execution via Alpaca options API (paper trading).

Sell-to-open one cash-secured put contract. Buy-to-close on exit.
Requires options_trading_level >= 1 on the Alpaca account.
"""

from datetime import date

try:
    from alpaca.trading.client   import TradingClient
    from alpaca.trading.requests import GetOptionContractsRequest, LimitOrderRequest
    from alpaca.trading.enums    import OrderSide, TimeInForce, ContractType
    ALPACA_OPTIONS = True
except ImportError:
    ALPACA_OPTIONS = False

from signals.short_put_screener import load_positions, save_positions


def _client(cfg: dict):
    if not ALPACA_OPTIONS:
        return None
    k = cfg.get("alpaca_api_key", "")
    s = cfg.get("alpaca_secret_key", "")
    if not k or not s:
        return None
    return TradingClient(k, s, paper=cfg.get("alpaca_paper", True))


def execute_short_put(cfg: dict, opp: dict, dry_run: bool = False) -> dict:
    """Sell-to-open one put contract at the bid. Returns result dict."""
    symbol  = opp["symbol"]
    strike  = opp["strike"]
    expiry  = opp["expiry"]
    premium = opp["premium"]

    if dry_run:
        return {"dry_run": True, "symbol": symbol, "strike": strike,
                "expiry": expiry, "premium": premium,
                "credit": round(premium * 100, 2)}

    c = _client(cfg)
    if not c:
        return {"error": "Alpaca not configured"}

    # Buying-power guard: a cash-secured put locks strike*100 as collateral.
    # Skip cleanly when the account can't cover it instead of letting Alpaca
    # reject the order with a 403 (which spammed errors every run).
    collateral = strike * 100
    try:
        avail = float(c.get_account().options_buying_power)
    except Exception:
        avail = None
    if avail is not None and collateral > avail:
        return {"skipped": True, "symbol": symbol, "strike": strike,
                "reason": f"insufficient buying power — need ${collateral:,.0f}, have ${avail:,.0f}"}

    try:
        contracts = c.get_option_contracts(GetOptionContractsRequest(
            underlying_symbols=[symbol],
            expiration_date=expiry,
            type=ContractType.PUT,
            strike_price_gte=str(strike - 0.01),
            strike_price_lte=str(strike + 0.01),
        ))
        items = (contracts.option_contracts
                 if hasattr(contracts, "option_contracts") else list(contracts))
        if not items:
            return {"error": f"No contract found: {symbol} ${strike}P {expiry}"}

        occ   = items[0].symbol
        order = c.submit_order(LimitOrderRequest(
            symbol=occ, qty=1, side=OrderSide.SELL,
            type="limit", limit_price=premium,
            time_in_force=TimeInForce.DAY,
        ))

        result = {
            "alpaca_order_id": str(order.id),
            "occ_symbol":      occ,
            "symbol":          symbol,
            "strike":          strike,
            "expiry":          expiry,
            "premium":         premium,
            "credit":          round(premium * 100, 2),
            "status":          str(order.status),
        }

        positions = load_positions()
        positions[symbol] = {
            "symbol":          symbol,
            "occ_symbol":      occ,
            "strike":          strike,
            "expiry":          expiry,
            "entry_premium":   premium,
            "qty":             1,
            "credit":          result["credit"],
            "opened_at":       date.today().isoformat(),
            "alpaca_order_id": str(order.id),
        }
        save_positions(positions)
        return result

    except Exception as e:
        return {"error": str(e)}


def close_short_put(cfg: dict, pos: dict, current_premium: float,
                    dry_run: bool = False) -> dict:
    """Buy-to-close an existing short put. Returns result dict."""
    entry = pos.get("entry_premium", current_premium)
    qty   = pos.get("qty", 1)
    pnl   = round((entry - current_premium) * qty * 100, 2)

    if dry_run:
        return {"dry_run": True, "symbol": pos["symbol"], "pnl": pnl}

    c = _client(cfg)
    if not c:
        return {"error": "Alpaca not configured"}

    try:
        occ   = pos.get("occ_symbol", "")
        order = c.submit_order(LimitOrderRequest(
            symbol=occ, qty=qty, side=OrderSide.BUY,
            type="limit", limit_price=current_premium,
            time_in_force=TimeInForce.DAY,
        ))

        positions = load_positions()
        positions.pop(pos["symbol"], None)
        save_positions(positions)

        return {"alpaca_order_id": str(order.id), "symbol": pos["symbol"],
                "pnl": pnl, "status": str(order.status)}
    except Exception as e:
        return {"error": str(e)}
