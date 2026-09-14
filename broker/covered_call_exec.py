"""covered_call_exec.py — Sell covered calls against existing long positions.
Requires 100 shares of underlying. Uses same Alpaca options API as short_put_exec.py.
"""

import json
from datetime import date
from pathlib import Path
import yfinance as yf

try:
    from alpaca.trading.client   import TradingClient
    from alpaca.trading.requests import GetOptionContractsRequest, LimitOrderRequest
    from alpaca.trading.enums    import OrderSide, TimeInForce, ContractType
    ALPACA_OPTIONS = True
except ImportError:
    ALPACA_OPTIONS = False

from broker.short_put_exec import round_to_tick

POSITIONS_FILE = Path(__file__).parent / "covered_call_positions.json"
OTM_MIN_PCT, OTM_MAX_PCT = 4, 6
MIN_DTE, MAX_DTE         = 26, 35
MIN_PREMIUM              = 0.10


def load_positions() -> dict:
    if POSITIONS_FILE.exists():
        try:
            return json.loads(POSITIONS_FILE.read_text())
        except Exception:
            pass
    return {}


def save_positions(data: dict) -> None:
    POSITIONS_FILE.write_text(json.dumps(data, indent=2))


def _client(cfg: dict):
    if not ALPACA_OPTIONS:
        return None
    k, s = cfg.get("alpaca_api_key", ""), cfg.get("alpaca_secret_key", "")
    if not k or not s:
        return None
    return TradingClient(k, s, paper=cfg.get("alpaca_paper", True))


def find_covered_call_opportunity(cfg: dict, symbol: str, current_price: float,
                                  shares_held: int) -> dict | None:
    if shares_held < 100 or symbol in load_positions():
        return None
    try:
        tk   = yf.Ticker(symbol)
        exps = tk.options
        if not exps:
            return None
        today = date.today()
        valid = [(e, (date.fromisoformat(e) - today).days) for e in exps
                 if MIN_DTE <= (date.fromisoformat(e) - today).days <= MAX_DTE]
        if not valid:
            return None
        exp_str, dte = min(valid, key=lambda x: abs(x[1] - 30))
        calls = tk.option_chain(exp_str).calls
        calls = calls[calls["bid"] > 0].copy()
        if calls.empty:
            return None
        lo, hi     = current_price * (1 + OTM_MIN_PCT / 100), current_price * (1 + OTM_MAX_PCT / 100)
        candidates = calls[(calls["strike"] >= lo) & (calls["strike"] <= hi)]
        if candidates.empty:
            return None
        best    = candidates.loc[candidates["bid"].idxmax()]
        strike  = float(best["strike"])
        premium = round(float(best["bid"]), 2)
        if premium < MIN_PREMIUM:
            return None
        return {"symbol": symbol, "price": round(current_price, 2), "strike": strike,
                "expiry": exp_str, "dte": dte, "premium": premium,
                "otm_pct": round((strike - current_price) / current_price * 100, 1),
                "credit": round(premium * 100, 2)}
    except Exception:
        return None


def execute_covered_call(cfg: dict, opp: dict, dry_run: bool = False) -> dict:
    symbol, strike, expiry, premium = opp["symbol"], opp["strike"], opp["expiry"], opp["premium"]
    if dry_run:
        return {"dry_run": True, "symbol": symbol, "strike": strike,
                "expiry": expiry, "premium": premium, "credit": round(premium * 100, 2)}
    c = _client(cfg)
    if not c:
        return {"error": "Alpaca not configured"}
    try:
        contracts = c.get_option_contracts(GetOptionContractsRequest(
            underlying_symbols=[symbol], expiration_date=expiry,
            type=ContractType.CALL,
            strike_price_gte=str(strike - 0.01), strike_price_lte=str(strike + 0.01),
        ))
        items = (contracts.option_contracts
                 if hasattr(contracts, "option_contracts") else list(contracts))
        if not items:
            return {"error": f"No contract found: {symbol} ${strike}C {expiry}"}
        occ   = items[0].symbol
        order = c.submit_order(LimitOrderRequest(
            symbol=occ, qty=1, side=OrderSide.SELL,
            type="limit", limit_price=round_to_tick(premium), time_in_force=TimeInForce.DAY,
        ))
        result = {"alpaca_order_id": str(order.id), "occ_symbol": occ, "symbol": symbol,
                  "strike": strike, "expiry": expiry, "premium": premium,
                  "credit": round(premium * 100, 2), "status": str(order.status)}
        positions = load_positions()
        positions[symbol] = {"symbol": symbol, "occ_symbol": occ, "strike": strike,
                             "expiry": expiry, "entry_premium": premium, "qty": 1,
                             "credit": result["credit"], "opened_at": date.today().isoformat(),
                             "alpaca_order_id": str(order.id)}
        save_positions(positions)
        return result
    except Exception as e:
        return {"error": str(e)}


def close_covered_call(cfg: dict, pos: dict, current_premium: float,
                       dry_run: bool = False) -> dict:
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
            type="limit", limit_price=round_to_tick(current_premium), time_in_force=TimeInForce.DAY,
        ))
        positions = load_positions()
        positions.pop(pos["symbol"], None)
        save_positions(positions)
        return {"alpaca_order_id": str(order.id), "symbol": pos["symbol"],
                "pnl": pnl, "status": str(order.status)}
    except Exception as e:
        return {"error": str(e)}
