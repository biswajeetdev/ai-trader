"""Zerodha Kite Connect execution — paper simulation now, live-ready stub.

To go live:
  pip install kiteconnect pyotp
  Add to config.json:
    "zerodha_api_key":    "...",
    "zerodha_api_secret": "...",
    "zerodha_totp_secret": "...",   # base32 secret from Zerodha 2FA setup
    "zerodha_user_id":    "..."
  Set "zerodha_live": true in config.json when ready.
"""

import json, time
from pathlib import Path
from broker.risk import record_open, record_close, load_positions

_TOKEN_FILE = Path(__file__).parent.parent / "zerodha_token.json"

# ── Auth helpers (used by approval_bot daily login) ────────────────────────────

def _daily_login(cfg: dict) -> str:
    """Perform Zerodha TOTP login and return access_token. Requires kiteconnect + pyotp."""
    import pyotp
    from kiteconnect import KiteConnect
    kite = KiteConnect(api_key=cfg["zerodha_api_key"])
    totp = pyotp.TOTP(cfg["zerodha_totp_secret"]).now()
    # Headless login via requests (Zerodha login URL)
    import requests
    session = requests.Session()
    r = session.post("https://kite.zerodha.com/api/login",
                     data={"user_id": cfg["zerodha_user_id"],
                           "password": cfg["zerodha_password"]},
                     timeout=10)
    data = r.json()
    request_id = data["data"]["request_id"]
    r2 = session.post("https://kite.zerodha.com/api/twofa",
                      data={"user_id": cfg["zerodha_user_id"],
                            "request_id": request_id,
                            "twofa_value": totp},
                      timeout=10)
    # Exchange request_token for access_token
    request_token = r2.url.split("request_token=")[1].split("&")[0]
    sess = kite.generate_session(request_token, api_secret=cfg["zerodha_api_secret"])
    access_token = sess["access_token"]
    _TOKEN_FILE.write_text(json.dumps({"access_token": access_token,
                                        "ts": time.time()}))
    return access_token


def get_access_token(cfg: dict) -> str | None:
    """Return cached access_token if fresh (< 8h), else re-login."""
    if _TOKEN_FILE.exists():
        data = json.loads(_TOKEN_FILE.read_text())
        if time.time() - data.get("ts", 0) < 28800:  # 8 hours
            return data["access_token"]
    try:
        return _daily_login(cfg)
    except Exception:
        return None


# ── Order execution ────────────────────────────────────────────────────────────

def execute_india_trade(cfg: dict, symbol: str, action: str, qty: int,
                        reason: str, price: float, atr: float,
                        dry_run: bool = False) -> dict:
    """Execute an approved Indian market trade.

    Currently: paper simulation via positions.json.
    Flip cfg["zerodha_live"] = true to route through Kite Connect.
    """
    nse_symbol = symbol.replace(".NS", "")

    if cfg.get("zerodha_live") and not dry_run:
        return _execute_kite(cfg, nse_symbol, action, qty, price, atr, dry_run)

    # ── Paper simulation ───────────────────────────────────────────────────────
    local_positions = load_positions()
    result = {"simulated": True, "symbol": symbol, "action": action,
              "qty": qty, "price": price, "market": "in-stock"}
    if not dry_run:
        if action == "BUY":
            record_open(symbol, price, qty, atr, "in-stock")
        elif action in ("SELL", "COVER") and symbol in local_positions:
            record_close(symbol, price)
    print(f"   [PAPER] {action} {qty}x {symbol} @ ₹{price:.2f}")
    return result


def _execute_kite(cfg: dict, nse_symbol: str, action: str, qty: int,
                  price: float, atr: float, dry_run: bool) -> dict:
    """Live Kite Connect execution with GTT stop-loss."""
    from kiteconnect import KiteConnect
    kite = KiteConnect(api_key=cfg["zerodha_api_key"])
    kite.set_access_token(get_access_token(cfg))

    tx = "BUY" if action == "BUY" else "SELL"
    order_id = kite.place_order(
        variety=kite.VARIETY_REGULAR,
        exchange=kite.EXCHANGE_NSE,
        tradingsymbol=nse_symbol,
        transaction_type=tx,
        quantity=qty,
        order_type=kite.ORDER_TYPE_MARKET,
        product=kite.PRODUCT_CNC,  # delivery
    )

    result = {"order_id": order_id, "symbol": nse_symbol,
              "action": action, "qty": qty, "live": True}

    # Place GTT stop-loss for BUY orders
    if action == "BUY" and atr:
        stop_price  = round(price - 3.0 * atr, 2)
        target_price = round(price + 7.5 * atr, 2)
        gtt_id = kite.place_gtt(
            trigger_type=kite.GTT_TYPE_OCO,
            tradingsymbol=nse_symbol,
            exchange=kite.EXCHANGE_NSE,
            trigger_values=[stop_price, target_price],
            last_price=price,
            orders=[
                {"transaction_type": "SELL", "quantity": qty,
                 "order_type": "LIMIT", "price": stop_price,  "product": "CNC"},
                {"transaction_type": "SELL", "quantity": qty,
                 "order_type": "LIMIT", "price": target_price, "product": "CNC"},
            ],
        )
        result["gtt_id"]    = gtt_id
        result["stop"]      = stop_price
        result["target"]    = target_price
        print(f"   GTT set: stop ₹{stop_price} | target ₹{target_price}")

    return result


# ── F&O short put execution (paper + live) ────────────────────────────────────

def execute_india_short_put(cfg: dict, opp: dict, dry_run: bool = False) -> dict:
    """
    Sell 1 lot of the OTM put described in opp.

    opp fields (from india_short_put_screener.find_india_short_put_opportunity):
        symbol, strike, expiry_str, dte, premium, lot_size, credit_inr, otm_pct
    """
    from signals.india_short_put_screener import load_positions, save_positions
    from datetime import datetime

    symbol    = opp["symbol"]
    strike    = opp["strike"]
    exp_str   = opp["expiry_str"]     # "26-Jun-2026" format for Kite
    lot_size  = opp["lot_size"]
    premium   = opp["premium"]

    # Kite F&O tradingsymbol format: RELIANCE26JUN25250PE
    # Simpler: use the NSE symbol directly in live mode
    result = {
        "symbol":        symbol,
        "strike":        strike,
        "expiry":        opp["expiry"],
        "lot_size":      lot_size,
        "entry_premium": premium,
        "credit_inr":    opp["credit_inr"],
        "qty":           1,   # 1 contract = lot_size shares
        "entry_date":    datetime.now().strftime("%Y-%m-%d"),
        "dry_run":       dry_run,
    }

    if dry_run:
        print(f"   [DRY] SELL PUT {symbol} ₹{strike}P {exp_str} "
              f"bid ₹{premium:.1f}/sh → credit ₹{opp['credit_inr']:.0f} "
              f"({lot_size} lot, {opp['dte']}DTE, {opp['otm_pct']}% OTM)")
        result["status"] = "DRY"
        return result

    if cfg.get("zerodha_live"):
        try:
            from broker.zerodha_enctoken import get_enctoken, place_order as enc_order
            enc = get_enctoken(cfg)
            if not enc:
                result["error"] = "No enctoken — run: python3 scripts/kite_daily_login.py"
                return result
            ts  = opp.get("tradingsymbol") or _kite_option_symbol(symbol, strike, exp_str, "PE")
            res = enc_order(ts, lot_size, premium, "SELL", enc)
            if "error" in res:
                result["error"] = res["error"]
                return result
            result["order_id"] = res["order_id"]
            result["status"]   = "LIVE"
        except Exception as e:
            result["error"] = str(e)
            return result
    else:
        print(f"   [PAPER] SELL PUT {symbol} ₹{strike}P {exp_str} "
              f"bid ₹{premium:.1f}/sh → credit ₹{opp['credit_inr']:.0f}")
        result["status"] = "PAPER"

    # Track position
    positions = load_positions()
    positions[symbol] = result
    save_positions(positions)
    return result


def close_india_short_put(cfg: dict, pos: dict,
                           current_premium: float | None,
                           dry_run: bool = False) -> dict:
    """Buy back the short put to close the position."""
    from signals.india_short_put_screener import load_positions, save_positions

    symbol    = pos["symbol"]
    current   = current_premium or pos.get("entry_premium", 0) * 0.01
    entry     = pos.get("entry_premium", 0)
    lot_size  = pos.get("lot_size", 250)
    pnl       = round((entry - current) * lot_size, 2)

    result = {"symbol": symbol, "pnl_inr": pnl, "status": "closed"}

    if dry_run:
        print(f"   [DRY] BUY PUT {symbol} @ ₹{current:.1f} | P&L ₹{pnl:+.0f}")
        result["status"] = "DRY"
        return result

    if cfg.get("zerodha_live"):
        try:
            from broker.zerodha_enctoken import get_enctoken, place_order as enc_order
            enc = get_enctoken(cfg)
            if not enc:
                result["error"] = "No enctoken"
                return result
            ts  = pos.get("tradingsymbol") or _kite_option_symbol(
                symbol, pos["strike"], pos.get("expiry_str", ""), "PE")
            res = enc_order(ts, lot_size, current, "BUY", enc, order_type="MARKET")
            if "error" in res:
                result["error"] = res["error"]
                return result
        except Exception as e:
            result["error"] = str(e)
            return result
    else:
        print(f"   [PAPER] BUY PUT {symbol} @ ₹{current:.1f} | P&L ₹{pnl:+.0f}")

    positions = load_positions()
    positions.pop(symbol, None)
    save_positions(positions)
    return result


def _kite_option_symbol(symbol: str, strike: float, exp_str: str, opt_type: str) -> str:
    """Build Zerodha NFO tradingsymbol like RELIANCE26JUN25250PE."""
    try:
        dt = None
        for fmt in ("%d-%b-%Y", "%d-%B-%Y"):
            try:
                from datetime import datetime as _dt
                dt = _dt.strptime(exp_str, fmt)
                break
            except ValueError:
                pass
        if not dt:
            return f"{symbol}{int(strike)}{opt_type}"
        day   = dt.strftime("%d")
        month = dt.strftime("%b").upper()
        year  = dt.strftime("%y")
        return f"{symbol}{day}{month}{year}{int(strike)}{opt_type}"
    except Exception:
        return f"{symbol}{int(strike)}{opt_type}"
