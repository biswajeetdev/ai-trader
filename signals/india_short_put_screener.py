"""India short put screener — OTM puts on NSE individual stocks.

Entry: RSI 40-65, SMA50 > SMA200 (bull regime), no earnings within 8d,
       OTM 4-9%, DTE 21-45, bid >= ₹2.
Exit:  50% profit or DTE <= 7 (gamma risk).

Data:
  1. Zerodha enctoken (live bid/ask via OMS) — requires kite_daily_login.py
  2. NSE web API fallback (may be rate-limited)
Exec:  Paper: JSON tracking only. Live: zerodha_exec.py + enctoken.
"""

import json
import time
import requests
import yfinance as yf
from datetime import date, datetime
from pathlib import Path

DIR            = Path(__file__).parent.parent
POSITIONS_FILE = DIR / "india_short_put_positions.json"

# NSE lot sizes — verified from live Kite instruments CSV 2026-06-04
NSE_LOT_SIZES = {
    "RELIANCE":   500,
    "TCS":        175,
    "HDFCBANK":   550,
    "INFY":       400,
    "ICICIBANK":  700,
    "WIPRO":     3000,
    "SBIN":       750,
    "BAJFINANCE": 750,
    "TITAN":      175,
    "ASIANPAINT": 250,
    "KOTAKBANK": 2000,
    "AXISBANK":   625,
    "LT":         175,
    "MARUTI":      50,
    "BHARTIARTL": 475,
    "NIFTY":       25,
    "BANKNIFTY":   15,
    "FINNIFTY":    40,
}
DEFAULT_LOT = 500

# Screener params
MAX_VIX          = 20
MIN_RSI          = 40
MAX_RSI          = 65
OTM_MIN_PCT      = 4.0
OTM_MAX_PCT      = 9.0
MIN_DTE          = 21
MAX_DTE          = 45
MIN_PREMIUM_INR  = 2.0   # ₹2/share minimum
MAX_OPEN         = 3
PROFIT_CLOSE_PCT = 0.50
GAMMA_DTE        = 7


# ── Position store ────────────────────────────────────────────────────────────

def load_positions() -> dict:
    if POSITIONS_FILE.exists():
        try:
            return json.loads(POSITIONS_FILE.read_text())
        except Exception:
            pass
    return {}


def save_positions(data: dict) -> None:
    POSITIONS_FILE.write_text(json.dumps(data, indent=2))


# ── Option chain helpers ──────────────────────────────────────────────────────

def _puts_via_enctoken(nse_symbol: str, spot: float) -> list[dict]:
    """Use Zerodha enctoken for live bid/ask (primary path)."""
    try:
        from broker.zerodha_enctoken import (
            get_enctoken, find_otm_puts, enrich_puts_with_quotes
        )
        enc = get_enctoken()
        if not enc:
            return []
        candidates = find_otm_puts(nse_symbol, spot, MIN_DTE, MAX_DTE,
                                   OTM_MIN_PCT, OTM_MAX_PCT)
        if not candidates:
            return []
        enriched = enrich_puts_with_quotes(candidates, enc)
        return [
            {
                "tradingsymbol": r["tradingsymbol"],
                "strike":        r["_strike"],
                "expiry":        r["_expiry"].isoformat(),
                "dte":           r["_dte"],
                "bid":           r["_bid"],
                "ask":           r["_ask"],
                "last":          r["_ltp"],
                "lot_size":      r["_lot_size"] or NSE_LOT_SIZES.get(nse_symbol, DEFAULT_LOT),
                "oi":            r["_oi"],
                "source":        "enctoken",
            }
            for r in enriched
        ]
    except Exception:
        return []


_NSE_SESSION = None
_NSE_TS: float = 0


def _puts_via_nse(nse_symbol: str, spot: float) -> list[dict]:
    """NSE web API fallback — may be blocked outside market hours."""
    global _NSE_SESSION, _NSE_TS
    try:
        now = time.time()
        if not _NSE_SESSION or now - _NSE_TS > 300:
            s = requests.Session()
            s.headers.update({
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                              "AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
                "Referer": "https://www.nseindia.com/",
            })
            s.get("https://www.nseindia.com", timeout=8)
            _NSE_SESSION, _NSE_TS = s, now

        url = (f"https://www.nseindia.com/api/option-chain-equities"
               f"?symbol={nse_symbol}")
        r = _NSE_SESSION.get(url, timeout=10)
        if not r.ok:
            return []

        today   = date.today()
        records = r.json().get("records", {}).get("data", [])
        puts    = []
        lo = spot * (1 - OTM_MAX_PCT / 100)
        hi = spot * (1 - OTM_MIN_PCT / 100)

        for row in records:
            pe = row.get("PE")
            if not pe:
                continue
            strike = float(row.get("strikePrice", 0))
            if not (lo <= strike <= hi):
                continue
            try:
                exp_dt = datetime.strptime(pe["expiryDate"], "%d-%b-%Y").date()
            except Exception:
                continue
            dte = (exp_dt - today).days
            if not (MIN_DTE <= dte <= MAX_DTE):
                continue
            bid = float(pe.get("bidprice") or 0)
            puts.append({
                "tradingsymbol": "",
                "strike":        strike,
                "expiry":        exp_dt.isoformat(),
                "dte":           dte,
                "bid":           bid,
                "ask":           float(pe.get("askPrice") or 0),
                "last":          float(pe.get("lastPrice") or 0),
                "lot_size":      NSE_LOT_SIZES.get(nse_symbol, DEFAULT_LOT),
                "oi":            int(pe.get("openInterest") or 0),
                "source":        "nse",
            })
        return puts
    except Exception:
        return []


# ── Screener ──────────────────────────────────────────────────────────────────

def find_india_short_put_opportunity(nse_symbol: str, ind: dict, fund: dict,
                                     india_vix: float | None) -> dict | None:
    """
    Screen an NSE stock for a short put entry.
    nse_symbol: bare symbol e.g. 'RELIANCE' (not 'RELIANCE.NS')
    ind:  compute_indicators() dict  fund: get_fundamentals() dict
    """
    if india_vix and india_vix > MAX_VIX:
        return None

    positions = load_positions()
    if nse_symbol in positions or len(positions) >= MAX_OPEN:
        return None

    rsi   = ind.get("rsi14", 50)
    price = ind.get("price", 0)
    sma50 = ind.get("sma50", price)

    if not (MIN_RSI <= rsi <= MAX_RSI):
        return None
    if price < sma50:
        return None

    days_earn = fund.get("days_to_earnings")
    if days_earn is not None and 0 < days_earn < 8:
        return None

    try:
        spot = price

        # Primary: Zerodha enctoken (live bid/ask)
        puts = _puts_via_enctoken(nse_symbol, spot)
        source = "enctoken"

        # Fallback: NSE web API
        if not puts:
            puts = _puts_via_nse(nse_symbol, spot)
            source = "nse"

        if not puts:
            return None

        # Filter by minimum premium
        valid = [p for p in puts if p["bid"] >= MIN_PREMIUM_INR]
        if not valid:
            return None

        # Best: closest to 30 DTE, then highest bid
        valid.sort(key=lambda p: (abs(p["dte"] - 30), -p["bid"]))
        best     = valid[0]
        lot_size = best.get("lot_size") or NSE_LOT_SIZES.get(nse_symbol, DEFAULT_LOT)
        credit   = round(best["bid"] * lot_size, 2)
        max_risk = round((best["strike"] - best["bid"]) * lot_size, 2)

        print(f"     [chain:{source}] {len(valid)} valid OTM puts for {nse_symbol}")

        return {
            "symbol":        nse_symbol,
            "ns_symbol":     nse_symbol + ".NS",
            "tradingsymbol": best.get("tradingsymbol", ""),
            "price":         round(spot, 2),
            "strike":        best["strike"],
            "expiry":        best["expiry"],
            "expiry_str":    best["expiry"],
            "dte":           best["dte"],
            "premium":       best["bid"],
            "lot_size":      lot_size,
            "credit_inr":    credit,
            "max_risk_inr":  max_risk,
            "otm_pct":       round((spot - best["strike"]) / spot * 100, 1),
            "oi":            best.get("oi", 0),
            "rsi":           round(rsi, 1),
            "data_source":   source,
        }
    except Exception:
        return None


def check_india_exits() -> list[dict]:
    """Return open positions that hit 50%-profit or DTE ≤ 7."""
    positions = load_positions()
    to_close  = []
    today     = date.today()

    for symbol, pos in list(positions.items()):
        try:
            dte = (date.fromisoformat(pos["expiry"]) - today).days
            if dte <= GAMMA_DTE:
                to_close.append({**pos, "close_reason": f"DTE={dte}",
                                 "current_premium": None})
                continue

            # Try to get current premium via enctoken
            current = None
            ts = pos.get("tradingsymbol", "")
            if ts:
                try:
                    from broker.zerodha_enctoken import get_enctoken, get_ltp
                    enc = get_enctoken()
                    if enc:
                        ltp_map = get_ltp([ts], enc)
                        current = ltp_map.get(ts)
                except Exception:
                    pass

            if current is None:
                continue   # can't verify — skip until next run

            entry = pos.get("entry_premium", 0)
            if current <= entry * (1 - PROFIT_CLOSE_PCT):
                pnl = round((entry - current) * pos.get("lot_size", DEFAULT_LOT), 2)
                to_close.append({**pos, "close_reason": f"50% profit +₹{pnl}",
                                 "current_premium": current})
        except Exception:
            pass
    return to_close
