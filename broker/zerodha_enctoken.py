"""
zerodha_enctoken.py — Zerodha internal API using enctoken (no paid API needed).

Authentication: TOTP login → enctoken cookie → use as Bearer token.
Data:  NFO instruments CSV (public, no auth) + OMS quotes (requires enctoken).
Orders: kite.zerodha.com/oms/orders/regular (requires enctoken + zerodha_live: true).

Setup in config.json:
    "zerodha_user_id":     "ZQ1234",
    "zerodha_password":    "yourpassword",
    "zerodha_totp_secret": "BASE32_FROM_2FA_APP",
    "zerodha_live":        false   ← set true only when ready for real orders
"""

import csv, io, json, time
import requests
from datetime import date, datetime
from pathlib import Path

DIR        = Path(__file__).parent.parent
TOKEN_FILE = DIR / "zerodha_token.json"

KITE_BASE  = "https://kite.zerodha.com"
OMS_BASE   = "https://kite.zerodha.com/oms"
NFO_CSV    = "https://api.kite.trade/instruments/NFO"

# Cache
_instruments_cache: list[dict] = []
_instruments_date: str = ""


# ── Auth ──────────────────────────────────────────────────────────────────────

def login(cfg: dict) -> str:
    """
    TOTP login → returns enctoken. Saves to zerodha_token.json.
    Requires: zerodha_user_id, zerodha_password, zerodha_totp_secret in cfg.
    """
    import pyotp

    s    = requests.Session()
    s.headers.update({
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/124.0.0.0 Safari/537.36"),
        "X-Kite-Version": "3",
    })

    # Step 1: password login
    r1 = s.post(f"{KITE_BASE}/api/login",
                data={"user_id": cfg["zerodha_user_id"],
                      "password": cfg["zerodha_password"]},
                timeout=10)
    r1.raise_for_status()
    data1 = r1.json()
    if data1.get("status") != "success":
        raise RuntimeError(f"Login failed: {data1.get('message', r1.text[:100])}")
    request_id = data1["data"]["request_id"]

    # Step 2: TOTP 2FA
    totp = pyotp.TOTP(cfg["zerodha_totp_secret"]).now()
    r2   = s.post(f"{KITE_BASE}/api/twofa",
                  data={"user_id":    cfg["zerodha_user_id"],
                        "request_id": request_id,
                        "twofa_value": totp,
                        "twofa_type":  "totp"},
                  timeout=10)
    r2.raise_for_status()

    # enctoken is in the cookie jar after 2FA
    enctoken = s.cookies.get("enctoken")
    if not enctoken:
        # Some versions return it in the JSON body
        body = r2.json()
        enctoken = (body.get("data", {}) or {}).get("enctoken")
    if not enctoken:
        raise RuntimeError("enctoken not found in response cookies or body")

    _save_token(enctoken)
    return enctoken


def _save_token(enctoken: str) -> None:
    TOKEN_FILE.write_text(json.dumps({
        "enctoken": enctoken,
        "ts":       time.time(),
        "date":     date.today().isoformat(),
    }, indent=2))
    TOKEN_FILE.chmod(0o600)


def get_enctoken(cfg: dict | None = None) -> str | None:
    """Return today's cached enctoken, or None if missing/stale."""
    if TOKEN_FILE.exists():
        try:
            d = json.loads(TOKEN_FILE.read_text())
            if d.get("date") == date.today().isoformat() and d.get("enctoken"):
                return d["enctoken"]
        except Exception:
            pass
    if cfg:
        try:
            token = login(cfg)
            print("   [Kite] Auto-login successful, enctoken saved.")
            return token
        except Exception as e:
            print(f"   [Kite] Auto-login failed: {e}")
    return None


def _headers(enctoken: str) -> dict:
    return {
        "Authorization": f"enctoken {enctoken}",
        "X-Kite-Version": "3",
        "Content-Type":   "application/x-www-form-urlencoded",
        "User-Agent":     "Mozilla/5.0",
    }


# ── NFO Instruments (public, no auth) ────────────────────────────────────────

def get_nfo_instruments(force: bool = False) -> list[dict]:
    """
    Download and cache NFO instruments CSV. Public endpoint, no auth needed.
    Updates once per day. Returns list of dicts with instrument fields.
    """
    global _instruments_cache, _instruments_date
    today = date.today().isoformat()
    if not force and _instruments_cache and _instruments_date == today:
        return _instruments_cache

    try:
        r = requests.get(NFO_CSV, timeout=20)
        r.raise_for_status()
        reader = csv.DictReader(io.StringIO(r.text))
        rows   = []
        for row in reader:
            # Parse expiry to date
            try:
                row["_expiry"] = date.fromisoformat(row["expiry"])
            except Exception:
                row["_expiry"] = None
            try:
                row["_strike"]   = float(row["strike"] or 0)
                row["_lot_size"] = int(row["lot_size"] or 0)
            except Exception:
                row["_strike"] = row["_lot_size"] = 0
            rows.append(row)
        _instruments_cache = rows
        _instruments_date  = today
        return rows
    except Exception as e:
        print(f"   [Kite] NFO instruments fetch failed: {e}")
        return _instruments_cache   # return stale cache rather than empty


def find_otm_puts(nse_symbol: str, spot: float,
                  min_dte: int = 21, max_dte: int = 45,
                  otm_min_pct: float = 4, otm_max_pct: float = 9) -> list[dict]:
    """
    Filter NFO instruments for OTM puts on the given symbol within DTE/OTM range.
    Returns list of instruments (dicts with all CSV fields + computed fields).
    """
    instruments = get_nfo_instruments()
    today       = date.today()
    lo = spot * (1 - otm_max_pct / 100)
    hi = spot * (1 - otm_min_pct / 100)

    candidates = []
    for row in instruments:
        if (row.get("name", "").strip('"') != nse_symbol):
            continue
        if row.get("instrument_type") != "PE":
            continue
        if row.get("segment") != "NFO-OPT":
            continue
        exp = row["_expiry"]
        if not exp:
            continue
        dte = (exp - today).days
        if not (min_dte <= dte <= max_dte):
            continue
        strike = row["_strike"]
        if not (lo <= strike <= hi):
            continue
        row["_dte"] = dte
        candidates.append(row)

    return sorted(candidates, key=lambda r: (abs(r["_dte"] - 30), -r["_strike"]))


# ── Live Quotes (requires enctoken) ──────────────────────────────────────────

def get_quotes(tradingsymbols: list[str], enctoken: str) -> dict:
    """
    Fetch live quotes for a list of "NFO:RELIANCE26JUN1300PE" instruments.
    Returns dict: {"NFO:SYMBOL": {last_price, depth, oi, ...}}
    """
    if not tradingsymbols:
        return {}
    try:
        params = "&".join(f"i=NFO:{ts}" for ts in tradingsymbols)
        r = requests.get(f"{OMS_BASE}/instruments/quote?{params}",
                         headers=_headers(enctoken), timeout=10)
        if not r.ok:
            return {}
        return r.json().get("data", {})
    except Exception:
        return {}


def get_ltp(tradingsymbols: list[str], enctoken: str) -> dict:
    """Faster endpoint — only last_price, no depth. {ts: last_price}"""
    if not tradingsymbols:
        return {}
    try:
        params = "&".join(f"i=NFO:{ts}" for ts in tradingsymbols)
        r = requests.get(f"{OMS_BASE}/instruments/ltp?{params}",
                         headers=_headers(enctoken), timeout=8)
        if not r.ok:
            return {}
        data = r.json().get("data", {})
        return {k.replace("NFO:", ""): v.get("last_price", 0) for k, v in data.items()}
    except Exception:
        return {}


def enrich_puts_with_quotes(candidates: list[dict], enctoken: str) -> list[dict]:
    """
    Fetch live bid/ask for each candidate put and add to the dict.
    candidates: output of find_otm_puts()
    """
    if not candidates or not enctoken:
        return candidates

    ts_list = [r["tradingsymbol"] for r in candidates]
    quotes  = {}
    for i in range(0, len(ts_list), 10):   # Kite quotes max 10 per call
        batch = ts_list[i:i+10]
        quotes.update(get_quotes(batch, enctoken))

    enriched = []
    for row in candidates:
        ts  = row["tradingsymbol"]
        q   = quotes.get(f"NFO:{ts}", {})
        depth = q.get("depth", {})
        buy_depth  = (depth.get("buy")  or [{}])
        sell_depth = (depth.get("sell") or [{}])
        bid = float(buy_depth[0].get("price", 0)  if buy_depth  else 0)
        ask = float(sell_depth[0].get("price", 0) if sell_depth else 0)
        ltp = float(q.get("last_price", float(row.get("last_price", 0))))
        if bid <= 0 and ltp > 0:
            bid = round(ltp * 0.97, 2)   # estimate when no book data
        row["_bid"] = round(bid, 2)
        row["_ask"] = round(ask, 2)
        row["_ltp"] = round(ltp, 2)
        row["_oi"]  = int(q.get("oi", 0))
        enriched.append(row)
    return enriched


# ── Order placement (requires enctoken + zerodha_live: true) ─────────────────

def place_order(tradingsymbol: str, qty: int, price: float,
                transaction_type: str, enctoken: str,
                order_type: str = "LIMIT", product: str = "NRML") -> dict:
    """
    Place NFO order via Zerodha OMS.
    transaction_type: "BUY" or "SELL"
    Returns {"order_id": "...", "status": "ok"} or {"error": "..."}
    """
    try:
        r = requests.post(
            f"{OMS_BASE}/orders/regular",
            headers=_headers(enctoken),
            data={
                "exchange":         "NFO",
                "tradingsymbol":    tradingsymbol,
                "transaction_type": transaction_type,
                "quantity":         str(qty),
                "price":            str(price),
                "trigger_price":    "0",
                "order_type":       order_type,
                "product":          product,
                "validity":         "DAY",
                "disclosed_quantity": "0",
                "squareoff":        "0",
                "stoploss":         "0",
                "trailing_stoploss": "0",
                "variety":          "regular",
            },
            timeout=12,
        )
        body = r.json()
        if r.ok and body.get("status") == "success":
            return {"order_id": body["data"]["order_id"], "status": "ok"}
        return {"error": body.get("message", r.text[:200])}
    except Exception as e:
        return {"error": str(e)}


def get_positions(enctoken: str) -> list[dict]:
    """Fetch open F&O positions."""
    try:
        r = requests.get(f"{OMS_BASE}/portfolio/positions",
                         headers=_headers(enctoken), timeout=10)
        data = r.json().get("data", {})
        return data.get("day", []) + data.get("net", [])
    except Exception:
        return []


def get_margins(enctoken: str) -> dict:
    """Fetch available margin (funds)."""
    try:
        r = requests.get(f"{OMS_BASE}/user/margins",
                         headers=_headers(enctoken), timeout=10)
        return r.json().get("data", {}).get("equity", {})
    except Exception:
        return {}
