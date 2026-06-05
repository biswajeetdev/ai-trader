"""Indian market signals: NSE bulk deals, India VIX, NSE options flow."""

import json, time
from pathlib import Path
import requests
import yfinance as yf

_CACHE_FILE = Path(__file__).parent.parent / "signals_cache.json"
_CACHE: dict = {}


def _load_cache() -> dict:
    global _CACHE
    if not _CACHE and _CACHE_FILE.exists():
        try:
            _CACHE = json.loads(_CACHE_FILE.read_text())
        except Exception:
            _CACHE = {}
    return _CACHE


def _save_cache(data: dict):
    global _CACHE
    _CACHE = data
    try:
        _CACHE_FILE.write_text(json.dumps(data, indent=2))
    except Exception:
        pass


def _cached(key: str, ttl: int, fn):
    cache = _load_cache()
    entry = cache.get(key, {})
    if entry and time.time() - entry.get("ts", 0) < ttl:
        return entry["data"]
    result = fn()
    cache[key] = {"ts": time.time(), "data": result}
    _save_cache(cache)
    return result


def get_nse_bulk_deals(tickers: list) -> list:
    def _fetch():
        try:
            url = "https://www.nseindia.com/api/bulk-deal-data"
            headers = {
                "User-Agent": "Mozilla/5.0",
                "Accept": "*/*",
                "Referer": "https://www.nseindia.com",
            }
            session = requests.Session()
            session.get("https://www.nseindia.com", headers=headers, timeout=8)
            r = session.get(url, headers=headers, timeout=8)
            if not r.ok:
                return []
            raw = r.json()
            deals = raw if isinstance(raw, list) else raw.get("data", [])
            # Normalise: NSE returns snake_case or camelCase depending on endpoint
            result = []
            stripped = {t.replace(".NS", "").upper() for t in tickers}
            for d in deals:
                sym = (d.get("symbol") or d.get("Symbol") or "").upper()
                if sym in stripped:
                    result.append({
                        "symbol": sym,
                        "client_name": d.get("clientName") or d.get("client_name") or "",
                        "buy_sell": d.get("buySell") or d.get("buy_sell") or "",
                        "quantity": d.get("quantityTraded") or d.get("quantity") or 0,
                        "price": d.get("tradePrice") or d.get("price") or 0,
                        "date": d.get("mktType") or d.get("date") or "",
                    })
            return result
        except Exception:
            return []

    return _cached("nse_bulk_deals", 3600, _fetch)


def get_india_vix() -> dict:
    def _fetch():
        try:
            df = yf.download("^INDIAVIX", period="5d", interval="1d",
                             progress=False, auto_adjust=True)
            vix = float(df["Close"].squeeze().dropna().iloc[-1])
            if vix < 13:
                level = "CALM"
            elif vix < 20:
                level = "NORMAL"
            elif vix < 30:
                level = "ELEVATED"
            else:
                level = "CRISIS"
            return {"vix": round(vix, 2), "level": level}
        except Exception:
            return {"vix": None, "level": "UNKNOWN"}

    return _cached("india_vix", 900, _fetch)


def get_nse_options_flow(symbol: str) -> str:
    try:
        from nsepython import nse_optionchain_scrapper
        data = nse_optionchain_scrapper(symbol)
        records = data.get("records", {}).get("data", [])
        call_oi = sum(r.get("CE", {}).get("openInterest", 0) for r in records if r.get("CE"))
        put_oi  = sum(r.get("PE", {}).get("openInterest", 0) for r in records if r.get("PE"))
        pcr = round(put_oi / call_oi, 2) if call_oi > 0 else 1.0
        if pcr < 0.7:
            sentiment = "BULLISH"
        elif pcr > 1.3:
            sentiment = "BEARISH"
        else:
            sentiment = "NEUTRAL"
        call_str = f"{call_oi/1e5:.1f}L" if call_oi > 1e5 else str(call_oi)
        put_str  = f"{put_oi/1e5:.1f}L"  if put_oi  > 1e5 else str(put_oi)
        return f"PCR {pcr} | Call OI {call_str} | Put OI {put_str} | {sentiment}"
    except Exception:
        return ""


def format_bulk_deals_for_llm(deals: list, symbol: str) -> str:
    sym = symbol.replace(".NS", "").upper()
    filtered = [d for d in deals if d.get("symbol", "").upper() == sym]
    if not filtered:
        return ""
    lines = [f"Bulk deal: {d['client_name']} {d['buy_sell']} {d['quantity']}@₹{d['price']}"
             for d in filtered]
    return "\n".join(lines)
