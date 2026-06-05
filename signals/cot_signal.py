"""
cot_signal.py — CFTC Commitment of Traders (COT) macro regime signal
Free public API: https://publicreporting.cftc.gov/resource/6dca-aqww.json

Logic:
- Commercial hedger net position at multi-year HIGH → bullish (they use futures to hedge, so net long = they're selling hedges = bullish for underlying)
- Non-commercial (speculator) net position at extreme HIGH → contrarian bearish (crowded long)
- Focuses on equity index futures (S&P 500 E-mini) as market-wide regime gauge
"""

import json, time, requests
from pathlib import Path
from datetime import datetime, timezone

CACHE_FILE = Path(__file__).parent.parent / "signals_cache.json"
CACHE_TTL  = 86400  # COT is weekly, 24h cache is fine

# CFTC public API — S&P 500 E-mini futures (market code 13874A)
COT_URL = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
COT_PARAMS = {
    "cftc_market_code_in_initials": "CME",
    "$where": "market_and_exchange_names LIKE '%E-MINI S&P 500%'",
    "$order": "report_date_as_mm_dd_yyyy DESC",
    "$limit": "12",
}


def _cache(key, val=None):
    try:
        data = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
        if val is None:
            e = data.get(key)
            return e["data"] if e and time.time() - e.get("ts", 0) < CACHE_TTL else None
        data[key] = {"data": val, "ts": time.time()}
        CACHE_FILE.write_text(json.dumps(data, indent=2))
    except Exception:
        return None


def get_cot_signal() -> dict:
    """Returns COT-based regime signal for S&P 500 E-mini."""
    cached = _cache("cot_signal")
    if cached:
        return cached

    try:
        r = requests.get(COT_URL, params=COT_PARAMS, timeout=15)
        rows = r.json()
    except Exception:
        return {"signal": "NEUTRAL", "confidence_delta": 0, "summary": "COT data unavailable"}

    if not rows or len(rows) < 4:
        return {"signal": "NEUTRAL", "confidence_delta": 0, "summary": "COT insufficient data"}

    try:
        # Non-commercial (speculators) net position trend
        nc_nets = []
        for row in rows[:8]:
            nc_long  = float(row.get("noncomm_positions_long_all", 0) or 0)
            nc_short = float(row.get("noncomm_positions_short_all", 0) or 0)
            nc_nets.append(nc_long - nc_short)

        current_nc = nc_nets[0]
        avg_nc = sum(nc_nets[1:]) / max(len(nc_nets) - 1, 1)
        nc_z = (current_nc - avg_nc) / (max(abs(avg_nc), 1))  # rough z-score

        # Commercial hedgers net position (inverted signal)
        comm_long  = float(rows[0].get("comm_positions_long_all", 0) or 0)
        comm_short = float(rows[0].get("comm_positions_short_all", 0) or 0)
        comm_net = comm_long - comm_short

        # Interpret
        if nc_z > 1.5:
            signal = "BEARISH"  # speculators extremely crowded long → contrarian
            delta = -10
            summary = f"COT: speculators crowded LONG (z={nc_z:.1f}) → contrarian bearish"
        elif nc_z < -1.5:
            signal = "BULLISH"  # speculators extremely short → contrarian bullish
            delta = 8
            summary = f"COT: speculators crowded SHORT (z={nc_z:.1f}) → contrarian bullish"
        elif comm_net > 0:
            signal = "BULLISH"
            delta = 5
            summary = f"COT: commercials net LONG (hedge sellers bullish) | specs z={nc_z:.1f}"
        else:
            signal = "NEUTRAL"
            delta = 0
            summary = f"COT: neutral positioning | specs z={nc_z:.1f}"

        result = {
            "signal":            signal,
            "confidence_delta":  delta,
            "summary":           summary,
            "nc_net":            round(current_nc),
            "nc_z_score":        round(nc_z, 2),
            "comm_net":          round(comm_net),
            "report_date":       rows[0].get("report_date_as_mm_dd_yyyy", ""),
        }
        _cache("cot_signal", result)
        return result

    except Exception as e:
        return {"signal": "NEUTRAL", "confidence_delta": 0, "summary": f"COT parse error: {e}"}


if __name__ == "__main__":
    r = get_cot_signal()
    print(r["summary"])
