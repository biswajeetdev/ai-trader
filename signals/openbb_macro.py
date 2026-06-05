"""
openbb_macro.py — FRED macro enrichment via OpenBB

Adds signals the VIX-only regime misses:
  - Yield curve spread (T10Y2Y): inversion = recession warning
  - CPI 3-month trend: rising = tighter Fed, bearish for equities
  - Unemployment trend: rising = labor deterioration
  - Fed Funds Rate: level = monetary tightness

All data via FRED (free key). Cached — FRED data is daily/monthly, no need to hit per run.
"""

import json, time
from pathlib import Path
from datetime import datetime, timezone

CACHE_FILE = Path(__file__).parent.parent / "signals_cache.json"
CACHE_TTL  = 7200   # 2h — FRED updates daily/monthly, no need to refresh often


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


def _fred(symbol: str, limit: int = 6) -> list[float | None]:
    """Fetch last N values for a FRED series. Returns list newest-first."""
    try:
        from openbb import obb
        r = obb.economy.fred_series(symbol=symbol, limit=limit, sort="desc")
        return [getattr(row, symbol, None) for row in r.results]
    except Exception:
        return []


def get_macro_enrichment() -> dict:
    """
    Returns enriched macro context dict to augment regime.py output.

    Keys added:
        yield_curve        float | None   — T10Y2Y spread (negative = inverted)
        yield_curve_signal str            — "INVERTED" | "FLAT" | "NORMAL"
        cpi_trend          str            — "RISING" | "FALLING" | "FLAT" | "UNKNOWN"
        cpi_latest         float | None
        unemployment       float | None
        unemployment_trend str            — "RISING" | "FALLING" | "FLAT" | "UNKNOWN"
        fed_rate           float | None   — current Fed Funds rate
        recession_risk     str            — "HIGH" | "ELEVATED" | "LOW"
        macro_size_adj     float          — additional size multiplier (0.5–1.0)
        macro_summary      str            — one-line LLM-ready summary
    """
    cached = _cache("openbb_macro")
    if cached:
        return cached

    # ── Yield curve (T10Y2Y): 10yr minus 2yr Treasury spread ─────────────────
    yc_vals = _fred("T10Y2Y", limit=5)
    yield_curve = yc_vals[0] if yc_vals else None
    if yield_curve is not None:
        if yield_curve < -0.3:
            yc_signal = "INVERTED"
        elif yield_curve < 0.3:
            yc_signal = "FLAT"
        else:
            yc_signal = "NORMAL"
    else:
        yc_signal = "UNKNOWN"

    # ── CPI trend (last 3 months) ─────────────────────────────────────────────
    cpi_vals = _fred("CPIAUCSL", limit=4)
    cpi_latest = cpi_vals[0] if cpi_vals else None
    if len(cpi_vals) >= 3:
        # Compare newest vs 3 months ago
        delta = (cpi_vals[0] or 0) - (cpi_vals[2] or 0)
        cpi_trend = "RISING" if delta > 0.3 else ("FALLING" if delta < -0.3 else "FLAT")
    else:
        cpi_trend = "UNKNOWN"

    # ── Unemployment trend (last 3 months) ────────────────────────────────────
    ur_vals = _fred("UNRATE", limit=4)
    unemployment = ur_vals[0] if ur_vals else None
    if len(ur_vals) >= 3 and ur_vals[0] is not None and ur_vals[2] is not None:
        delta = ur_vals[0] - ur_vals[2]
        ur_trend = "RISING" if delta > 0.2 else ("FALLING" if delta < -0.2 else "FLAT")
    else:
        ur_trend = "UNKNOWN"

    # ── Fed Funds Rate ────────────────────────────────────────────────────────
    ffr_vals = _fred("FEDFUNDS", limit=2)
    fed_rate = ffr_vals[0] if ffr_vals else None

    # ── Composite recession risk ──────────────────────────────────────────────
    risk_score = 0
    if yc_signal == "INVERTED":
        risk_score += 2
    elif yc_signal == "FLAT":
        risk_score += 1
    if ur_trend == "RISING":
        risk_score += 2
    if cpi_trend == "RISING":
        risk_score += 1

    if risk_score >= 4:
        recession_risk = "HIGH"
        macro_size_adj = 0.6
    elif risk_score >= 2:
        recession_risk = "ELEVATED"
        macro_size_adj = 0.8
    else:
        recession_risk = "LOW"
        macro_size_adj = 1.0

    # ── Summary ───────────────────────────────────────────────────────────────
    parts = []
    if yield_curve is not None:
        parts.append(f"Yield curve {yield_curve:+.2f}% ({yc_signal})")
    if cpi_latest is not None:
        parts.append(f"CPI {cpi_latest:.1f} ({cpi_trend})")
    if unemployment is not None:
        parts.append(f"Unemployment {unemployment:.1f}% ({ur_trend})")
    if fed_rate is not None:
        parts.append(f"Fed rate {fed_rate:.2f}%")
    parts.append(f"Recession risk: {recession_risk}")

    result = {
        "yield_curve":        round(yield_curve, 3) if yield_curve is not None else None,
        "yield_curve_signal": yc_signal,
        "cpi_latest":         round(cpi_latest, 2) if cpi_latest is not None else None,
        "cpi_trend":          cpi_trend,
        "unemployment":       round(unemployment, 2) if unemployment is not None else None,
        "unemployment_trend": ur_trend,
        "fed_rate":           round(fed_rate, 2) if fed_rate is not None else None,
        "recession_risk":     recession_risk,
        "macro_size_adj":     macro_size_adj,
        "macro_summary":      " | ".join(parts),
        "fetched_at":         datetime.now(timezone.utc).isoformat(),
    }
    _cache("openbb_macro", result)
    return result


def format_for_llm(macro: dict) -> str:
    return macro.get("macro_summary", "FRED macro unavailable")
