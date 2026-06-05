"""
bb_patterns.py — Bollinger Band Pattern Recognition

Detects W-Bottom, M-Top, and Head & Shoulders patterns
using the last 30 candles of daily OHLCV data.
"""

import numpy as np
from scipy.signal import find_peaks


def detect_bb_pattern(df, ind) -> dict:
    """
    Detect Bollinger Band chart patterns from the last 30 candles.

    Args:
        df:  pandas DataFrame from yfinance (90d daily, columns: Open/High/Low/Close/Volume)
        ind: dict with keys: price, bb_position (0-1), rsi14, atr14, sma20

    Returns:
        dict with keys: pattern, signal, confidence, description
    """
    _NONE = {"pattern": "NONE", "signal": "NEUTRAL", "confidence": 0,
             "description": "No Bollinger Band pattern detected"}

    try:
        window = df.tail(30).copy()
        if len(window) < 15:
            return _NONE

        close = window["Close"].values.flatten().astype(float)
        n = len(close)

        # ── Compute per-candle BB position (0=lower, 1=upper) ────────────────
        sma = np.convolve(close, np.ones(20) / 20, mode="same")
        # rolling std — use min_periods=5 approach via loop for simplicity
        rol_std = np.array([
            np.std(close[max(0, i - 19):i + 1]) for i in range(n)
        ])
        rol_std = np.where(rol_std < 1e-8, 1e-8, rol_std)
        upper_bb = sma + 2 * rol_std
        lower_bb = sma - 2 * rol_std
        bb_range = upper_bb - lower_bb
        bb_range = np.where(bb_range < 1e-8, 1e-8, bb_range)
        bb_pos = (close - lower_bb) / bb_range  # 0 = lower band, 1 = upper band

        # ── Find peaks and troughs ────────────────────────────────────────────
        # Minimum distance of 3 bars between peaks/troughs
        peak_idxs, _   = find_peaks(close,  distance=3)
        trough_idxs, _ = find_peaks(-close, distance=3)

        result = _NONE

        # ────────────────────────────────────────────────────────────────────
        # W-BOTTOM: two troughs near lower BB with bullish RSI divergence
        # ────────────────────────────────────────────────────────────────────
        if len(trough_idxs) >= 2:
            t1_idx = trough_idxs[-2]
            t2_idx = trough_idxs[-1]
            t1_price = close[t1_idx]
            t2_price = close[t2_idx]
            t1_bb    = bb_pos[t1_idx]
            t2_bb    = bb_pos[t2_idx]

            # Both troughs within 5% of each other
            price_diff_pct = abs(t1_price - t2_price) / max(t1_price, t2_price)

            # Both near/below lower BB
            both_near_lower = (t1_bb < 0.15) and (t2_bb < 0.15)

            # Second trough higher than first (higher low)
            higher_low = t2_price > t1_price

            if price_diff_pct <= 0.05 and both_near_lower and higher_low:
                # Approximate RSI at each trough via momentum proxy
                # Use 14-bar momentum as RSI proxy since we don't have per-bar RSI
                def _rsi_approx(prices, idx, period=14):
                    start = max(0, idx - period)
                    chunk = prices[start:idx + 1]
                    if len(chunk) < 3:
                        return 50.0
                    gains = np.maximum(np.diff(chunk), 0)
                    losses = np.maximum(-np.diff(chunk), 0)
                    avg_gain = np.mean(gains) if len(gains) else 0
                    avg_loss = np.mean(losses) if len(losses) else 1e-8
                    rs = avg_gain / (avg_loss + 1e-8)
                    return 100 - 100 / (1 + rs)

                rsi_t1 = _rsi_approx(close, t1_idx)
                rsi_t2 = _rsi_approx(close, t2_idx)
                bullish_div = rsi_t2 > rsi_t1  # RSI higher at second trough

                if bullish_div:
                    div_strength = min((rsi_t2 - rsi_t1) / 20.0, 1.0)  # 0–1
                    confidence   = int(65 + div_strength * 20)           # 65–85
                    confidence   = max(65, min(85, confidence))
                    result = {
                        "pattern":     "W_BOTTOM",
                        "signal":      "BUY",
                        "confidence":  confidence,
                        "description": (
                            f"W-Bottom: two troughs at ${t1_price:.2f}/{t2_price:.2f} "
                            f"both below lower BB; RSI divergence +{rsi_t2-rsi_t1:.1f}pt"
                        ),
                    }

        # ────────────────────────────────────────────────────────────────────
        # M-TOP: two peaks near upper BB with bearish RSI divergence
        # ────────────────────────────────────────────────────────────────────
        if result["pattern"] == "NONE" and len(peak_idxs) >= 2:
            p1_idx = peak_idxs[-2]
            p2_idx = peak_idxs[-1]
            p1_price = close[p1_idx]
            p2_price = close[p2_idx]
            p1_bb    = bb_pos[p1_idx]
            p2_bb    = bb_pos[p2_idx]

            price_diff_pct = abs(p1_price - p2_price) / max(p1_price, p2_price)
            both_near_upper = (p1_bb > 0.85) and (p2_bb > 0.85)
            lower_high = p2_price < p1_price  # lower high

            if price_diff_pct <= 0.05 and both_near_upper and lower_high:
                def _rsi_approx(prices, idx, period=14):
                    start = max(0, idx - period)
                    chunk = prices[start:idx + 1]
                    if len(chunk) < 3:
                        return 50.0
                    gains = np.maximum(np.diff(chunk), 0)
                    losses = np.maximum(-np.diff(chunk), 0)
                    avg_gain = np.mean(gains) if len(gains) else 0
                    avg_loss = np.mean(losses) if len(losses) else 1e-8
                    rs = avg_gain / (avg_loss + 1e-8)
                    return 100 - 100 / (1 + rs)

                rsi_p1 = _rsi_approx(close, p1_idx)
                rsi_p2 = _rsi_approx(close, p2_idx)
                bearish_div = rsi_p2 < rsi_p1  # RSI lower at second peak

                if bearish_div:
                    div_strength = min((rsi_p1 - rsi_p2) / 20.0, 1.0)
                    confidence   = int(65 + div_strength * 20)
                    confidence   = max(65, min(85, confidence))
                    result = {
                        "pattern":     "M_TOP",
                        "signal":      "SELL",
                        "confidence":  confidence,
                        "description": (
                            f"M-Top: two peaks at ${p1_price:.2f}/{p2_price:.2f} "
                            f"both above upper BB; RSI divergence -{rsi_p1-rsi_p2:.1f}pt"
                        ),
                    }

        # ────────────────────────────────────────────────────────────────────
        # HEAD AND SHOULDERS: three peaks, head > shoulders by ≥2%, shoulders within 3%
        # ────────────────────────────────────────────────────────────────────
        if result["pattern"] == "NONE" and len(peak_idxs) >= 3:
            ls_idx  = peak_idxs[-3]  # left shoulder
            h_idx   = peak_idxs[-2]  # head
            rs_idx  = peak_idxs[-1]  # right shoulder
            ls_price = close[ls_idx]
            h_price  = close[h_idx]
            rs_price = close[rs_idx]

            # Head at least 2% above both shoulders
            head_above_ls = (h_price - ls_price) / max(ls_price, 1e-8) >= 0.02
            head_above_rs = (h_price - rs_price) / max(rs_price, 1e-8) >= 0.02

            # Shoulders within 3% of each other
            shoulder_diff = abs(ls_price - rs_price) / max(ls_price, rs_price)

            if head_above_ls and head_above_rs and shoulder_diff <= 0.03:
                # Confidence based on symmetry: more symmetric → higher confidence
                symmetry_score = 1.0 - (shoulder_diff / 0.03)  # 0–1
                confidence = int(60 + symmetry_score * 15)      # 60–75
                confidence = max(60, min(75, confidence))
                result = {
                    "pattern":     "HEAD_SHOULDERS",
                    "signal":      "SELL",
                    "confidence":  confidence,
                    "description": (
                        f"Head & Shoulders: shoulders ${ls_price:.2f}/{rs_price:.2f}, "
                        f"head ${h_price:.2f} ({((h_price/ls_price)-1)*100:.1f}% above LS)"
                    ),
                }

        return result

    except Exception:
        return {"pattern": "NONE", "signal": "NEUTRAL", "confidence": 0,
                "description": "No Bollinger Band pattern detected"}
