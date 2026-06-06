"""leveraged_etf.py — TQQQ/SQQQ rotation based on BotScore regime.
Bull regime (BotScore > 65) → TQQQ. Bear regime (BotScore < 35) → SQQQ.
Neutral → hold cash (no position).
"""


def get_leveraged_etf_signal(botscore: int, vix: float,
                              current_holding: str | None = None) -> dict:
    # current_holding: pass "TQQQ" or "SQQQ" if already positioned, enables flip logic

    bull = botscore > 65 and vix < 20
    bear = botscore < 35 and vix > 25

    if bull:
        if current_holding == "SQQQ":
            # regime flipped bull — exit bear ETF before buying bull ETF
            return {
                "symbol":     "SQQQ",
                "action":     "SELL",
                "reason":     f"Regime flipped bull (BotScore {botscore}, VIX {vix:.1f}) — exit SQQQ first",
                "confidence": 75,
            }
        return {
            "symbol":     "TQQQ",
            "action":     "BUY",
            "reason":     f"Bull regime: BotScore {botscore} > 65 and VIX {vix:.1f} < 20",
            "confidence": 75,
        }

    if bear:
        if current_holding == "TQQQ":
            # regime flipped bear — exit bull ETF before buying bear ETF
            return {
                "symbol":     "TQQQ",
                "action":     "SELL",
                "reason":     f"Regime flipped bear (BotScore {botscore}, VIX {vix:.1f}) — exit TQQQ first",
                "confidence": 75,
            }
        return {
            "symbol":     "SQQQ",
            "action":     "BUY",
            "reason":     f"Bear regime: BotScore {botscore} < 35 and VIX {vix:.1f} > 25",
            "confidence": 75,
        }

    # Neutral — neither condition met, hold cash
    return {
        "symbol":     None,
        "action":     "HOLD",
        "reason":     f"Neutral regime (BotScore {botscore}, VIX {vix:.1f}) — hold cash",
        "confidence": 0,
    }
