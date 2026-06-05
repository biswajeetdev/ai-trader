"""
rl_sizer.py — Contextual bandit position sizing.

Uses Thompson sampling (Beta distribution) to learn which size multipliers
work best in which market regimes. Gets smarter after every closed trade.

State: (regime, vix_bucket, confidence_bucket) → 3 × 3 × 3 = 27 contexts
Actions: [0.5, 0.75, 1.0, 1.25, 1.5] × base_size

After 10+ attributed trades the bandit starts outperforming fixed sizing.
"""

import json, random
from pathlib import Path

BANDIT_FILE = Path(__file__).parent / "bandit_state.json"
ACTIONS     = [0.5, 0.75, 1.0, 1.25, 1.5]
MIN_TRADES  = 5  # cold-start threshold per context


def _encode_state(regime: str, vix: float, confidence: int) -> str:
    """Returns a string key like 'BOT_DRIVEN|CALM|HIGH'."""
    vix_bucket  = "CALM" if vix < 15 else "NORMAL" if vix < 25 else "ELEVATED"
    conf_bucket = "LOW"  if confidence < 60 else "MED" if confidence < 80 else "HIGH"
    return f"{regime}|{vix_bucket}|{conf_bucket}"


def _load_state() -> dict:
    if not BANDIT_FILE.exists():
        return {}
    try:
        return json.loads(BANDIT_FILE.read_text())
    except Exception:
        return {}


def _save_state(state: dict) -> None:
    try:
        BANDIT_FILE.write_text(json.dumps(state, indent=2))
    except Exception:
        pass


def _default_arms() -> dict:
    return {str(a): {"alpha": 1, "beta": 1} for a in ACTIONS}


def _attributed_trades(arms: dict) -> int:
    """Total attributed trades for a context (alpha+beta-2 per arm, summed)."""
    return sum(v["alpha"] + v["beta"] - 2 for v in arms.values())


def get_size_multiplier(regime: str, vix: float, confidence: int) -> float:
    """Thompson sampling: return the best size multiplier for the current state."""
    state = _load_state()
    key   = _encode_state(regime, vix, confidence)
    arms  = state.get(key, _default_arms())

    if _attributed_trades(arms) < MIN_TRADES:
        return 1.0  # cold-start: safe default until enough data

    samples = {a: random.betavariate(arms[str(a)]["alpha"], arms[str(a)]["beta"])
               for a in ACTIONS}
    return max(samples, key=samples.get)


def update_bandit(regime: str, vix: float, confidence: int,
                  multiplier: float, win: bool) -> None:
    """Called after trade closes — updates Beta params with outcome."""
    state = _load_state()
    key   = _encode_state(regime, vix, confidence)
    if key not in state:
        state[key] = _default_arms()
    arm = state[key].get(str(multiplier))
    if arm is None:
        return
    if win:
        arm["alpha"] += 1
    else:
        arm["beta"] += 1
    _save_state(state)


def get_bandit_stats() -> str:
    """Return a human-readable summary of bandit performance."""
    state = _load_state()
    if not state:
        return "Bandit: no attributed trades yet."

    total_trades = 0
    best_ctx, best_action, best_win_rate = "", 1.0, 0.0

    for ctx, arms in state.items():
        for action, params in arms.items():
            trades = params["alpha"] + params["beta"] - 2
            if trades == 0:
                continue
            total_trades += trades
            wr = params["alpha"] / (params["alpha"] + params["beta"])
            if wr > best_win_rate:
                best_win_rate, best_ctx, best_action = wr, ctx, action

    if total_trades == 0:
        return "Bandit: no attributed trades yet."
    return (f"Bandit: {total_trades} attributed trades. "
            f"Best action {best_ctx} → {best_action}× ({int(best_win_rate * 100)}% win)")
