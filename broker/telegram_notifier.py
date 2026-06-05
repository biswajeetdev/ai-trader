"""
telegram_notifier.py — Telegram trade alerts for AI-Trader

Setup:
  1. Message @BotFather on Telegram → /newbot → copy token
  2. Message @userinfobot → copy your chat_id
  3. Add to config.json: "telegram_bot_token": "...", "telegram_chat_id": "..."

All functions are fire-and-forget (send in background thread).
If token/chat_id missing → silent no-op.
"""

import threading
import requests
from datetime import date, datetime


def _send(cfg: dict, text: str) -> None:
    """Internal worker — posts message to Telegram Bot API. Never raises."""
    token = cfg.get("telegram_bot_token", "")
    chat_id = cfg.get("telegram_chat_id", "")
    if not token or not chat_id:
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception:
        pass  # Never crash the bot


def _fire(cfg: dict, text: str) -> None:
    """Dispatch _send in a non-daemon thread so it completes before process exit."""
    t = threading.Thread(target=_send, args=(cfg, text), daemon=False)
    t.start()


def send_trade_alert(
    cfg: dict,
    action: str,
    symbol: str,
    qty,
    price,
    reason: str,
    confidence,
    alpaca_order_id: str = "",
) -> None:
    """Fire-and-forget trade alert."""
    action_icon = {"BUY": "BUY\U0001f7e2", "SELL": "SELL\U0001f534"}.get(
        action.upper(), f"{action}⚪"
    )
    lines = [
        "\U0001f916 <b>AI-Trader</b>",
        f"{action_icon} {symbol}",
        f"Qty: {qty} @ ${price}",
        f"Conf: {confidence}%",
        f"Reason: {str(reason)[:120]}",
    ]
    if alpaca_order_id:
        lines.append(f"Alpaca: {alpaca_order_id[:8]}")
    _fire(cfg, "\n".join(lines))


def send_daily_summary(
    cfg: dict,
    trades_today: list,
    cash: float,
    portfolio_value: float,
    drawdown_pct: float,
) -> None:
    """Fire-and-forget daily summary."""
    today = date.today().isoformat()
    lines = [
        f"\U0001f4ca <b>Daily Summary — {today}</b>",
        f"Trades: {len(trades_today)}",
        f"Cash: ${cash:,.0f}  Portfolio: ${portfolio_value:,.0f}",
        f"Drawdown: {drawdown_pct:.1f}%",
    ]
    for t in trades_today:
        sym = t.get("symbol", "?")
        act = t.get("action", "?")
        qty = t.get("quantity", "?")
        lines.append(f"  {sym} {act} {qty}")
    _fire(cfg, "\n".join(lines))


def send_run_status(cfg: dict, cash: float, trades_today: list,
                    positions_count: int, sp_open: int = 0) -> None:
    """Per-run heartbeat — fires every run so the user sees the bot is alive."""
    now = datetime.now().strftime("%H:%M IST")
    trade_line = (
        f"{len(trades_today)} trade(s) executed" if trades_today else "No trades — all HOLD"
    )
    lines = [
        f"\U0001f916 <b>AI-Trader</b> | {now}",
        f"\U0001f4b5 Cash: ${cash:,.0f}",
        f"\U0001f4ca {trade_line}",
        f"\U0001f4c1 {positions_count} stock position(s)",
    ]
    if sp_open:
        lines.append(f"\U0001f7e3 {sp_open} short put(s) open")
    _fire(cfg, "\n".join(lines))


def send_error_alert(cfg: dict, message: str) -> None:
    """Fire-and-forget error alert."""
    text = f"⚠️ <b>AI-Trader Error</b>\n{str(message)[:200]}"
    _fire(cfg, text)


def send_approval_request(cfg: dict, trade_id: str, action: str,
                           symbol: str, qty, price: float,
                           conf: int, reason: str) -> int | None:
    """Send approval message with inline keyboard. Returns Telegram message_id."""
    token   = cfg.get("telegram_bot_token", "")
    chat_id = cfg.get("telegram_chat_id", "")
    if not token or not chat_id:
        return None

    icon = "🟢" if action == "BUY" else "🔴"
    text = (
        f"🤖 <b>Trade Signal — Approval Needed</b>\n"
        f"{icon} <b>{action} {symbol}</b>\n"
        f"Qty: {qty} @ ₹{price:,.2f}\n"
        f"Conf: {conf}%\n"
        f"Reason: {str(reason)[:120]}\n"
        f"<i>Expires in 15 min</i>"
    )
    keyboard = {
        "inline_keyboard": [[
            {"text": "✅ Execute", "callback_data": f"approve:{trade_id}"},
            {"text": "❌ Skip",    "callback_data": f"skip:{trade_id}"},
        ]]
    }
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(url, json={
            "chat_id": chat_id, "text": text,
            "parse_mode": "HTML", "reply_markup": keyboard,
        }, timeout=10)
        if r.ok:
            return r.json()["result"]["message_id"]
    except Exception:
        pass
    return None


def edit_approval_message(cfg: dict, chat_id: str, message_id: int,
                           new_text: str) -> None:
    """Edit an existing approval message (called after approve/skip/expire)."""
    token = cfg.get("telegram_bot_token", "")
    if not token or not message_id:
        return
    url = f"https://api.telegram.org/bot{token}/editMessageText"
    try:
        requests.post(url, json={
            "chat_id": chat_id, "message_id": message_id,
            "text": new_text, "parse_mode": "HTML",
        }, timeout=10)
    except Exception:
        pass
