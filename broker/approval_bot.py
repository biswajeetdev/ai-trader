#!/usr/bin/env python3
"""
approval_bot.py — Telegram approval bot for pending trades.
Run alongside trader.py: python3 broker/approval_bot.py

Long-polls Telegram for callback_query events (✅ Execute / ❌ Skip buttons).
Matches trade_id from callback_data against pending_trades.json.
Executes approved trades via Alpaca (US stocks) or logs for manual execution (India).
"""

import json
import time
import threading
import requests
from datetime import datetime
from pathlib import Path

from broker.approval_queue import get_trade, mark_done, get_expired
from broker.telegram_notifier import edit_approval_message, send_error_alert
from broker.alpaca_exec import execute_alpaca_trade

DIR = Path(__file__).parent.parent


def _load_cfg() -> dict:
    return json.loads((DIR / "config.json").read_text())


def _answer_callback(token: str, callback_query_id: str, text: str = "") -> None:
    """Answer a callback query to remove Telegram's loading spinner."""
    url = f"https://api.telegram.org/bot{token}/answerCallbackQuery"
    try:
        requests.post(url, json={"callback_query_id": callback_query_id, "text": text},
                      timeout=10)
    except Exception:
        pass


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _execute_trade(cfg: dict, trade: dict) -> str:
    """Execute trade and return a result summary string."""
    market = trade.get("market", "us-stock")
    symbol = trade["symbol"]
    action = trade["action"]
    qty    = trade["qty"]
    price  = trade.get("price")
    atr    = trade.get("atr")
    reason = trade.get("reason", "")

    if market in ("us-stock", "crypto"):
        result = execute_alpaca_trade(cfg, symbol, market, action, qty, reason,
                                      price=price, atr=atr)
        if "error" in result:
            return f"ERROR: {result['error']}"
        if "skipped" in result:
            return f"Skipped: {result['skipped']}"
        order_id = result.get("alpaca_order_id", "")[:8]
        status   = result.get("status", "submitted")
        return f"Alpaca order {order_id} — {status}"
    else:
        # India / Zerodha — manual
        print(f"[{_ts()}] MANUAL {action} {symbol} qty={qty} — check Zerodha")
        return "Manual execution needed for India — check Zerodha"


def _handle_callback(cfg: dict, callback_query: dict) -> None:
    """Process a single callback_query from Telegram."""
    token          = cfg.get("telegram_bot_token", "")
    cq_id          = callback_query["id"]
    callback_data  = callback_query.get("data", "")
    chat_id        = str(callback_query["message"]["chat"]["id"])
    message_id     = callback_query["message"]["message_id"]

    if ":" not in callback_data:
        _answer_callback(token, cq_id, "Unknown action")
        return

    action_type, trade_id = callback_data.split(":", 1)
    trade = get_trade(trade_id)

    if trade is None or trade["status"] != "pending":
        _answer_callback(token, cq_id, "Already processed")
        return

    msg_chat_id = trade.get("chat_id") or chat_id

    if action_type == "approve":
        _answer_callback(token, cq_id, "Executing trade...")
        symbol = trade["symbol"]
        qty    = trade["qty"]
        action = trade["action"]
        print(f"[{_ts()}] APPROVE {action} {symbol} qty={qty}")

        try:
            result_str = _execute_trade(cfg, trade)
            mark_done(trade_id, "approved", {"result": result_str})
            edit_approval_message(cfg, msg_chat_id, message_id,
                                  f"✅ <b>Executed</b>: {action} {symbol} x{qty}\n{result_str}")
            print(f"[{_ts()}] DONE    {result_str}")
        except Exception as e:
            err = str(e)[:120]
            mark_done(trade_id, "approved", {"error": err})
            edit_approval_message(cfg, msg_chat_id, message_id,
                                  f"⚠️ <b>Execution failed</b>: {symbol}\n{err}")
            print(f"[{_ts()}] ERROR   {err}")
            send_error_alert(cfg, f"Trade execution failed for {trade_id}: {err}")

    elif action_type == "skip":
        _answer_callback(token, cq_id, "Trade skipped")
        symbol = trade["symbol"]
        action = trade["action"]
        mark_done(trade_id, "skipped")
        edit_approval_message(cfg, msg_chat_id, message_id,
                              f"⏭ <b>Skipped</b>: {action} {symbol}")
        print(f"[{_ts()}] SKIP    {action} {symbol} {trade_id}")

    else:
        _answer_callback(token, cq_id, "Unknown action")


def expire_loop(cfg: dict) -> None:
    """Background thread: expire pending trades older than 15 min every 60 s."""
    while True:
        time.sleep(60)
        try:
            cfg = _load_cfg()  # reload in case config changed
            for trade in get_expired():
                tid     = trade["trade_id"]
                chat_id = trade.get("chat_id", "")
                msg_id  = trade.get("message_id")
                mark_done(tid, "expired")
                if msg_id:
                    edit_approval_message(cfg, chat_id, msg_id,
                                          "⏰ <b>Expired</b> — not executed")
                print(f"[{_ts()}] EXPIRE  {trade['action']} {trade['symbol']} {tid}")
        except Exception as e:
            print(f"[{_ts()}] expire_loop error: {e}")


def main() -> None:
    cfg   = _load_cfg()
    token = cfg.get("telegram_bot_token", "")
    if not token:
        print("ERROR: telegram_bot_token missing from config.json")
        return

    print(f"[{_ts()}] approval_bot started — long-polling Telegram")

    # Start expiry background thread
    t = threading.Thread(target=expire_loop, args=(cfg,), daemon=True)
    t.start()

    offset = 0
    url    = f"https://api.telegram.org/bot{token}/getUpdates"

    while True:
        try:
            cfg = _load_cfg()  # pick up any token/config changes
            params = {"timeout": 25, "offset": offset, "allowed_updates": ["callback_query"]}
            resp = requests.get(url, params=params, timeout=30)
            if not resp.ok:
                print(f"[{_ts()}] Telegram error {resp.status_code}, retrying in 10s")
                time.sleep(10)
                continue

            updates = resp.json().get("result", [])
            for update in updates:
                offset = update["update_id"] + 1
                cq = update.get("callback_query")
                if cq:
                    try:
                        _handle_callback(cfg, cq)
                    except Exception as e:
                        print(f"[{_ts()}] callback error: {e}")

        except requests.exceptions.RequestException as e:
            print(f"[{_ts()}] Network error: {e} — retrying in 10s")
            time.sleep(10)
        except Exception as e:
            print(f"[{_ts()}] Unexpected error: {e} — retrying in 5s")
            time.sleep(5)


if __name__ == "__main__":
    main()
