#!/usr/bin/env python3
"""
approval_bot.py — Telegram approval daemon for AI-Trader India phase.

Runs 24/7 (launchd or screen). Polls Telegram for button presses on
pending trade approvals. On ✅ Execute: runs execute_india_trade().
On ❌ Skip or 15-min timeout: marks trade skipped/expired.

Run:
  python3 approval_bot.py

launchd plist: ~/Library/LaunchAgents/com.aitrader.approvalbot.plist
"""

import json, os, sys, time, threading, requests
from pathlib import Path

DIR = Path(__file__).parent
sys.path.insert(0, str(DIR))

from broker.approval_queue import (
    get_trade, mark_done, get_expired, get_pending, cleanup_old,
)
from broker.zerodha_exec import execute_india_trade
from broker.telegram_notifier import edit_approval_message

CONFIG = DIR / "config.json"
POLL_TIMEOUT = 30   # long-poll seconds
TIMEOUT_CHECK = 60  # how often to scan for expired trades


def load_cfg() -> dict:
    return json.loads(CONFIG.read_text()) if CONFIG.exists() else {}


def _get_updates(token: str, offset: int) -> list:
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    try:
        r = requests.get(url, params={"timeout": POLL_TIMEOUT,
                                       "offset": offset,
                                       "allowed_updates": ["callback_query"]},
                         timeout=POLL_TIMEOUT + 5)
        if r.ok:
            return r.json().get("result", [])
    except Exception:
        pass
    return []


def _answer_callback(token: str, callback_id: str, text: str = ""):
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/answerCallbackQuery",
            json={"callback_query_id": callback_id, "text": text},
            timeout=5,
        )
    except Exception:
        pass


def handle_callback(cfg: dict, update: dict):
    cq = update.get("callback_query", {})
    callback_id = cq.get("id", "")
    data        = cq.get("data", "")
    chat_id     = str(cq.get("message", {}).get("chat", {}).get("id", ""))
    message_id  = cq.get("message", {}).get("message_id")
    token       = cfg.get("telegram_bot_token", "")

    if ":" not in data:
        return

    action_type, trade_id = data.split(":", 1)
    trade = get_trade(trade_id)

    if trade is None:
        _answer_callback(token, callback_id, "Trade not found (already resolved?)")
        return

    if trade["status"] != "pending":
        _answer_callback(token, callback_id,
                         f"Already {trade['status']} — no action taken.")
        return

    sym    = trade["symbol"]
    action = trade["action"]
    qty    = trade["qty"]
    price  = trade["price"]
    atr    = trade.get("atr", 0)
    conf   = trade.get("conf", 0)

    if action_type == "approve":
        _answer_callback(token, callback_id, f"Executing {action} {sym}...")
        try:
            result = execute_india_trade(cfg, sym, action, qty,
                                          trade.get("reason", ""),
                                          price, atr)
            mark_done(trade_id, "approved", result)
            result_line = f"✅ <b>EXECUTED</b> {action} {qty}x {sym} @ ₹{price:,.2f}"
        except Exception as e:
            mark_done(trade_id, "approved", {"error": str(e)})
            result_line = f"⚠️ Execution error: {str(e)[:80]}"

        edit_approval_message(cfg, chat_id, message_id,
                               f"{result_line}\nConf: {conf}%")

    elif action_type == "skip":
        _answer_callback(token, callback_id, f"Skipped {sym}.")
        mark_done(trade_id, "skipped")
        edit_approval_message(cfg, chat_id, message_id,
                               f"❌ <b>SKIPPED</b> {action} {sym}")


def timeout_loop():
    """Background thread: expire trades that weren't actioned in time."""
    while True:
        time.sleep(TIMEOUT_CHECK)
        try:
            cfg = load_cfg()
            for trade in get_expired():
                trade_id   = trade["trade_id"]
                sym        = trade["symbol"]
                chat_id    = trade.get("chat_id", "")
                message_id = trade.get("message_id")
                mark_done(trade_id, "expired")
                if message_id:
                    edit_approval_message(cfg, chat_id, message_id,
                                           f"⏱ <b>EXPIRED</b> — {trade['action']} {sym} "
                                           f"(no response in 15 min)")
            cleanup_old()
        except Exception:
            pass


def main():
    print("approval_bot started — polling Telegram...")
    threading.Thread(target=timeout_loop, daemon=True).start()

    offset = 0
    while True:
        cfg   = load_cfg()
        token = cfg.get("telegram_bot_token", "")
        if not token:
            print("[approval_bot] No telegram_bot_token in config.json. Retrying in 60s.")
            time.sleep(60)
            continue

        updates = _get_updates(token, offset)
        for upd in updates:
            offset = upd["update_id"] + 1
            if "callback_query" in upd:
                try:
                    handle_callback(cfg, upd)
                except Exception as e:
                    print(f"[approval_bot] callback error: {e}")

        # Reload config each long-poll cycle (catches token rotation)
        time.sleep(0.1)


if __name__ == "__main__":
    main()
