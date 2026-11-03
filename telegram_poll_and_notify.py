"""
Poll Telegram for an on-demand request, then send KCM signal.

Triggers (any of these in a message from TELEGRAM_CHAT_ID):
  /signal  /now  /kcm  /start  신호  시그널

Env:
  TELEGRAM_BOT_TOKEN
  TELEGRAM_CHAT_ID
"""
from __future__ import annotations

import os
import re
import sys

import requests

from notify_kcm_telegram import build_message, send_telegram

TRIGGER = re.compile(
    r"(^|\s)(/signal|/now|/kcm|/start|신호|시그널)(@\w+)?(\s|$)",
    re.IGNORECASE,
)


def main() -> int:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = str(os.environ.get("TELEGRAM_CHAT_ID", "")).strip()
    if not token or not chat_id:
        print("Missing TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID", file=sys.stderr)
        return 2

    # Ensure polling mode (clear any webhook)
    requests.get(
        f"https://api.telegram.org/bot{token}/deleteWebhook",
        params={"drop_pending_updates": "false"},
        timeout=30,
    ).raise_for_status()

    r = requests.get(
        f"https://api.telegram.org/bot{token}/getUpdates",
        params={"timeout": 0, "limit": 50},
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    if not data.get("ok"):
        raise RuntimeError(data)

    updates = data.get("result") or []
    wanted = False
    max_id = 0
    for u in updates:
        max_id = max(max_id, int(u.get("update_id", 0)))
        msg = u.get("message") or u.get("edited_message") or {}
        chat = msg.get("chat") or {}
        if str(chat.get("id", "")) != chat_id:
            continue
        text = (msg.get("text") or "").strip()
        if TRIGGER.search(text):
            wanted = True
            print(f"Trigger from chat {chat_id}: {text!r}")

    if max_id:
        # Acknowledge so the same messages are not reprocessed
        requests.get(
            f"https://api.telegram.org/bot{token}/getUpdates",
            params={"offset": max_id + 1, "timeout": 0},
            timeout=60,
        ).raise_for_status()
        print(f"Acked updates through {max_id}")

    if not wanted:
        print("No trigger message — exit without sending.")
        return 0

    msg = build_message()
    print(msg)
    send_telegram(msg, token, chat_id)
    print("Sent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
