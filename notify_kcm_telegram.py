"""
Daily KCM signal → Telegram.

Reports:
  1) SMH (USD) — signal from SMH itself
  2) TQQQ      — signal from QQQ (not SMH)

Signal at T close → act at T+1 open.
Run after US cash close (e.g. 06:30 KST next calendar day).

Env:
  TELEGRAM_BOT_TOKEN
  TELEGRAM_CHAT_ID

Usage:
  python notify_kcm_telegram.py           # send Telegram
  python notify_kcm_telegram.py --dry-run # print only
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import yfinance as yf

BUF = 0.01


def rsi_wilder(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def kcm_weights(close: pd.Series, vix: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return (weight 0/1, in_position bool) aligned to close.index."""
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    vs200 = (close / sma200 - 1).fillna(0)
    vs50 = (close / sma50 - 1).fillna(0)
    rsi = rsi_wilder(close).ffill().fillna(50)
    vx = vix.reindex(close.index, method="ffill")
    vx10 = vx.rolling(10).mean()

    n = len(close)
    w = np.zeros(n)
    inp = bool(vs200.iloc[0] >= 0)
    states = np.zeros(n, dtype=bool)
    w[0] = 1.0 if inp else 0.0
    states[0] = inp

    for i in range(1, n):
        has = pd.notna(vx.iloc[i]) and pd.notna(vx10.iloc[i])
        vr = bool(has and vx.iloc[i] > 35 and vx.iloc[i] < vx10.iloc[i])
        if inp:
            if vs200.iloc[i] < -BUF and vs50.iloc[i] < 0:
                inp = False
                w[i] = 1.0 if (vr or rsi.iloc[i] < 30) else 0.0
            else:
                w[i] = 1.0
        else:
            if vs200.iloc[i] >= BUF or vs50.iloc[i] >= 0:
                inp = True
                w[i] = 1.0
            elif vr or float(rsi.iloc[i]) < 30:
                w[i] = 1.0
            else:
                w[i] = 0.0
        states[i] = inp

    return (
        pd.Series(w, index=close.index, name="w"),
        pd.Series(states, index=close.index, name="inp"),
    )


def action_text(w_prev: float, w_now: float) -> str:
    """What to do at next open given today's close signal w_now vs yesterday."""
    if w_now >= 1.0 and w_prev < 1.0:
        return "BUY 100% (내일 시가)"
    if w_now <= 0.0 and w_prev > 0.0:
        return "SELL 0% (내일 시가)"
    if w_now >= 1.0:
        return "HOLD 100%"
    return "STAY CASH 0%"


def reason_line(inp: bool, vs200: float, vs50: float, rsi: float, vix: float, vix10: float, w: float) -> str:
    bits = []
    if inp:
        bits.append("SMA레짐 ON")
    else:
        bits.append("SMA레짐 OFF")
        if vs200 < -BUF and vs50 < 0:
            bits.append("청산조건충족")
        if rsi < 30:
            bits.append(f"RSI딥({rsi:.1f})")
        if vix > 35 and vix < vix10:
            bits.append(f"VIXre({vix:.1f})")
    bits.append(f"vs200={vs200:+.1%}")
    bits.append(f"vs50={vs50:+.1%}")
    bits.append(f"RSI={rsi:.1f}")
    bits.append(f"VIX={vix:.1f}")
    bits.append(f"signal={w:.0%}")
    return " | ".join(bits)


def load_close(ticker: str, start: str = "2018-01-01") -> pd.DataFrame:
    d = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    if d.empty:
        raise RuntimeError(f"No data for {ticker}")
    return d


def _snapshot(close: pd.Series, vix: pd.Series, label: str, px_series: pd.Series | None = None):
    """Build one vehicle block from signal series `close` (and optional display price)."""
    w, inp = kcm_weights(close, vix)
    i = -1
    w_now = float(w.iloc[i])
    w_prev = float(w.iloc[i - 1]) if len(w) >= 2 else w_now
    row_date = w.index[i]

    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    vs200 = float(close.iloc[i] / sma200.iloc[i] - 1)
    vs50 = float(close.iloc[i] / sma50.iloc[i] - 1)
    rsi = float(rsi_wilder(close).iloc[i])
    vx = float(vix.reindex(close.index, method="ffill").iloc[i])
    vx10 = float(vix.reindex(close.index, method="ffill").rolling(10).mean().iloc[i])
    in_pos = bool(inp.iloc[i])

    show = px_series.reindex(close.index, method="ffill") if px_series is not None else close
    px = float(show.iloc[i])

    return {
        "date": row_date,
        "label": label,
        "px": px,
        "act": action_text(w_prev, w_now),
        "why": reason_line(in_pos, vs200, vs50, rsi, vx, vx10, w_now),
        "w": w_now,
    }


def build_message() -> str:
    smh = load_close("SMH")
    qqq = load_close("QQQ")
    tqqq = load_close("TQQQ")
    vix_df = load_close("^VIX")

    smh_c = smh["Close"].dropna()
    qqq_c = qqq["Close"].dropna()
    tqqq_c = tqqq["Close"].dropna()
    vix_c = vix_df["Close"].dropna()

    smh_snap = _snapshot(smh_c, vix_c, "SMH USD")
    tqqq_snap = _snapshot(qqq_c, vix_c, "TQQQ (QQQ 신호)", px_series=tqqq_c)

    # Prefer QQQ/SMH last date; show both if they match
    d0 = smh_snap["date"].date()
    d1 = tqqq_snap["date"].date()
    date_line = f"{d0}" if d0 == d1 else f"SMH {d0} / QQQ {d1}"

    lines = [
        f"KCM Daily  ({date_line})",
        "기준: 각 신호 종가 → 다음날 시가 체결",
        "",
        f"[SMH USD]  신호=SMH  ${smh_snap['px']:.2f}",
        f"  → {smh_snap['act']}",
        f"  {smh_snap['why']}",
        "",
        f"[TQQQ]  신호=QQQ  ${tqqq_snap['px']:.2f}",
        f"  → {tqqq_snap['act']}",
        f"  {tqqq_snap['why']}",
        "",
        "규칙: SMA200±1% + SMA50 | RSI<30 or VIX>35&하락 = 임시100%",
    ]
    return "\n".join(lines)


def send_telegram(text: str, token: str, chat_id: str) -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    r = requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not data.get("ok"):
        raise RuntimeError(f"Telegram API error: {data}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    try:
        msg = build_message()
    except Exception as e:
        msg = f"KCM Daily FAILED: {e}"
        print(msg, file=sys.stderr)
        if not args.dry_run:
            token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
            chat = os.environ.get("TELEGRAM_CHAT_ID", "")
            if token and chat:
                send_telegram(msg, token, chat)
        return 1

    print(msg)
    if args.dry_run:
        return 0

    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat:
        print(
            "Missing TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID — dry message printed only.",
            file=sys.stderr,
        )
        return 2

    send_telegram(msg, token, chat)
    print(f"\nSent at {datetime.now(timezone.utc).isoformat()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
