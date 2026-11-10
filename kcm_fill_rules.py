"""
KCM trigger-specific fill rules and hybrid execution backtest.

Signal is always decided on the signal asset close (QQQ/NDX).
Fill timing depends on *why* weight changed:

  Price-level triggers (observable intraday):
    SMA200 진입/탈출, SMA50 진입/탈출  → intraday touch
  Close-confirmed triggers (RSI / VIX):
    RSI·VIX 진입/탈출             → LOC (same close)
    (다음날 시가도 허용; 기본은 LOC)

레짐 청산은 항상 SMA200-1% AND SMA50 동시 미만.
다만 *어느 레벨이 마지막에 깨졌는지*로 체결가를 나눔:
  - 이미 SMA200-1% 아래 + SMA50 위 → SMA50 이탈 = SMA50탈출 (장중)
  - 이미 SMA50 아래 + SMA200-1% 위 → SMA200-1% 이탈 = SMA200탈출 (장중)
  - 하루만에 둘 다 깨짐 → SMA동시탈출 (장중, 더 낮은 레벨=두 번째 터치)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from notify_kcm_telegram import BUF, RSI_DIP, VIX_TH, rsi_sma, vt_scale

COST_BPS_DEFAULT = 9.0
LEV_DEFAULT = 3.0

# Fill method per trigger tag
FILL_RULES: dict[str, str] = {
    "SMA200진입": "intraday",
    "SMA50진입": "intraday",
    "SMA200탈출": "intraday",
    "SMA50탈출": "intraday",
    "SMA동시탈출": "intraday",
    "RSI진입": "loc",
    "VIX진입": "loc",
    "RSI탈출": "loc",
    "VIX탈출": "loc",
}


@dataclass
class FillEvent:
    date: pd.Timestamp
    side: str  # BUY / SELL
    reason: str
    fill: str  # intraday / loc / next_open
    w_from: float
    w_to: float


def classify_transitions(
    close: pd.Series,
    vix: pd.Series,
    *,
    rsi_th: float = RSI_DIP,
    vix_th: float = VIX_TH,
    buf: float = BUF,
) -> tuple[pd.Series, pd.Series, list[FillEvent]]:
    """Return (weight 0/1, regime_on, fill events on change days)."""
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    vs200 = (close / sma200 - 1).fillna(0)
    vs50 = (close / sma50 - 1).fillna(0)
    rsi = rsi_sma(close).ffill().fillna(50)
    vv = vix.reindex(close.index).ffill()
    vx10 = vv.rolling(10).mean()

    n = len(close)
    w = np.zeros(n)
    inp_a = np.zeros(n, dtype=bool)
    events: list[FillEvent] = []

    inp = bool(vs200.iloc[0] >= 0)
    w[0] = 1.0 if inp else 0.0
    inp_a[0] = inp

    for i in range(1, n):
        has = pd.notna(vv.iloc[i]) and pd.notna(vx10.iloc[i])
        vr = bool(has and vv.iloc[i] > vix_th and vv.iloc[i] < vx10.iloc[i])
        dip = float(rsi.iloc[i]) < rsi_th
        prev_w = w[i - 1]
        reason = ""

        if inp:
            if vs200.iloc[i] < -buf and vs50.iloc[i] < 0:
                inp = False
                if vr or dip:
                    w[i] = 1.0
                    # regime off but immediate temp hold — no cash exit event;
                    # weight stays 1 (ON→temp). Tag only if we need audit.
                    reason = "temp_after_OFF"
                else:
                    w[i] = 0.0
                    prev_lo200 = bool(vs200.iloc[i - 1] < -buf)
                    prev_lo50 = bool(vs50.iloc[i - 1] < 0)
                    if prev_lo200 and not prev_lo50:
                        reason = "SMA50탈출"  # already <SMA200-1%, lost SMA50
                    elif prev_lo50 and not prev_lo200:
                        reason = "SMA200탈출"  # already <SMA50, lost SMA200-1%
                    else:
                        reason = "SMA동시탈출"  # both broken same day
            else:
                w[i] = 1.0
        else:
            if vs200.iloc[i] >= buf or vs50.iloc[i] >= 0:
                inp = True
                w[i] = 1.0
                reason = "SMA200진입" if vs200.iloc[i] >= buf else "SMA50진입"
            elif vr or dip:
                w[i] = 1.0
                if dip and vr:
                    reason = "RSI진입"  # dual: treat as RSI for fill rule (both use loc)
                elif dip:
                    reason = "RSI진입"
                else:
                    reason = "VIX진입"
            else:
                w[i] = 0.0
                if prev_w >= 0.5:
                    prev_dip = float(rsi.iloc[i - 1]) < rsi_th
                    prev_has = pd.notna(vv.iloc[i - 1]) and pd.notna(vx10.iloc[i - 1])
                    prev_vr = bool(
                        prev_has
                        and vv.iloc[i - 1] > vix_th
                        and vv.iloc[i - 1] < vx10.iloc[i - 1]
                    )
                    if prev_vr and not prev_dip:
                        reason = "VIX탈출"
                    else:
                        reason = "RSI탈출"

        inp_a[i] = inp
        if abs(w[i] - prev_w) > 0.5 and reason in FILL_RULES:
            events.append(
                FillEvent(
                    date=close.index[i],
                    side="BUY" if w[i] > prev_w else "SELL",
                    reason=reason,
                    fill=FILL_RULES[reason],
                    w_from=float(prev_w),
                    w_to=float(w[i]),
                )
            )

    return (
        pd.Series(w, index=close.index, name="w"),
        pd.Series(inp_a, index=close.index, name="inp"),
        events,
    )


def _buy_touch(sig_o, sig_h, sma50, sma200, i, reason: str) -> float:
    o, h = float(sig_o.iloc[i]), float(sig_h.iloc[i])
    thr200 = float(sma200.iloc[i]) * (1 + BUF)
    thr50 = float(sma50.iloc[i])
    if reason == "SMA200진입" and h >= thr200:
        return max(o, thr200)
    if reason == "SMA50진입" and h >= thr50:
        return max(o, thr50)
    # fallback
    cands = []
    if h >= thr200:
        cands.append(max(o, thr200))
    if h >= thr50:
        cands.append(max(o, thr50))
    return min(cands) if cands else o


def _sell_touch(sig_o, sig_l, sma50, sma200, i, reason: str = "") -> float:
    o, l = float(sig_o.iloc[i]), float(sig_l.iloc[i])
    thr200 = float(sma200.iloc[i]) * (1 - BUF)
    thr50 = float(sma50.iloc[i])
    if reason == "SMA50탈출":
        level = thr50
    elif reason == "SMA200탈출":
        level = thr200
    else:
        # SMA동시탈출 / fallback: second (lower) level completes dual exit
        level = min(thr200, thr50)
    if l <= level:
        return o if o <= level else level
    return o


def hybrid_backtest(
    sig_close: pd.Series,
    sig_open: pd.Series,
    sig_high: pd.Series,
    sig_low: pd.Series,
    veh_close: pd.Series,
    veh_open: pd.Series,
    vix: pd.Series,
    *,
    start: str = "2000-01-01",
    end: str | None = None,
    lev: float = LEV_DEFAULT,
    cost_bps: float = COST_BPS_DEFAULT,
    vt_target: float | None = None,
    fill_mode: str = "hybrid",
) -> dict:
    """
    fill_mode:
      hybrid     — trigger-specific rules (FILL_RULES)
      next_open  — all flips next open
      loc        — all flips same close
      intraday   — all flips intraday touch/open proxy
    """
    w_bin, _, events = classify_transitions(sig_close, vix)
    if vt_target is not None:
        # sticky VT: freeze scale on entry until exit
        sc = vt_scale(sig_close, target=vt_target)
        w = pd.Series(0.0, index=w_bin.index)
        held = 0.0
        for i in range(len(w_bin)):
            on = float(w_bin.iloc[i]) > 0.5
            prev = float(w_bin.iloc[i - 1]) > 0.5 if i else False
            if on and not prev:
                held = float(sc.iloc[i])
            w.iloc[i] = held if on else 0.0
            if not on:
                held = 0.0
    else:
        w = w_bin

    event_by_date = {e.date: e for e in events}
    sma50 = sig_close.rolling(50).mean()
    sma200 = sig_close.rolling(200).mean()

    sub = sig_close.loc[start:end].index
    # drop trailing incomplete vehicle bars
    sub = sub[np.isfinite(veh_close.reindex(sub)) & np.isfinite(veh_open.reindex(sub))]
    ww = w.reindex(sub).fillna(0.0).values
    vc = veh_close.reindex(sub).astype(float).values
    vo = veh_open.reindex(sub).astype(float).values
    sc_ = sig_close.reindex(sub).astype(float).values
    cost = cost_bps / 1e4

    # Resolve effective weight path under fill delays
    # For each day i, determine weight used from prior close → today close segments.
    # We simulate chronologically with pending next_open orders.

    nav = 1.0
    pos = 0.0  # current position weight
    pending: FillEvent | None = None  # next_open from prior signal
    trades = 0
    curve = []
    locmap = {d: int(sig_close.index.get_loc(d)) for d in sub}
    # When True, today's open→close should use open-to-close (pending was filled at open)
    opened_today = False

    def trade_to(target: float):
        nonlocal nav, pos, trades
        if abs(target - pos) > 1e-12:
            trades += 1
            nav *= 1 - abs(target - pos) * cost
            pos = target

    for i in range(len(sub)):
        d = sub[i]
        fi = locmap[d]
        opened_today = False

        # 1) fill pending next_open at today's open
        if pending is not None:
            # gap close[i-1]→open[i] with OLD weight still (signal was yesterday)
            if i > 0 and vo[i] and vc[i - 1]:
                nav *= 1 + pos * (vo[i] / vc[i - 1] - 1)
            trade_to(pending.w_to)
            pending = None
            opened_today = True

        if i == 0:
            curve.append(nav)
            continue

        ev = event_by_date.get(d)
        style = None
        if abs(ww[i] - ww[i - 1]) > 0.5:
            if fill_mode == "hybrid" and ev is not None:
                style = ev.fill
            elif fill_mode == "hybrid":
                style = "next_open"
            else:
                style = fill_mode

        # If we already filled a pending order at the open, finish the regular
        # open→close with the post-fill weight when there is no same-day flip.
        if opened_today and style is None:
            if vo[i]:
                nav *= 1 + pos * (vc[i] / vo[i] - 1)
            if not np.isfinite(nav) or nav <= 0:
                nav = 1e-15
            curve.append(nav)
            continue

        if style is None:
            r = vc[i] / vc[i - 1] - 1 if vc[i - 1] else 0.0
            nav *= 1 + pos * r
        elif style == "next_open":
            # close-to-close with current pos; execute tomorrow's open
            if opened_today and vo[i]:
                nav *= 1 + pos * (vc[i] / vo[i] - 1)
            else:
                r = vc[i] / vc[i - 1] - 1 if vc[i - 1] else 0.0
                nav *= 1 + pos * r
            pending = FillEvent(
                d,
                "BUY" if ww[i] > pos else "SELL",
                ev.reason if ev else "?",
                "next_open",
                pos,
                float(ww[i]),
            )
        elif style == "loc":
            if opened_today and vo[i]:
                nav *= 1 + pos * (vc[i] / vo[i] - 1)
            else:
                r = vc[i] / vc[i - 1] - 1 if vc[i - 1] else 0.0
                nav *= 1 + pos * r
            trade_to(float(ww[i]))
        elif style == "intraday":
            reason = ev.reason if ev else ""
            if ww[i] > ww[i - 1]:
                sig_f = _buy_touch(sig_open, sig_high, sma50, sma200, fi, reason)
            else:
                sig_f = _sell_touch(sig_open, sig_low, sma50, sma200, fi, reason)
            v_ref = vo[i] if opened_today else vc[i - 1]
            s_ref = float(sig_open.iloc[fi]) if opened_today else sc_[i - 1]
            if s_ref <= 0 or v_ref <= 0:
                v_fill = vc[i]
            else:
                v_fill = v_ref * (1 + lev * (sig_f / s_ref - 1))
            day = abs(vc[i] / v_ref - 1) + 0.6 if v_ref else 0.6
            v_fill = float(np.clip(v_fill, max(v_ref * (1 - day), 1e-9), v_ref * (1 + day)))
            nav *= 1 + pos * (v_fill / v_ref - 1)
            trade_to(float(ww[i]))
            nav *= 1 + pos * (vc[i] / v_fill - 1)
        else:
            raise ValueError(style)

        if not np.isfinite(nav) or nav <= 0:
            nav = 1e-15
        curve.append(nav)

    s = pd.Series(curve, index=sub)
    yrs = max((sub[-1] - sub[0]).days / 365.25, 1e-9)
    terminal = float(s.iloc[-1])
    cagr = float(terminal ** (1 / yrs) - 1) if terminal > 0 else float("nan")
    mdd = float((s / s.cummax() - 1).min())
    return {
        "nav": s,
        "cagr": cagr,
        "total": terminal - 1,
        "mdd": mdd,
        "trades": trades,
        "events": events,
        "weights": w.reindex(sub),
    }
