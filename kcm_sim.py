"""
Realistic KCM execution simulator with Korean overseas-stock tax.

Fill per category (sma_fill × temp_fill)
  sma_fill   intraday / loc / next_open  — SMA200·SMA50 레짐 전환
  temp_fill  loc / next_open             — RSI·VIX 임시보유 전환
  (레짐 전환과 임시보유가 겹쳐 실제 매매가 없으면 체결도 없음)

  intraday   SMA 레벨은 실전 자동주문(감시주문)처럼 처리:
               - 레벨 = 전일 종가 기준 SMA (장 시작 전 알 수 있는 값)
               - 진입: OFF 상태에서 고가 >= min(SMA200*(1+BUF), SMA50) → 터치 즉시 체결
               - 청산: ON 상태에서 저가 <= min(SMA200*(1-BUF), SMA50) → 터치 즉시 체결
               - 종가가 되돌아와도 체결 유지 (휩쏘 포함)
             RSI/VIX 임시보유는 종가 확정 → LOC
  loc        모든 신호를 당일 종가 판정 → 당일 종가 체결
  next_open  모든 신호를 당일 종가 판정 → 다음날 시가 체결

Tax (해외주식 양도소득세)
  연간 실현손익 합산(같은 해 손익 상계, 이월공제 없음),
  기본공제 250만원 초과분 22%, 해당 연도 말에 현금에서 납부
  (현금 부족 시 보유분 일부 매도).

Cash
  미투자 현금은 외화RP 연이율(기본 3.45%)을 일할로 가산 (cash *= 1+r/252).
  이자소득세는 미반영(양도세만).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from kcm_fill_rules import classify_transitions
from notify_kcm_telegram import BUF, RSI_DIP, VIX_TH, rsi_sma, trim_scale, vt_scale

TAX_RATE = 0.22
TAX_DEDUCTION_KRW = 2_500_000
INITIAL_KRW = 100_000_000
CASH_RATE = 0.0345  # 외화RP 연이율


def _temp_hold(close: pd.Series, vix: pd.Series, rsi_th: float) -> np.ndarray:
    rsi = rsi_sma(close).ffill().fillna(50)
    vv = vix.reindex(close.index).ffill()
    vx10 = vv.rolling(10).mean()
    vr = (vv > VIX_TH) & (vv < vx10)
    return ((rsi < rsi_th) | vr.fillna(False)).values


def simulate(
    sig_c: pd.Series,
    sig_o: pd.Series,
    sig_h: pd.Series,
    sig_l: pd.Series,
    veh_c: pd.Series,
    veh_o: pd.Series,
    vix: pd.Series,
    *,
    fill_mode: str | None = None,
    sma_fill: str = "next_open",
    temp_fill: str = "next_open",
    start: str = "2000-01-01",
    end: str | None = None,
    lev: float = 3.0,
    cost_bps: float = 9.0,
    vt_target: float | None = None,
    tax: bool = True,
    initial: float = INITIAL_KRW,
    rsi_th: float = RSI_DIP,
    trim: pd.Series | str | None = "default",
    rebal: float = 0.1,
    cash_rate: float = CASH_RATE,
) -> dict:
    """trim: close-decided exposure multiplier (0..1] while ON; rebalanced next open
    when target moves by >= rebal. "default" = trim_scale (vs200>15%→50%, <10% 복귀);
    None = no trim. cash_rate: annual RP yield on idle cash (default 3.45%)."""
    if isinstance(trim, str):
        trim = trim_scale(sig_c)
    if fill_mode is not None:  # uniform shortcut
        sma_fill = fill_mode
        temp_fill = "loc" if fill_mode == "intraday" else fill_mode
    idx = sig_c.index
    c, o, h, l = (s.astype(float).values for s in (sig_c, sig_o, sig_h, sig_l))
    vc = veh_c.reindex(idx).astype(float).values
    vo = veh_o.reindex(idx).astype(float).values
    s50p = sig_c.rolling(50).mean().shift(1).values
    s200p = sig_c.rolling(200).mean().shift(1).values
    temp = _temp_hold(sig_c, vix, rsi_th)
    vts = vt_scale(sig_c, target=vt_target).values if vt_target else np.ones(len(idx))
    _, inp_close, _ = classify_transitions(sig_c, vix, rsi_th=rsi_th)
    inp_c = inp_close.values
    m = trim.reindex(idx).ffill().fillna(1.0).clip(0.0, 1.0).values if trim is not None else np.ones(len(idx))
    r_cash = cash_rate / 252.0

    t0 = idx.searchsorted(pd.Timestamp(start))
    t1 = len(idx) if end is None else idx.searchsorted(pd.Timestamp(end), side="right")
    while t1 > t0 and not (np.isfinite(vc[t1 - 1]) and np.isfinite(vo[t1 - 1])):
        t1 -= 1
    cost = cost_bps / 1e4

    cash, units, avg = initial, 0.0, 0.0
    realized = 0.0
    tax_paid = 0.0
    trades = 0
    curve = np.full(t1 - t0, np.nan)

    def eq(p: float) -> float:
        return cash + units * p

    def trade(target_w: float, p: float) -> None:
        nonlocal cash, units, avg, realized, trades
        tgt_units = target_w * eq(p) / p
        du = tgt_units - units
        if abs(du) * p < 1e-6 * max(eq(p), 1.0):
            return
        trades += 1
        fee = abs(du) * p * cost
        if du > 0:
            avg = (avg * units + du * p) / (units + du)
            cash -= du * p + fee
        else:
            realized += (p - avg) * (-du) - fee
            cash += (-du) * p - fee
        units = tgt_units
        if units <= 1e-12:
            units, avg = 0.0, 0.0

    def pay_tax(p: float) -> None:
        nonlocal cash, units, realized, tax_paid
        due = TAX_RATE * max(0.0, realized - TAX_DEDUCTION_KRW) if tax else 0.0
        realized = 0.0
        if due <= 0:
            return
        tax_paid += due
        if cash < due and units > 0:
            sell_u = min(units, (due - cash) / p)
            realized_part = (p - avg) * sell_u
            cash += sell_u * p
            units -= sell_u
            realized += realized_part  # counts toward next year
        cash -= due

    def vehicle_at(i: int, sig_px: float) -> float:
        if o[i] <= 0 or not np.isfinite(vo[i]):
            return vc[i]
        p = vo[i] * (1 + lev * (sig_px / o[i] - 1))
        return float(max(p, vo[i] * 0.05))

    # ---- state warm-up (before start) ----
    if sma_fill == "intraday":
        inp = bool(c[0] >= (s200p[1] if np.isfinite(s200p[1]) else c[0]))
        for i in range(1, t0):
            if not (np.isfinite(s50p[i]) and np.isfinite(s200p[i])):
                continue
            if not inp and h[i] >= min(s200p[i] * (1 + BUF), s50p[i]):
                inp = True
            elif inp and l[i] <= min(s200p[i] * (1 - BUF), s50p[i]):
                inp = False
    else:
        inp = bool(inp_c[t0 - 1]) if t0 > 0 else bool(inp_c[0])
    held_w = 0.0  # current weight
    base_w = 1.0  # sticky VT scale fixed at entry
    pending = None

    for k, i in enumerate(range(t0, t1)):
        if k > 0 and idx[i].year != idx[i - 1].year:
            pay_tax(vc[i - 1])

        # overnight RP interest on cash held after previous close
        if k > 0 and r_cash != 0.0 and cash > 0:
            cash *= 1.0 + r_cash

        if pending is not None:
            trade(pending, vo[i])
            held_w = pending
            pending = None

        if sma_fill == "intraday":
            ok = np.isfinite(s50p[i]) and np.isfinite(s200p[i])
            if ok and k > 0:
                lb = min(s200p[i] * (1 + BUF), s50p[i])
                ls = min(s200p[i] * (1 - BUF), s50p[i])
                if not inp and h[i] >= lb:
                    inp = True
                    if held_w == 0.0:
                        base_w = float(vts[i - 1])
                        held_w = base_w * m[i - 1]
                        trade(held_w, vehicle_at(i, max(o[i], lb)))
                elif inp and l[i] <= ls:
                    inp = False
                    if held_w > 0.0:
                        trade(0.0, vehicle_at(i, min(o[i], ls)))
                        held_w = 0.0
            regime_flip = False  # SMA trades already done intraday
        else:
            prev_inp = inp
            inp = bool(inp_c[i])
            regime_flip = inp != prev_inp

        want_on = inp or bool(temp[i])
        if k == 0:
            if want_on:
                base_w = float(vts[i])
                held_w = base_w * m[i]
                trade(held_w, vc[i])
        elif want_on != (held_w > 0.0):
            # cause: regime flip decides unless temp hold makes the flip moot
            style = sma_fill if regime_flip else temp_fill
            if want_on:
                base_w = float(vts[i])
            tgt = base_w * m[i] if want_on else 0.0
            if style == "loc":
                trade(tgt, vc[i])
                held_w = tgt
            elif style == "next_open":
                pending = tgt
            else:
                raise ValueError(style)
        elif want_on and held_w > 0.0 and trim is not None:
            tgt = base_w * m[i]
            if abs(tgt - held_w) >= rebal - 1e-9:
                pending = tgt

        curve[k] = eq(vc[i])

    pay_tax(vc[t1 - 1])  # final year
    curve[-1] = eq(vc[t1 - 1])

    s = pd.Series(curve / initial, index=idx[t0:t1])
    yrs = max((s.index[-1] - s.index[0]).days / 365.25, 1e-9)
    term = float(s.iloc[-1])
    return {
        "nav": s,
        "cagr": term ** (1 / yrs) - 1 if term > 0 else float("nan"),
        "mdd": float((s / s.cummax() - 1).min()),
        "total": term - 1,
        "trades": trades,
        "tax_paid_ratio": tax_paid / initial,
    }
