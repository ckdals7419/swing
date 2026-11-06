"""
Review KCM rules that act while the regime is OFF (below SMA200):
  - SMA-RSI<26 temporary hold (baseline)
  - VIX>35 & falling temporary hold
  - SMA50 re-entry (vs50 >= 0)

Signal: ^NDX (1990+). Vehicle: synthetic 3x NDX spliced to real TQQQ (2010-02-11+).
Baseline is unscaled 0/100% KCM; VT25 shown as an optional second pass.
9 bps/side, signal at T close -> fill T+1 open.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from analog_regime_break import build
from notify_kcm_telegram import RSI_DIP

COST = 9e-4


@dataclass
class Rules:
    name: str
    rsi_th: float | None = RSI_DIP   # None disables RSI temp hold
    vix_th: float | None = 35.0      # None disables VIX temp hold
    hold_days: int = 0               # keep temp hold at least N days once triggered
    rsi_exit: float | None = None    # keep temp hold until RSI >= this (None: exit when trigger ends)
    sma50_reentry: bool = True       # vs50 >= 0 re-enters regime
    sma50_buf: float = 0.0           # vs50 must be >= this to re-enter
    confirm: int = 1                 # consecutive days entry condition must hold
    tags: list[str] = field(default_factory=list)


def weights(df: pd.DataFrame, r: Rules) -> tuple[pd.Series, pd.Series]:
    vs200 = df["vs200"].fillna(0).values
    vs50 = df["vs50"].fillna(0).values
    rsi = df["rsi"].fillna(50).values
    vx = df["vix"].values
    vx10 = df["vix10"].values
    n = len(df)
    w = np.zeros(n)
    temp_flag = np.zeros(n, dtype=bool)
    inp = bool(vs200[0] >= 0)
    w[0] = float(inp)
    streak = 0
    temp_left = 0
    in_temp = False
    for i in range(1, n):
        rsi_trig = r.rsi_th is not None and rsi[i] < r.rsi_th
        vix_trig = (r.vix_th is not None and not np.isnan(vx10[i])
                    and vx[i] > r.vix_th and vx[i] < vx10[i])
        trig = rsi_trig or vix_trig

        if inp:
            if vs200[i] < -0.01 and vs50[i] < 0:
                inp = False
                streak = 0
            else:
                w[i] = 1.0
                in_temp = False
                continue

        entry = vs200[i] >= 0.01 or (r.sma50_reentry and vs50[i] >= r.sma50_buf)
        streak = streak + 1 if entry else 0
        if streak >= r.confirm:
            inp = True
            in_temp = False
            w[i] = 1.0
            continue

        if trig:
            in_temp = True
            temp_left = r.hold_days
        elif in_temp:
            keep = False
            if temp_left > 0:
                temp_left -= 1
                keep = True
            if r.rsi_exit is not None and rsi[i] < r.rsi_exit:
                keep = True
            in_temp = keep
        w[i] = 1.0 if in_temp else 0.0
        temp_flag[i] = in_temp
    return pd.Series(w, index=df.index), pd.Series(temp_flag, index=df.index)


def backtest(df: pd.DataFrame, w: pd.Series, start: str, end: str | None = None) -> pd.Series:
    sub = df.loc[start:end]
    ws = w.reindex(sub.index).values
    vo, vc = sub["veh_o"].values, sub["veh_c"].values
    nav, prev = 1.0, 0.0
    out = np.empty(len(sub))
    for i in range(len(sub)):
        ww = ws[i - 1] if i > 0 else 0.0
        nav *= 1 - abs(ww - prev) * COST
        r = vo[i + 1] / vo[i] - 1 if i < len(sub) - 1 else vc[i] / vo[i] - 1
        nav *= 1 + ww * r
        out[i] = nav
        prev = ww
    return pd.Series(out, index=sub.index)


def stats(nav: pd.Series) -> tuple[float, float]:
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    return float(nav.iloc[-1] ** (1 / yrs) - 1), float((nav / nav.cummax() - 1).min())


def temp_attribution(df: pd.DataFrame, w: pd.Series, temp: pd.Series, start: str) -> tuple[int, float, float]:
    """Return (#episodes, avg episode return on vehicle, win rate) for temp holds."""
    sub = df.loc[start:]
    t = temp.reindex(sub.index).fillna(False).values
    vo = sub["veh_o"].values
    eps = []
    i = 0
    n = len(sub)
    while i < n - 2:
        if t[i] and (i == 0 or not t[i - 1]):
            j = i
            while j < n - 1 and t[j]:
                j += 1
            # weight decided at close i..j-1 applies opens i+1..j+1
            eps.append(vo[min(j + 1, n - 1)] / vo[i + 1] - 1)
            i = j
        i += 1
    if not eps:
        return 0, 0.0, 0.0
    a = np.array(eps)
    return len(a), float(a.mean()), float((a > 0).mean())


def main() -> None:
    df = build().loc["1990-01-01":]
    scale = df["scale"].fillna(1.0)

    variants = [
        Rules(f"기준 (SMA-RSI{RSI_DIP:.0f}+VIX35+SMA50재진입)"),
        Rules("임시매수 전부 없음", rsi_th=None, vix_th=None),
        Rules("RSI만 (VIX 끔)", vix_th=None),
        Rules("VIX만 (RSI 끔)", rsi_th=None),
        Rules("RSI<24", rsi_th=24),
        Rules("RSI<30", rsi_th=30),
        Rules("VIX>30", vix_th=30),
        Rules("VIX>40", vix_th=40),
        Rules("임시보유 최소 5일", hold_days=5),
        Rules("임시보유 최소 10일", hold_days=10),
        Rules("임시보유 RSI 40 회복까지", rsi_exit=40),
        Rules("임시보유 RSI 50 회복까지", rsi_exit=50),
        Rules("SMA50 재진입 끔 (SMA200+1%만)", sma50_reentry=False),
        Rules("SMA50 재진입 +1% 버퍼", sma50_buf=0.01),
        Rules("SMA50 재진입 +2% 버퍼", sma50_buf=0.02),
        Rules("재진입 2일 확인", confirm=2),
        Rules("재진입 3일 확인", confirm=3),
    ]

    periods = [
        ("1990-99", "1990-01-01", "1999-12-31"),
        ("2000-09", "2000-01-01", "2009-12-31"),
        ("2010+실물", "2010-02-11", None),
        ("전체 1990+", "1990-01-01", None),
    ]

    print("기준 = VT 없음 (0/100%), 비용 9bps, 각 칸 = CAGR / MDD")
    hdr = f"{'규칙':<34}" + "".join(f"{p[0]:>20}" for p in periods)
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for r in variants:
        w, temp = weights(df, r)
        line = f"{r.name:<32}"
        vals = []
        for _, s, e in periods:
            c, m = stats(backtest(df, w, s, e))
            vals.append((c, m))
            line += f"  {c:+6.1%} / {m:6.1%}"
        print(line)
        n_ep, avg, win = temp_attribution(df, w, temp, "1990-01-01")
        rows.append((r.name, vals, n_ep, avg, win))

    print("\n[옵션] 동일 규칙 × VT25 스케일")
    print(hdr)
    print("-" * len(hdr))
    for r in variants[:4]:
        w, _ = weights(df, r)
        wvt = (w * scale).clip(0, 1)
        line = f"{r.name + '+VT25':<32}"
        for _, s, e in periods:
            c, m = stats(backtest(df, wvt, s, e))
            line += f"  {c:+6.1%} / {m:6.1%}"
        print(line)

    print("\n임시보유 에피소드 성과 (1990+, 3x 차량, VT 전)")
    print(f"{'규칙':<34}{'횟수':>6}{'평균수익':>10}{'승률':>8}")
    for name, _, n_ep, avg, win in rows:
        if n_ep:
            print(f"{name:<32}{n_ep:>6}{avg:>+10.1%}{win:>8.0%}")


if __name__ == "__main__":
    main()
