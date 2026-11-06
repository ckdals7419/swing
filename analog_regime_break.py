"""
Find historical analogs of: regime just turned OFF + SMA-RSI<26 + VIX spike,
then compare policies over the following days (VT25 as optional overlay).

Signal: ^NDX (1990+, VIX available). Vehicle: synthetic 3x NDX, spliced to real TQQQ.
Signal at T close -> trade at T+1 open.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

from notify_kcm_telegram import BUF, RSI_DIP, VT_TARGET, rsi_sma

COST_BPS = 9.0
HORIZONS = (1, 5, 10, 20, 60)


def _dl(t: str, start: str = "1985-01-01") -> pd.DataFrame:
    d = yf.download(t, start=start, auto_adjust=True, progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    return d.dropna(subset=["Close"])


def build() -> pd.DataFrame:
    ndx, tqqq, vix = _dl("^NDX"), _dl("TQQQ", "2010-01-01"), _dl("^VIX", "1990-01-01")
    c, o = ndx["Close"].astype(float), ndx["Open"].astype(float)
    ret = c.pct_change().fillna(0.0)
    oc = (o / c.shift(1)).fillna(1.0)

    veh = pd.Series(100.0, index=c.index)
    for i in range(1, len(veh)):
        veh.iloc[i] = veh.iloc[i - 1] * (1 + 3 * float(ret.iloc[i]) - 1e-4)
    vo = veh.shift(1) * (1 + 3 * (oc - 1))
    vo.iloc[0] = 100.0

    rc, ro = tqqq["Close"].astype(float), tqqq["Open"].astype(float)
    sc = float(veh.loc[rc.index[0]]) / float(rc.iloc[0])
    cm = rc.index.intersection(veh.index)
    veh.loc[cm] = rc.loc[cm] * sc
    vo.loc[cm] = ro.loc[cm] * sc

    df = pd.DataFrame({"close": c, "veh_c": veh, "veh_o": vo})
    df["vix"] = vix["Close"].astype(float).reindex(df.index).ffill()
    df["vs200"] = (c / c.rolling(200).mean() - 1)
    df["vs50"] = (c / c.rolling(50).mean() - 1)
    df["rsi"] = rsi_sma(c)
    df["vix10"] = df["vix"].rolling(10).mean()
    df["rvol"] = c.pct_change().rolling(20).std() * np.sqrt(252)
    df["scale"] = (VT_TARGET / df["rvol"]).clip(upper=1.0)
    return df


def kcm(df: pd.DataFrame, temp_buys: bool = True) -> tuple[pd.Series, pd.Series]:
    vs200, vs50 = df["vs200"].fillna(0), df["vs50"].fillna(0)
    rsi, vx, vx10 = df["rsi"].fillna(50), df["vix"], df["vix10"]
    n = len(df)
    w = np.zeros(n)
    st = np.zeros(n, dtype=bool)
    inp = bool(vs200.iloc[0] >= 0)
    w[0], st[0] = float(inp), inp
    for i in range(1, n):
        vr = bool(pd.notna(vx10.iloc[i]) and vx.iloc[i] > 35 and vx.iloc[i] < vx10.iloc[i])
        temp = temp_buys and (vr or rsi.iloc[i] < RSI_DIP)
        if inp:
            if vs200.iloc[i] < -BUF and vs50.iloc[i] < 0:
                inp = False
                w[i] = 1.0 if temp else 0.0
            else:
                w[i] = 1.0
        else:
            if vs200.iloc[i] >= BUF or vs50.iloc[i] >= 0:
                inp = True
                w[i] = 1.0
            else:
                w[i] = 1.0 if temp else 0.0
        st[i] = inp
    return pd.Series(w, index=df.index), pd.Series(st, index=df.index)


def find_analogs(df: pd.DataFrame, inp: pd.Series, strict: bool) -> list[pd.Timestamp]:
    fresh_off = inp.shift(1).fillna(False).astype(bool) & ~inp
    vix_jump = df["vix"] / df["vix"].shift(1) - 1
    vix_vs_ma = df["vix"] / df["vix10"]
    if strict:
        cond = (fresh_off & (df["rsi"] < RSI_DIP) & (vix_jump >= 0.25) & (df["scale"] < 1.0))
    else:
        recent_off = (~inp) & inp.shift(1).rolling(5).max().fillna(0).astype(bool)
        cond = (recent_off & (df["rsi"] < RSI_DIP + 2) & (vix_vs_ma >= 1.25) & (df["scale"] < 1.0))
    days = list(df.index[cond.fillna(False)])
    out: list[pd.Timestamp] = []
    for d in days:
        if not out or (d - out[-1]).days > 30:
            out.append(d)
    return out


def run_window(df: pd.DataFrame, w: pd.Series, t: pd.Timestamp, h: int) -> tuple[float, float]:
    """Weight decided at close of day k applies to open[k+1] -> open[k+2]."""
    idx = df.index
    i0 = idx.get_loc(t)
    if i0 + h + 1 >= len(idx):
        return np.nan, np.nan
    nav, peak, mdd, prev = 1.0, 1.0, 0.0, None
    for k in range(h):
        i = i0 + k
        ww = float(w.iloc[i])
        if prev is not None:
            nav *= 1 - abs(ww - prev) * COST_BPS / 1e4
        r = float(df["veh_o"].iloc[i + 2] / df["veh_o"].iloc[i + 1] - 1)
        nav *= 1 + ww * r
        peak = max(peak, nav)
        mdd = min(mdd, nav / peak - 1)
        prev = ww
    return nav - 1, mdd


def main() -> None:
    df = build().loc["1990-01-01":]
    w_kcm, inp = kcm(df, temp_buys=True)
    w_notemp, _ = kcm(df, temp_buys=False)
    scale = df["scale"].fillna(1.0)

    policies = {
        "KCM (기준, VT 없음)": w_kcm,
        "KCM+VT25 (옵션)": (w_kcm * scale).clip(0, 1),
        "임시매수 없이 즉시청산": w_notemp,
        "임시매수없음+VT25": (w_notemp * scale).clip(0, 1),
        "100% 계속보유": pd.Series(1.0, index=df.index),
    }

    for strict in (True, False):
        label = "엄격" if strict else "완화"
        days = find_analogs(df, inp, strict)
        print("=" * 110)
        print(f"[{label} 조건] 사례 {len(days)}건")
        if strict:
            print(f"  조건: 당일 레짐 ON→OFF, SMA-RSI<{RSI_DIP:.0f}, VIX 전일대비 +25% 이상, VT scale<100%")
        else:
            print(f"  조건: 최근 5일 내 레짐 OFF 전환, SMA-RSI<{RSI_DIP + 2:.0f}, VIX/10일평균 ≥1.25, VT scale<100%")
        print("=" * 110)
        for d in days:
            r = df.loc[d]
            vj = r["vix"] / df["vix"].shift(1).loc[d] - 1
            print(f"  {d.date()}  NDX일간 {df['close'].pct_change().loc[d]:+.1%}  vs200 {r['vs200']:+.1%}  "
                  f"RSI {r['rsi']:.0f}  VIX {r['vix']:.0f} ({vj:+.0%})  VT {r['scale']:.0%}")

        for h in HORIZONS:
            print(f"\n  -- 이후 {h}거래일 (신호 다음날 시가부터) --")
            print(f"  {'정책':<26}{'평균':>8}{'중앙값':>8}{'승률':>7}{'최악':>8}{'평균MDD':>9}")
            for name, w in policies.items():
                rets, mdds = [], []
                for d in days:
                    ret, mdd = run_window(df, w, d, h)
                    if not np.isnan(ret):
                        rets.append(ret)
                        mdds.append(mdd)
                if not rets:
                    continue
                a = np.array(rets)
                print(f"  {name:<24}{a.mean():>+8.1%}{np.median(a):>+8.1%}"
                      f"{(a > 0).mean():>7.0%}{a.min():>+8.1%}{np.mean(mdds):>+9.1%}")

        print(f"\n  -- 사례별 20거래일 수익 --")
        names = list(policies)
        print(f"  {'날짜':<12}" + "".join(f"{n[:12]:>14}" for n in names))
        for d in days:
            row = f"  {str(d.date()):<12}"
            for n in names:
                ret, _ = run_window(df, policies[n], d, 20)
                row += f"{ret:>+13.1%} " if not np.isnan(ret) else f"{'n/a':>13} "
            print(row)


if __name__ == "__main__":
    main()
