"""
Overextension trim test: SMA200 위 과열 시 현금 비중 확대.

All rules are decided on signal close (QQQ/NDX) and executed next open
(rebalance when target moves >= 10%p). Base: SMA 시가 · RSI/VIX LOC, 세후,
현금 RP 3.45% (`kcm_sim.CASH_RATE`).

Selection: IS 2000-2012 → OOS 2013+.
Results CSV: results_kcm_trim.csv

Usage:
  python backtest_kcm_trim.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest_kcm_fills import COST, build_vehicle
from kcm_sim import CASH_RATE, simulate
from notify_kcm_telegram import rsi_sma

OUT_CSV = "results_kcm_trim.csv"

PERIODS = [
    ("IS00-12", "2000-01-01", "2012-12-31"),
    ("OOS13+", "2013-01-01", None),
    ("2000+", "2000-01-01", None),
    ("2010+", "2010-02-11", None),
    ("닷컴", "2000-01-01", "2002-12-31"),
    ("08위기", "2007-01-01", "2009-12-31"),
    ("코로나", "2020-01-01", "2020-12-31"),
    ("2022", "2022-01-01", "2022-12-31"),
]


def hysteresis(x: pd.Series, on_th: float, off_th: float, y: float) -> pd.Series:
    """y while x went above on_th and has not yet fallen below off_th, else 1."""
    out = np.ones(len(x))
    hot = False
    v = x.values
    for i in range(len(v)):
        if not np.isfinite(v[i]):
            continue
        if not hot and v[i] > on_th:
            hot = True
        elif hot and v[i] < off_th:
            hot = False
        out[i] = y if hot else 1.0
    return pd.Series(out, index=x.index)


def build_rules(c: pd.Series) -> dict[str, pd.Series | None]:
    vs200 = c / c.rolling(200).mean() - 1
    vs50 = c / c.rolling(50).mean() - 1
    rsi = rsi_sma(c)
    r252 = c / c.shift(252) - 1
    rules: dict[str, pd.Series | None] = {"기준(트림 없음)": None}
    for x in (0.15, 0.20, 0.25, 0.30, 0.40):
        for y in (0.5, 0.7, 0.0):
            rules[f"vs200>{x:.0%}→{y:.0%}"] = hysteresis(vs200, x, x - 0.05, y)
    for a, b in ((0.10, 0.20), (0.10, 0.40), (0.20, 0.20), (0.20, 0.40)):
        rules[f"선형 vs200 {a:.0%}~{a + b:.0%}→50%"] = (1 - (vs200 - a) / b * 0.5).clip(0.5, 1.0)
    for t in (75, 80, 85):
        for y in (0.5, 0.7):
            rules[f"RSI>{t}→{y:.0%}"] = hysteresis(rsi, t, 60, y)
    for x in (0.08, 0.10, 0.15):
        for y in (0.5, 0.7):
            rules[f"vs50>{x:.0%}→{y:.0%}"] = hysteresis(vs50, x, x - 0.04, y)
    for x in (0.40, 0.60):
        for y in (0.5, 0.7):
            rules[f"1년수익>{x:.0%}→{y:.0%}"] = hysteresis(r252, x, x - 0.10, y)
    return rules


def run(data, rules, vt) -> pd.DataFrame:
    rows = []
    for name, tr in rules.items():
        row = {"vt": "VT25" if vt else "VT없음", "rule": name, "cash_rate": CASH_RATE, "cost_bps": COST}
        for pn, ps, pe in PERIODS:
            r = simulate(
                *data,
                sma_fill="next_open",
                temp_fill="loc",
                start=ps,
                end=pe,
                cost_bps=COST,
                vt_target=vt,
                trim=tr,
                cash_rate=CASH_RATE,
            )
            row[f"{pn} CAGR"] = r["cagr"]
            row[f"{pn} MDD"] = r["mdd"]
            if pn == "2000+":
                row["trades"] = r["trades"]
        row["IS calmar"] = row["IS00-12 CAGR"] / abs(row["IS00-12 MDD"])
        rows.append(row)
    return pd.DataFrame(rows)


def show(df: pd.DataFrame, title: str) -> None:
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)
    base = df.iloc[0]
    cols = ["IS00-12", "OOS13+", "2000+", "2010+"]
    print(f"{'규칙':<26}" + "".join(f"{p:>17}" for p in cols) + f"{'거래':>6}")
    for _, r in df.sort_values("IS calmar", ascending=False).iterrows():
        cells = "".join(f"{r[p + ' CAGR']:>+8.1%}/{r[p + ' MDD']:>7.1%} " for p in cols)
        both = (r["OOS13+ CAGR"] >= base["OOS13+ CAGR"]) and (r["OOS13+ MDD"] >= base["OOS13+ MDD"])
        tag = " ★OOS 둘다개선" if both and r["rule"] != base["rule"] else ""
        print(f"{r['rule']:<26}{cells}{int(r['trades']):>6}{tag}")


def main() -> None:
    print("Building dataset...")
    data = build_vehicle()
    rules = build_rules(data[0])
    frames = []
    for vt_name, vt in [("VT없음", None), ("VT25", 0.25)]:
        df = run(data, rules, vt)
        show(df, f"과열 트림 | {vt_name} | 세후 | RP{CASH_RATE:.2%} | {COST:.0f}bps | IS calmar 순")
        frames.append(df)
        # depth spotlight
        keep = df[df["rule"].isin(["기준(트림 없음)", "vs200>15%→50%", "vs200>15%→0%"])].copy()
        print(f"\n깊이 비교 ({vt_name}, RP {CASH_RATE:.2%})")
        print(f"{'규칙':<18}" + "".join(f"{p:>16}" for p, _, _ in PERIODS))
        for _, r in keep.iterrows():
            cells = "".join(f"{r[p + ' CAGR']:>+7.1%}/{r[p + ' MDD']:6.1%}" for p, _, _ in PERIODS)
            print(f"{r['rule']:<18}{cells}")

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(OUT_CSV, index=False, float_format="%.6f")
    print(f"\nSaved {OUT_CSV} ({len(out)} rows)")


if __name__ == "__main__":
    main()
