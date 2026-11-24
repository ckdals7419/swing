"""
Multi-step overheat trim research (vs200 ladder) with overfit checks.

Families (all decided on signal close, rebalanced next open when target moves >= 10%p):
  단일      vs200 > on → w, < off → 100%                     (구 기본 15/10→50%, 현행 15/5→25%)
  단계복귀  vs200 > on → w1, < off1 → w2, < off2 → 100%       (예: 15/10/5)
  단계축소  vs200 > a → 75%, > a+s → 50%, > a+2s → 25%, 각 단계 gap 아래로 내려오면 해제
  2단       vs200 > on1 → 50%, > on2 → 25%
  선형      w = clip(1 - (vs200 - a) / span * (1 - wmin), wmin, 1)

Overfit checks:
  - family grid: median / worst neighbour vs 현행
  - IS 2000-12 Calmar 1위 → OOS 2013+ 순위
  - 닷컴 제외(2003+) MDD·Ulcer, 위기별 낙폭

Results CSV: results_kcm_trim_ladder.csv

Usage:
  python backtest_kcm_trim_ladder.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest_kcm_fills import COST, build_vehicle
from kcm_sim import CASH_RATE, simulate

OUT_CSV = "results_kcm_trim_ladder.csv"
BASE = "단일 15/10→50%"

CRISES = [
    ("GFC", "2007-06-01", "2009-12-31"),
    ("코로나", "2020-01-01", "2020-06-30"),
    ("2022", "2021-10-01", "2023-03-31"),
    ("2025", "2025-01-01", "2025-06-30"),
]


def ladder(x: pd.Series, levels: list[tuple[float, float, float]]) -> pd.Series:
    """Each level (on, off, w) latches when x > on and releases when x < off.
    Exposure = min(w of active levels), 1 if none."""
    v = x.values
    active = [False] * len(levels)
    out = np.ones(len(v))
    for i in range(len(v)):
        if np.isfinite(v[i]):
            for k, (on, off, _) in enumerate(levels):
                if not active[k] and v[i] > on:
                    active[k] = True
                elif active[k] and v[i] < off:
                    active[k] = False
        ws = [w for k, (_, _, w) in enumerate(levels) if active[k]]
        out[i] = min(ws) if ws else 1.0
    return pd.Series(out, index=x.index)


def build_rules(c: pd.Series) -> list[tuple[str, str, pd.Series | None]]:
    vs = c / c.rolling(200).mean() - 1
    R: list[tuple[str, str, pd.Series | None]] = [("없음", "트림없음", None)]

    for on in (0.12, 0.15, 0.18):
        for gap in (0.03, 0.05, 0.08):
            R.append(("단일", f"단일 {on*100:.0f}/{(on-gap)*100:.0f}→50%",
                      ladder(vs, [(on, on - gap, 0.5)])))

    for on in (0.12, 0.15, 0.18, 0.20):
        for gap in (0.04, 0.05, 0.06):
            o1, o2 = on - gap, on - 2 * gap
            for w1, w2 in ((0.5, 0.75), (0.25, 0.5)):
                R.append(("단계복귀",
                          f"단계복귀 {on*100:.0f}/{o1*100:.0f}/{o2*100:.0f} {w1:.0%}→{w2:.0%}",
                          ladder(vs, [(on, o2, w2), (on, o1, w1)])))

    for a in (0.08, 0.10, 0.12):
        for s in (0.04, 0.05, 0.06):
            for gap in (0.03, 0.05):
                lv = [(a, a - gap, 0.75), (a + s, a + s - gap, 0.5), (a + 2 * s, a + 2 * s - gap, 0.25)]
                R.append(("단계축소",
                          f"단계축소 {a*100:.0f}/{(a+s)*100:.0f}/{(a+2*s)*100:.0f} gap{gap*100:.0f}",
                          ladder(vs, lv)))

    for on1 in (0.12, 0.15, 0.18):
        for on2 in (on1 + 0.07, on1 + 0.10):
            R.append(("2단", f"2단 {on1*100:.0f}→50% {on2*100:.0f}→25%",
                      ladder(vs, [(on1, on1 - 0.05, 0.5), (on2, on2 - 0.05, 0.25)])))

    for a in (0.05, 0.10):
        for span in (0.10, 0.15, 0.20):
            for wmin in (0.5, 0.25):
                w = (1 - (vs - a) / span * (1 - wmin)).clip(wmin, 1.0).fillna(1.0)
                R.append(("선형", f"선형 {a*100:.0f}~{(a+span)*100:.0f}→{wmin:.0%}", w))
    return R


def _seg(nav: pd.Series, a: str, b: str | None) -> pd.Series:
    s = nav.loc[a:b] if b else nav.loc[a:]
    return s / s.iloc[0]


def _cagr(s: pd.Series) -> float:
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    return float(s.iloc[-1]) ** (1 / yrs) - 1


def _mdd(s: pd.Series) -> float:
    return float((s / s.cummax() - 1).min())


def _ulcer(s: pd.Series) -> float:
    dd = s / s.cummax() - 1
    return float(np.sqrt((dd ** 2).mean()))


def evaluate(data, rules) -> pd.DataFrame:
    rows = []
    for fam, name, tr in rules:
        r = simulate(*data, sma_fill="next_open", temp_fill="loc", start="2000-01-01",
                     cost_bps=COST, cash_rate=CASH_RATE, trim=tr)
        nav = r["nav"]
        row = {"family": fam, "rule": name, "trades": r["trades"],
               "trim_days": float((tr < 0.999).mean()) if tr is not None else 0.0}
        for lab, a, b in (("2000+", "2000-01-01", None), ("2003+", "2003-01-01", None),
                          ("IS00-12", "2000-01-01", "2012-12-31"), ("OOS13+", "2013-01-01", None)):
            s = _seg(nav, a, b)
            row[f"{lab} CAGR"], row[f"{lab} MDD"], row[f"{lab} Ulcer"] = _cagr(s), _mdd(s), _ulcer(s)
        row["IS calmar"] = row["IS00-12 CAGR"] / abs(row["IS00-12 MDD"])
        row["03 calmar"] = row["2003+ CAGR"] / abs(row["2003+ MDD"])
        cm = nav.cummax()
        for lab, a, b in CRISES:
            row[f"DD {lab}"] = float((nav.loc[a:b] / cm.loc[a:b] - 1).min())
        rows.append(row)
    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> None:
    base = df[df.rule == BASE].iloc[0]
    none = df[df.rule == "트림없음"].iloc[0]
    pct = lambda v: f"{v:+.1%}"

    print("\n=== 기준 ===")
    for r in (none, base):
        print(f"{r.rule:<24} 2000+ {pct(r['2000+ CAGR'])}/{r['2000+ MDD']:.0%} | "
              f"2003+ {pct(r['2003+ CAGR'])}/{r['2003+ MDD']:.1%} U{r['2003+ Ulcer']:.1%} | "
              f"OOS {pct(r['OOS13+ CAGR'])}/{r['OOS13+ MDD']:.1%} | 거래 {r.trades}")

    print("\n=== 계열별 분포 (2003+ 기준, 이웃 견고성) ===")
    print(f"{'계열':<8}{'n':>4}{'CAGR 중앙':>10}{'CAGR 최저':>10}{'MDD 중앙':>9}{'MDD 최악':>9}"
          f"{'Ulcer 중앙':>10}{'OOS CAGR중앙':>12}{'현행보다 MDD↑ & CAGR≥-1%p':>26}")
    for fam, g in df[df.family != "없음"].groupby("family", sort=False):
        ok = ((g["2003+ MDD"] > base["2003+ MDD"] + 1e-4)
              & (g["2003+ CAGR"] >= base["2003+ CAGR"] - 0.01)).sum()
        print(f"{fam:<8}{len(g):>4}{g['2003+ CAGR'].median():>+10.1%}{g['2003+ CAGR'].min():>+10.1%}"
              f"{g['2003+ MDD'].median():>9.1%}{g['2003+ MDD'].min():>9.1%}{g['2003+ Ulcer'].median():>10.1%}"
              f"{g['OOS13+ CAGR'].median():>+12.1%}{ok:>14}/{len(g)}")

    print("\n=== 계열별 상위 5 (2003+ Calmar) ===")
    cols = ["2000+ CAGR", "2000+ MDD", "2003+ CAGR", "2003+ MDD", "2003+ Ulcer", "OOS13+ CAGR", "OOS13+ MDD",
            "DD GFC", "DD 코로나", "DD 2022", "DD 2025", "trades"]
    for fam, g in df[df.family != "없음"].groupby("family", sort=False):
        print(f"\n[{fam}]")
        for _, r in g.sort_values("03 calmar", ascending=False).head(5).iterrows():
            print(f"  {r.rule:<34} 00+ {pct(r['2000+ CAGR'])}/{r['2000+ MDD']:.0%}  "
                  f"03+ {pct(r['2003+ CAGR'])}/{r['2003+ MDD']:.1%} U{r['2003+ Ulcer']:.1%}  "
                  f"OOS {pct(r['OOS13+ CAGR'])}/{r['OOS13+ MDD']:.0%}  "
                  f"GFC{r['DD GFC']:.0%} 코로나{r['DD 코로나']:.0%} 22{r['DD 2022']:.0%} 25{r['DD 2025']:.0%}  "
                  f"거래{r.trades}")

    print("\n=== IS(2000-12) Calmar 상위 10 → OOS(2013+) 성적 ===")
    d = df[df.family != "없음"].copy()
    d["OOS rank"] = d["OOS13+ CAGR"].rank(ascending=False)
    d["OOS MDD rank"] = d["OOS13+ MDD"].rank(ascending=False)
    n = len(d)
    for _, r in d.sort_values("IS calmar", ascending=False).head(10).iterrows():
        print(f"  {r.rule:<34} IS {pct(r['IS00-12 CAGR'])}/{r['IS00-12 MDD']:.0%}  →  "
              f"OOS {pct(r['OOS13+ CAGR'])}/{r['OOS13+ MDD']:.1%}  "
              f"(CAGR {int(r['OOS rank'])}/{n}, MDD {int(r['OOS MDD rank'])}/{n})")
    b = d[d.rule == BASE].iloc[0]
    print(f"  {'[현행] ' + BASE:<34} IS {pct(b['IS00-12 CAGR'])}/{b['IS00-12 MDD']:.0%}  →  "
          f"OOS {pct(b['OOS13+ CAGR'])}/{b['OOS13+ MDD']:.1%}  "
          f"(CAGR {int(b['OOS rank'])}/{n}, MDD {int(b['OOS MDD rank'])}/{n})")


def main() -> None:
    print("Building dataset...")
    data = build_vehicle()
    rules = build_rules(data[0])
    print(f"Simulating {len(rules)} rules...")
    df = evaluate(data, rules)
    report(df)
    df.to_csv(OUT_CSV, index=False, float_format="%.6f")
    print(f"\nSaved {OUT_CSV} ({len(df)} rows)")


if __name__ == "__main__":
    main()
