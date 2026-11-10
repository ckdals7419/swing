"""
Backtest KCM with trigger-specific fills (hybrid) vs uniform methods.

Usage:
  python backtest_kcm_fills.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

from kcm_fill_rules import FILL_RULES, hybrid_backtest, classify_transitions

LEV = 3.0
COST = 9.0


def dl(t: str, start: str = "1990-01-01") -> pd.DataFrame:
    d = yf.download(t, start=start, auto_adjust=True, progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    return d


def build_vehicle():
    ndx = dl("^NDX")
    tqqq = dl("TQQQ", "2010-01-01")
    vix = dl("^VIX")
    sig_c = ndx["Close"].astype(float)
    sig_o = ndx["Open"].astype(float).fillna(sig_c)
    sig_h = ndx["High"].astype(float).fillna(np.maximum(sig_o, sig_c))
    sig_l = ndx["Low"].astype(float).fillna(np.minimum(sig_o, sig_c))
    ret = sig_c.pct_change().fillna(0.0)
    oc = (sig_o / sig_c.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    veh_c = pd.Series(100.0, index=sig_c.index)
    for i in range(1, len(veh_c)):
        veh_c.iloc[i] = veh_c.iloc[i - 1] * (1 + LEV * float(ret.iloc[i]) - 1e-4)
    veh_o = veh_c.shift(1) * (1 + LEV * (oc - 1))
    veh_o.iloc[0] = 100.0
    rc = tqqq["Close"].astype(float)
    ro = tqqq["Open"].astype(float)
    ok = rc.notna() & ro.notna()
    rc, ro = rc[ok], ro[ok]
    sc0 = float(veh_c.loc[rc.index[0]]) / float(rc.iloc[0])
    cm = rc.index.intersection(veh_c.index)
    veh_c.loc[cm] = rc.loc[cm] * sc0
    veh_o.loc[cm] = ro.loc[cm] * sc0
    veh_o = veh_o.fillna(veh_c).ffill()
    veh_c = veh_c.ffill()
    vx = vix["Close"].astype(float).reindex(sig_c.index).ffill()
    return sig_c, sig_o, sig_h, sig_l, veh_c, veh_o, vx


def main() -> None:
    print("Building dataset...")
    sig_c, sig_o, sig_h, sig_l, veh_c, veh_o, vx = build_vehicle()
    _, _, events = classify_transitions(sig_c, vx)
    print("Fill rules:")
    for k, v in FILL_RULES.items():
        print(f"  {k}: {v}")
    print(f"Tagged flips (full hist): {len(events)}")

    periods = [
        ("2000+", "2000-01-01", None),
        ("2010+", "2010-02-11", None),
        ("닷컴00-02", "2000-01-01", "2002-12-31"),
        ("코로나20", "2020-01-01", "2020-12-31"),
        ("2022", "2022-01-01", "2022-12-31"),
        ("2025-26", "2025-01-01", None),
    ]
    modes = [
        ("hybrid(원칙)", "hybrid"),
        ("전부 장중", "intraday"),
        ("전부 LOC", "loc"),
        ("전부 시가", "next_open"),
    ]

    for vt_name, vt in [("VT없음", None), ("VT25", 0.25)]:
        print("\n" + "=" * 110)
        print(f"KCM SMA-RSI26+VIX | {vt_name} | {COST:.0f}bps | 3x→TQQQ")
        print("=" * 110)
        print(f"{'구간':<10}", end="")
        for lab, _ in modes:
            print(f" {lab:^22}", end="")
        print()
        results = {}
        for pname, ps, pe in periods:
            print(f"{pname:<10}", end="")
            best = None
            best_cagr = -1e9
            for lab, mode in modes:
                r = hybrid_backtest(
                    sig_c, sig_o, sig_h, sig_l, veh_c, veh_o, vx,
                    start=ps, end=pe, cost_bps=COST, vt_target=vt, fill_mode=mode,
                )
                results[(pname, lab)] = r
                print(f"  {r['cagr']:+6.1%}/{r['mdd']:6.1%}", end="")
                if np.isfinite(r["cagr"]) and r["cagr"] > best_cagr:
                    best_cagr = r["cagr"]
                    best = lab
            print(f"  ← {best}")

        print(f"\n상세 2000+ / 2010+ ({vt_name})")
        print(f"{'방식':<16}{'CAGR00':>9}{'MDD00':>8}{'Tot00':>10}{'TR':>6} | {'CAGR10':>9}{'MDD10':>8}{'TR':>6}")
        for lab, mode in modes:
            a = results[("2000+", lab)]
            b = results[("2010+", lab)]
            print(
                f"{lab:<16}{a['cagr']:>+8.1%}{a['mdd']:>7.1%}{a['total']:>+9.0%}{a['trades']:>6} | "
                f"{b['cagr']:>+8.1%}{b['mdd']:>7.1%}{b['trades']:>6}"
            )


if __name__ == "__main__":
    main()
