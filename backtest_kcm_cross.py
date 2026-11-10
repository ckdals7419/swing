"""
KCM cross-asset: NDX→TQQQ vs SOXX→SOXL vs SPX→UPRO (same rules).

Usage:
  python backtest_kcm_cross.py
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf

from backtest_kcm_fills import COST
from kcm_sim import CASH_RATE, simulate
from lev_synth import PRESETS, build_preset, tracking_report

OUT = "results_kcm_cross.csv"
PAIRS = ("TQQQ", "SOXL", "UPRO")
LABEL = {"TQQQ": "NDX→TQQQ", "SOXL": "SOXX→SOXL", "UPRO": "SPX→UPRO"}

PERIODS = [
    ("2000+", "2000-01-01", None),
    ("2010+", "2010-03-11", None),
    ("2013+", "2013-01-01", None),
    ("닷컴", "2000-01-01", "2002-12-31"),
    ("08위기", "2007-01-01", "2009-12-31"),
    ("코로나", "2020-01-01", "2020-12-31"),
    ("2022", "2022-01-01", "2022-12-31"),
]


def _vix(idx):
    d = yf.download("^VIX", start="1990-01-01", auto_adjust=True, progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    return d["Close"].astype(float).reindex(idx).ffill()


def pack(preset: str):
    b = build_preset(preset, start="1985-01-01")
    px = b["signal"]
    veh = b["vehicle"].reindex(px.index)
    return (
        px["Close"].astype(float),
        px["Open"].astype(float).fillna(px["Close"]),
        px["High"].astype(float).fillna(px[["Open", "Close"]].max(axis=1)),
        px["Low"].astype(float).fillna(px[["Open", "Close"]].min(axis=1)),
        veh["Close"].ffill(),
        veh["Open"].fillna(veh["Close"]).ffill(),
        _vix(px.index),
        b,
    )


def main() -> None:
    print("Synth tracking")
    for p in PAIRS:
        b = build_preset(p)
        t0 = str(b["real"].index[0].date())
        r = tracking_report(b["synth"]["Close"], b["real"]["Close"], t0, None)
        print(f"  {p}: drift={r['drift_ann']:+.2%} TE={r['te_ann']:.2%} maxdev={r['max_path_dev']:.1%} from {t0}")

    data = {LABEL[p]: pack(p) for p in PAIRS}
    rows = []
    for vt_name, vt in [("VT없음", None), ("VT25", 0.25)]:
        print("\n" + "=" * 120)
        print(f"KCM | {vt_name} | 세후 | RP{CASH_RATE:.2%} | trim50% | {COST:.0f}bps")
        print("=" * 120)
        print(f"{'구간':<10}" + "".join(f"{lab:^20}" for lab in data))
        for pn, ps, pe in PERIODS:
            cells = []
            for lab, args_meta in data.items():
                *args, _ = args_meta
                r = simulate(
                    *args, sma_fill="next_open", temp_fill="loc",
                    start=ps, end=pe, cost_bps=COST, vt_target=vt,
                )
                cells.append(f"{r['cagr']:+6.1%}/{r['mdd']:6.1%}")
                rows.append({
                    "pair": lab, "vt": vt_name, "period": pn,
                    "cagr": r["cagr"], "mdd": r["mdd"], "trades": r["trades"],
                })
            print(f"{pn:<10}" + "".join(f"{x:^20}" for x in cells))

    pd.DataFrame(rows).to_csv(OUT, index=False, float_format="%.6f")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
