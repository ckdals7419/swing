"""
KCM on SOXX → SOXL vs QQQ/NDX → TQQQ (same rules).

Defaults: SMA next-open, RSI/VIX LOC, overheat trim 50%, tax, RP 3.45%, 9bps.

Usage:
  python backtest_kcm_soxl.py
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf

from backtest_kcm_fills import COST, build_vehicle
from kcm_sim import CASH_RATE, simulate
from lev_synth import PRESETS, build_preset, tracking_report

OUT = "results_kcm_soxl.csv"

PERIODS = [
    ("2001+", "2001-07-13", None),   # SOXX start
    ("2010+", "2010-03-11", None),   # SOXL start
    ("2013+", "2013-01-01", None),
    ("닷컴잔여", "2001-07-13", "2002-12-31"),
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
    b = build_preset(preset, start="1990-01-01")
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
    rows = []
    print("Calibration check (synth vs real SOXL)")
    b = build_preset("SOXL")
    rep = tracking_report(b["synth"]["Close"], b["real"]["Close"], "2010-03-11", None)
    print(f"  full: drift={rep['drift_ann']:+.2%} TE={rep['te_ann']:.2%} maxdev={rep['max_path_dev']:.1%}")
    print(f"  params: {PRESETS['SOXL']}")

    datasets = {
        "NDX→TQQQ": pack("TQQQ"),
        "SOXX→SOXL": pack("SOXL"),
    }

    for vt_name, vt in [("VT없음", None), ("VT25", 0.25)]:
        print("\n" + "=" * 110)
        print(f"KCM | {vt_name} | 세후 | RP{CASH_RATE:.2%} | trim50% | {COST:.0f}bps")
        print("=" * 110)
        print(f"{'구간':<10}" + "".join(f"{lab:^22}" for lab in datasets))
        for pn, ps, pe in PERIODS:
            cells = []
            for lab, data in datasets.items():
                *args, _meta = data
                r = simulate(
                    *args,
                    sma_fill="next_open",
                    temp_fill="loc",
                    start=ps,
                    end=pe,
                    cost_bps=COST,
                    vt_target=vt,
                )
                cells.append(f"{r['cagr']:+7.1%} / {r['mdd']:6.1%}")
                rows.append({
                    "pair": lab, "vt": vt_name, "period": pn,
                    "cagr": r["cagr"], "mdd": r["mdd"], "trades": r["trades"],
                    "cash_rate": CASH_RATE, "cost_bps": COST,
                })
            print(f"{pn:<10}" + "".join(f"{x:^22}" for x in cells))

    pd.DataFrame(rows).to_csv(OUT, index=False, float_format="%.6f")
    print(f"\nSaved {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
