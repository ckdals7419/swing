"""
KCM parameter sensitivity heatmaps (NDX→TQQQ, default fills/tax/RP/trim).

Grids (baseline held fixed when not on axes):
  RSI 1D, trim_on × trim_w, trim_on × trim_gap, BUF 1D

VIX temp hold was removed from baseline; RSI×VIX grid retired.

Usage:
  python backtest_kcm_sensitivity.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest_kcm_fills import COST, build_vehicle
from kcm_sim import CASH_RATE, simulate
from notify_kcm_telegram import BUF, RSI_DIP, TRIM_OFF, TRIM_ON, TRIM_W

OUT_CSV = "results_kcm_sensitivity.csv"
OUT_DIR = Path("sensitivity_heatmaps")
PERIODS = [("2000+", "2000-01-01", None), ("2013+", "2013-01-01", None)]


def run_one(data, *, start, end=None, **kw):
    return simulate(
        *data,
        sma_fill="next_open",
        temp_fill="loc",
        start=start,
        end=end,
        cost_bps=COST,
        cash_rate=CASH_RATE,
        **kw,
    )


def heatmap(ax, mat, xs, ys, title, fmt="{:.0%}", vmin=None, vmax=None, cmap="RdYlGn"):
    im = ax.imshow(mat, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(xs)))
    ax.set_yticks(range(len(ys)))
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_yticklabels([str(y) for y in ys])
    ax.set_title(title, fontsize=10)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


def neighbor_std(mat: np.ndarray) -> float:
    diffs = []
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            for di, dj in ((0, 1), (1, 0)):
                ii, jj = i + di, j + dj
                if ii < mat.shape[0] and jj < mat.shape[1]:
                    a, b = mat[i, j], mat[ii, jj]
                    if np.isfinite(a) and np.isfinite(b):
                        diffs.append(a - b)
    return float(np.std(diffs)) if diffs else float("nan")


def main() -> None:
    print("Building dataset...")
    data = build_vehicle()
    rows = []
    OUT_DIR.mkdir(exist_ok=True)

    rsi_vals = [20, 22, 24, 26, 28, 30, 32]
    ton_vals = [0.10, 0.12, 0.15, 0.18, 0.20, 0.25]
    tw_vals = [0.15, 0.25, 0.35, 0.5, 0.7, 1.0]
    gap_vals = [0.02, 0.03, 0.05, 0.07, 0.10]
    buf_vals = [0.0, 0.005, 0.01, 0.015, 0.02]

    for pname, ps, pe in PERIODS:
        # --- RSI 1D ---
        cagr_r = np.full(len(rsi_vals), np.nan)
        mdd_r = np.full(len(rsi_vals), np.nan)
        for i, rsi in enumerate(rsi_vals):
            r = run_one(data, start=ps, end=pe, rsi_th=rsi)
            cagr_r[i], mdd_r[i] = r["cagr"], r["mdd"]
            rows.append({"period": pname, "grid": "rsi", "rsi": rsi,
                         "cagr": r["cagr"], "mdd": r["mdd"], "trades": r["trades"]})
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        axes[0].plot(rsi_vals, cagr_r, "o-", color="#2a6")
        axes[0].axvline(RSI_DIP, color="black", ls="--", lw=1)
        axes[0].set_title(f"{pname} CAGR  RSI th")
        axes[0].set_xlabel("RSI th"); axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        axes[1].plot(rsi_vals, mdd_r, "o-", color="#c33")
        axes[1].axvline(RSI_DIP, color="black", ls="--", lw=1)
        axes[1].set_title(f"{pname} MDD  RSI th")
        axes[1].set_xlabel("RSI th"); axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        fig.suptitle(f"RSI 1D | baseline RSI={RSI_DIP:.0f} (VIX temp hold removed)", fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"rsi_{pname}.png", dpi=140)
        plt.close(fig)
        base_i = rsi_vals.index(RSI_DIP)
        print(f"RSI {pname}: base={cagr_r[base_i]:+.1%}  range={np.nanmin(cagr_r):+.1%}..{np.nanmax(cagr_r):+.1%}")

        # --- trim_on × trim_w ---
        cagr = np.full((len(ton_vals), len(tw_vals)), np.nan)
        mdd = np.full_like(cagr, np.nan)
        for i, ton in enumerate(ton_vals):
            for j, tw in enumerate(tw_vals):
                gap = TRIM_ON - TRIM_OFF
                toff = ton - gap if ton - gap > 0 else ton / 2
                if tw >= 0.999:
                    r = run_one(data, start=ps, end=pe, trim=None)
                else:
                    r = run_one(data, start=ps, end=pe, trim_on=ton, trim_off=toff, trim_w=tw)
                cagr[i, j], mdd[i, j] = r["cagr"], r["mdd"]
                rows.append({"period": pname, "grid": "trim_on_w", "trim_on": ton, "trim_w": tw,
                             "trim_off": toff, "cagr": r["cagr"], "mdd": r["mdd"], "trades": r["trades"]})
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        heatmap(axes[0], cagr, [f"{w:.0%}" for w in tw_vals], [f"{x:.0%}" for x in ton_vals],
                f"{pname} CAGR  trim_on×w", "{:.1%}", vmin=np.nanmin(cagr), vmax=np.nanmax(cagr))
        axes[0].set_xlabel("trim weight"); axes[0].set_ylabel("trim ON (vs200)")
        heatmap(axes[1], mdd, [f"{w:.0%}" for w in tw_vals], [f"{x:.0%}" for x in ton_vals],
                f"{pname} MDD  trim_on×w", "{:.0%}", vmin=np.nanmin(mdd), vmax=np.nanmax(mdd), cmap="RdYlGn_r")
        axes[1].set_xlabel("trim weight"); axes[1].set_ylabel("trim ON (vs200)")
        bi, bj = ton_vals.index(TRIM_ON), tw_vals.index(TRIM_W)
        for ax in axes:
            ax.scatter([bj], [bi], s=80, facecolors="none", edgecolors="black", linewidths=1.5)
        fig.suptitle(f"trim ON×weight | neighbor CAGR σ={neighbor_std(cagr):.2%}", fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"trim_on_w_{pname}.png", dpi=140)
        plt.close(fig)
        print(f"trim_on×w {pname}: CAGR σ_nb={neighbor_std(cagr):.2%}  base={cagr[bi, bj]:+.1%}")

        # --- trim_on × gap (off = on - gap), weight=50% ---
        cagr = np.full((len(ton_vals), len(gap_vals)), np.nan)
        mdd = np.full_like(cagr, np.nan)
        for i, ton in enumerate(ton_vals):
            for j, gap in enumerate(gap_vals):
                toff = ton - gap
                if toff <= 0:
                    continue
                r = run_one(data, start=ps, end=pe, trim_on=ton, trim_off=toff, trim_w=TRIM_W)
                cagr[i, j], mdd[i, j] = r["cagr"], r["mdd"]
                rows.append({"period": pname, "grid": "trim_on_gap", "trim_on": ton, "trim_gap": gap,
                             "trim_off": toff, "cagr": r["cagr"], "mdd": r["mdd"], "trades": r["trades"]})
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        heatmap(axes[0], cagr, [f"{g:.0%}" for g in gap_vals], [f"{x:.0%}" for x in ton_vals],
                f"{pname} CAGR  trim_on×gap", "{:.1%}", vmin=np.nanmin(cagr), vmax=np.nanmax(cagr))
        axes[0].set_xlabel("hysteresis gap"); axes[0].set_ylabel("trim ON")
        heatmap(axes[1], mdd, [f"{g:.0%}" for g in gap_vals], [f"{x:.0%}" for x in ton_vals],
                f"{pname} MDD  trim_on×gap", "{:.0%}", vmin=np.nanmin(mdd), vmax=np.nanmax(mdd), cmap="RdYlGn_r")
        axes[1].set_xlabel("hysteresis gap"); axes[1].set_ylabel("trim ON")
        bi = ton_vals.index(TRIM_ON)
        bj = gap_vals.index(round(TRIM_ON - TRIM_OFF, 5))
        for ax in axes:
            ax.scatter([bj], [bi], s=80, facecolors="none", edgecolors="black", linewidths=1.5)
        fig.suptitle(f"trim ON×gap | neighbor CAGR σ={neighbor_std(cagr):.2%}", fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"trim_on_gap_{pname}.png", dpi=140)
        plt.close(fig)
        print(f"trim_on×gap {pname}: CAGR σ_nb={neighbor_std(cagr):.2%}  base={cagr[bi, bj]:+.1%}")

        # --- BUF 1D ---
        for buf in buf_vals:
            r = run_one(data, start=ps, end=pe, buf=buf)
            rows.append({"period": pname, "grid": "buf", "buf": buf,
                         "cagr": r["cagr"], "mdd": r["mdd"], "trades": r["trades"]})
        sub = [row for row in rows if row.get("grid") == "buf" and row["period"] == pname]
        fig, ax = plt.subplots(figsize=(6, 3.5))
        xs = [row["buf"] for row in sub]
        ax.plot(xs, [row["cagr"] for row in sub], "o-", label="CAGR")
        ax2 = ax.twinx()
        ax2.plot(xs, [row["mdd"] for row in sub], "s--", color="C3", label="MDD")
        ax.axvline(BUF, color="k", lw=1)
        ax.set_xlabel("SMA buffer"); ax.set_ylabel("CAGR"); ax2.set_ylabel("MDD")
        ax.set_title(f"{pname} SMA ±buffer (baseline {BUF:.1%})")
        ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
        ax2.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"buf_{pname}.png", dpi=140)
        plt.close(fig)

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False, float_format="%.6f")
    print(f"\nSaved {OUT_CSV} and PNGs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
