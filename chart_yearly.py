"""
QLD Vol-target 30% Strategy — Yearly Breakdown Charts

Generates one PNG per year showing:
  Panel 1: NAV (strategy vs buy-and-hold QLD) + position shading
  Panel 2: Drawdown
  Panel 3: Applied weight
  Panel 4: QQQ realized vol + QLD effective vol vs target
  Panel 5: QQQ vs SMA200 regime

Plus a summary image with annual return bar chart.

Usage:
    python chart_yearly.py
    python chart_yearly.py --start 2000 --end 2025
"""

import warnings
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

from strategy_qld import backtest, _nav_metrics, COST_BPS
from chart_dca_compare import load_ext, w_baseline, w_voltarget, TARGET_VOL

BG   = "#0e1117"
GRID = "#1e1e2e"
TXT  = "#cccccc"
GOLD = "#f0c040"
GREEN = "#00dd88"
RED  = "#ff4444"
BLUE = "#4a9eda"
PINK = "#ff5577"

OUT_DIR = "chart_yearly"


def add_month_lines(ax, year):
    for m in range(2, 13):
        ax.axvline(pd.Timestamp(f"{year}-{m:02d}-01"),
                   color="#333355", lw=0.3, ls=":", alpha=0.4)


def format_ax(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#333344")


def make_yearly_chart(year, df_full, nav_full_vt, nav_full_base, w_vt_full, w_base_full):
    """Generate a single year's 5-panel chart."""
    start = pd.Timestamp(f"{year}-01-01")
    end   = pd.Timestamp(f"{year}-12-31")

    mask = (df_full.index >= start) & (df_full.index <= end)
    if mask.sum() < 5:
        return None

    df = df_full.loc[mask]
    nav_vt = nav_full_vt.loc[mask]
    nav_base = nav_full_base.loc[mask]
    w_vt = w_vt_full.loc[mask]
    w_base = w_base_full.loc[mask]

    # Normalize NAV to 100 at start of year
    nav_vt_r   = nav_vt / nav_vt.iloc[0] * 100
    nav_base_r = nav_base / nav_base.iloc[0] * 100

    # Metrics for this year
    m_vt   = _nav_metrics(nav_vt, w_vt)
    m_base = _nav_metrics(nav_base, w_base)

    # Applied weight (T-1 signal)
    w_vt_applied = w_vt.shift(1).bfill()
    rv = df["realized_vol_20d"].ffill().fillna(0.20)

    # Buy/sell signals: detect weight transitions
    w_full_applied = w_vt_full.shift(1).fillna(0)
    w_prev = w_full_applied.shift(1).fillna(0)
    w_delta_full = w_full_applied - w_prev

    w_delta = w_delta_full.loc[mask]
    buys  = df.index[w_delta > 0.05]
    sells = df.index[w_delta < -0.05]

    # QQQ return for context
    qqq_start = df["qqq_close"].iloc[0]
    qqq_end   = df["qqq_close"].iloc[-1]
    qqq_ret   = qqq_end / qqq_start - 1

    # ── Chart ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(5, 1, figsize=(18, 16),
                             gridspec_kw={"height_ratios": [3.5, 1.0, 1.0, 1.0, 0.8]},
                             sharex=True)
    fig.patch.set_facecolor(BG)
    for ax in axes:
        format_ax(ax)

    # ── Panel 1: NAV ─────────────────────────────────────────────────────────
    ax1 = axes[0]

    # Position shading
    for i in range(1, len(df)):
        wval = w_vt_applied.iloc[i]
        if wval > 0:
            ax1.axvspan(df.index[i-1], df.index[i],
                        alpha=0.04 + 0.06 * wval, color=GREEN, lw=0)

    ax1.plot(nav_base_r.index, nav_base_r.values, color="#555577", lw=1.0, alpha=0.5,
             label=f"Baseline (SMA gate)  Ret={m_base['ann']:.1%}  MDD={m_base['mdd']:.1%}")
    ax1.plot(nav_vt_r.index, nav_vt_r.values, color=GOLD, lw=2.2,
             label=f"Vol-target 30%  Ret={m_vt['ann']:.1%}  MDD={m_vt['mdd']:.1%}")

    # Buy/sell markers
    for b in buys:
        if b in nav_vt_r.index:
            ax1.axvline(b, color=GREEN, lw=0.8, alpha=0.5)
    for s in sells:
        if s in nav_vt_r.index:
            ax1.axvline(s, color=RED, lw=0.8, alpha=0.5)

    add_month_lines(ax1, year)
    ax1.set_ylabel("NAV (100 base)", color=TXT, fontsize=10)
    ax1.set_title(
        f"{year}  |  Vol-target: {m_vt['ann']:+.1%}   Baseline: {m_base['ann']:+.1%}   "
        f"QQQ: {qqq_ret:+.1%}   MDD(VT): {m_vt['mdd']:.1%}   Avg Wt: {m_vt['avg_w']:.0%}",
        color="#eeeeee", fontsize=13, pad=12)
    ax1.legend(loc="upper left", fontsize=9,
               facecolor="#12121f", labelcolor=TXT, edgecolor="#444455", framealpha=0.9)
    ax1.grid(axis="y", color=GRID, lw=0.5)

    # ── Panel 2: Drawdown ────────────────────────────────────────────────────
    ax2 = axes[1]
    for nav_s, color, lw, alpha, lbl in [
        (nav_base, "#555577", 1.0, 0.4, f"Baseline  MDD={m_base['mdd']:.1%}"),
        (nav_vt,   GOLD,     1.8, 0.9, f"Vol-target  MDD={m_vt['mdd']:.1%}"),
    ]:
        peak = nav_s.expanding().max()
        dd = (nav_s - peak) / peak
        ax2.fill_between(dd.index, dd.values, 0, alpha=alpha * 0.3, color=color, lw=0)
        ax2.plot(dd.index, dd.values, color=color, lw=lw, alpha=alpha, label=lbl)
    ax2.set_ylabel("Drawdown", color=TXT, fontsize=9)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax2.legend(loc="lower left", fontsize=8,
               facecolor="#12121f", labelcolor=TXT, edgecolor="#444455", framealpha=0.9)
    ax2.grid(axis="y", color=GRID, lw=0.5)
    add_month_lines(ax2, year)

    # ── Panel 3: Weight (applied) ────────────────────────────────────────────
    ax3 = axes[2]
    w_vals = w_vt_applied.values
    w_idx  = w_vt_applied.index
    ax3.fill_between(w_idx, w_vals, 0, alpha=0.4, color=GREEN, lw=0)
    ax3.plot(w_idx, w_vals, color=GREEN, lw=1.2, alpha=0.9, label="QLD weight (applied)")
    ax3.axhline(1.0, color="#ffffff", lw=0.6, ls="--", alpha=0.3)
    ax3.axhline(0.5, color="#ff8844", lw=0.5, ls=":", alpha=0.3)
    ax3.set_ylabel("QLD\nWeight", color=TXT, fontsize=9)
    ax3.set_ylim(-0.05, 1.12)
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax3.legend(loc="lower right", fontsize=7.5,
               facecolor="#12121f", labelcolor=TXT, edgecolor="#444455", framealpha=0.9)
    ax3.grid(axis="y", color=GRID, lw=0.5)
    add_month_lines(ax3, year)

    # ── Panel 4: Volatility ─────────────────────────────────────────────────
    ax4 = axes[3]
    qld_vol = 2.0 * rv
    ax4.fill_between(rv.index, qld_vol.values, 0, alpha=0.15, color=PINK, lw=0)
    ax4.plot(rv.index, rv.values, color=BLUE, lw=1.0, alpha=0.8, label="QQQ vol 20d")
    ax4.plot(qld_vol.index, qld_vol.values, color=PINK, lw=1.0, alpha=0.7, label="QLD eff. vol (2x)")
    ax4.axhline(TARGET_VOL, color=GOLD, lw=1.5, ls="--", alpha=0.9, label=f"Target = {TARGET_VOL:.0%}")

    for i in range(1, len(df)):
        if qld_vol.iloc[i] > TARGET_VOL:
            ax4.axvspan(df.index[i-1], df.index[i], alpha=0.06, color=PINK, lw=0)

    ax4.set_ylabel("Volatility\n(annualized)", color=TXT, fontsize=9)
    ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax4.set_ylim(0, min(1.0, qld_vol.max() * 1.3))
    ax4.legend(loc="upper right", fontsize=8,
               facecolor="#12121f", labelcolor=TXT, edgecolor="#444455", framealpha=0.9)
    ax4.grid(axis="y", color=GRID, lw=0.5)
    add_month_lines(ax4, year)

    # ── Panel 5: QQQ vs SMA200 ──────────────────────────────────────────────
    ax5 = axes[4]
    vs200 = df["qqq_vs_sma200"].fillna(0)
    ax5.axhline(0, color="#ffffff", lw=0.6, alpha=0.3)
    ax5.fill_between(vs200.index, vs200.values, 0,
                     where=(vs200 >= 0), alpha=0.25, color=GREEN, lw=0)
    ax5.fill_between(vs200.index, vs200.values, 0,
                     where=(vs200 < 0), alpha=0.25, color=RED, lw=0)
    ax5.plot(vs200.index, vs200.values, color="#bbbbbb", lw=0.6, alpha=0.7)
    ax5.set_ylabel("QQQ vs\nSMA200", color=TXT, fontsize=9)
    ax5.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax5.grid(axis="y", color=GRID, lw=0.5)
    add_month_lines(ax5, year)

    ax5.xaxis.set_major_locator(mdates.MonthLocator())
    ax5.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=0, ha="center", fontsize=9)

    plt.tight_layout(h_pad=0.3)
    out = os.path.join(OUT_DIR, f"year_{year}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return dict(year=year, vt_ret=m_vt['ann'], base_ret=m_base['ann'], qqq_ret=qqq_ret,
                mdd_vt=m_vt['mdd'], mdd_base=m_base['mdd'],
                sharpe_vt=m_vt['sharpe'], calmar_vt=m_vt['calmar'],
                avg_w=m_vt['avg_w'])


def make_summary_chart(records):
    """Bar chart of annual returns: strategy vs baseline vs QQQ."""
    records = sorted(records, key=lambda r: r['year'])
    years   = [r['year'] for r in records]
    vt_rets = [r['vt_ret'] * 100 for r in records]
    base_rets = [r['base_ret'] * 100 for r in records]
    qqq_rets  = [r['qqq_ret'] * 100 for r in records]
    mdds      = [r['mdd_vt'] * 100 for r in records]

    x = np.arange(len(years))
    w = 0.25

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(18, len(years) * 0.9), 14),
                                    gridspec_kw={"height_ratios": [2, 1]})
    fig.patch.set_facecolor(BG)
    for ax in [ax1, ax2]:
        format_ax(ax)

    # ── Top: Annual returns ──────────────────────────────────────────────────
    bars_vt   = ax1.bar(x - w, vt_rets,   w, color=GOLD,     alpha=0.85, label="Vol-target 30%", edgecolor="#333344", lw=0.5)
    bars_base = ax1.bar(x,     base_rets, w, color="#555577", alpha=0.65, label="Baseline (SMA gate)", edgecolor="#333344", lw=0.5)
    bars_qqq  = ax1.bar(x + w, qqq_rets,  w, color=BLUE,     alpha=0.60, label="QQQ B&H", edgecolor="#333344", lw=0.5)

    # Value labels on vol-target bars
    for bar, val in zip(bars_vt, vt_rets):
        y = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, y + (1.5 if y >= 0 else -3.5),
                 f"{val:+.0f}%", ha="center", va="bottom" if y >= 0 else "top",
                 color=GOLD, fontsize=7, fontweight="bold")

    ax1.axhline(0, color="#ffffff", lw=0.5, alpha=0.3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(years, fontsize=9, color=TXT)
    ax1.set_ylabel("Annual Return (%)", color=TXT, fontsize=11)
    ax1.set_title("QLD Vol-target 30% — Annual Returns by Year", color="#eeeeee", fontsize=14, pad=12)
    ax1.legend(loc="upper left", fontsize=9,
               facecolor="#12121f", labelcolor=TXT, edgecolor="#444455", framealpha=0.9)
    ax1.grid(axis="y", color=GRID, lw=0.5)

    # ── Bottom: MDD per year ─────────────────────────────────────────────────
    colors = [RED if m < -20 else "#ff8844" if m < -10 else GREEN for m in mdds]
    ax2.bar(x, mdds, 0.5, color=colors, alpha=0.7, edgecolor="#333344", lw=0.5)
    for i, (xi, m) in enumerate(zip(x, mdds)):
        ax2.text(xi, m - 1.5, f"{m:.0f}%", ha="center", va="top",
                 color=TXT, fontsize=7.5, fontweight="bold")

    ax2.axhline(0, color="#ffffff", lw=0.5, alpha=0.3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(years, fontsize=9, color=TXT)
    ax2.set_ylabel("Max Drawdown (%)", color=TXT, fontsize=11)
    ax2.set_title("Max Drawdown by Year (Vol-target)", color="#eeeeee", fontsize=12, pad=8)
    ax2.grid(axis="y", color=GRID, lw=0.5)

    plt.tight_layout(h_pad=1.0)
    out = os.path.join(OUT_DIR, "summary_annual.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return out


def run(start_year=2000, end_year=2026):
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading extended features...")
    df = load_ext(f"{start_year}-01-01")
    print(f"  {len(df):,} trading days  {df.index[0].date()} -> {df.index[-1].date()}")

    # Full-period weights and NAV
    w_base = w_baseline(df)
    w_vt   = w_voltarget(df)

    nav_base = backtest(df, w_base, vehicle_open=df["qld_open"], vehicle_close=df["qld_close"])
    nav_vt   = backtest(df, w_vt,   vehicle_open=df["qld_open"], vehicle_close=df["qld_close"])

    actual_start = df.index[0].year
    actual_end   = min(end_year, df.index[-1].year)

    records = []
    for year in range(actual_start, actual_end + 1):
        print(f"  Generating {year}...", end=" ")
        rec = make_yearly_chart(year, df, nav_vt, nav_base, w_vt, w_base)
        if rec:
            records.append(rec)
            print(f"VT={rec['vt_ret']:+.1%}  Base={rec['base_ret']:+.1%}  "
                  f"QQQ={rec['qqq_ret']:+.1%}  MDD={rec['mdd_vt']:.1%}")
        else:
            print("skipped (too few days)")

    # Summary chart
    print(f"\n  Generating summary...")
    summary_path = make_summary_chart(records)

    print(f"\nDone! {len(records)} yearly charts + summary saved to {OUT_DIR}/")
    print(f"  Summary: {summary_path}")

    # Print table
    print(f"\n{'Year':>6} {'VT Ret':>8} {'Base Ret':>9} {'QQQ':>8} {'MDD(VT)':>8} {'Sharpe':>7} {'Calmar':>7} {'AvgWt':>6}")
    print("-" * 70)
    for r in records:
        print(f"{r['year']:>6} {r['vt_ret']:>8.1%} {r['base_ret']:>9.1%} {r['qqq_ret']:>8.1%} "
              f"{r['mdd_vt']:>8.1%} {r['sharpe_vt']:>7.2f} {r['calmar_vt']:>7.2f} {r['avg_w']:>6.0%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=2000)
    parser.add_argument("--end",   type=int, default=2026)
    args = parser.parse_args()
    run(start_year=args.start, end_year=args.end)
