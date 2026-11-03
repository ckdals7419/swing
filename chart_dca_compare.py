"""
KCM QLD Strategy — Detailed Visualization

KCM QLD (baseline):
  - SMA200 + SMA50 recovery gate
  - Anti-whipsaw buffer: exit at -1%, re-enter at +1%
  - VIX spike re-entry: 100% when VIX > 35 & declining (< SMA10)
  - RSI dip buy: 100% when RSI < 30 while out of position
  - (RSI/VIX do not flip in_position; binary 100/0 only)

KCM QLD Vol-target 30%:
  - KCM QLD gate logic + vol-target sizing: weight = min(1.0, 0.30 / (2 * vol))

Cash earns RP 3.25% annualized.

Usage:
    python chart_dca_compare.py
    python chart_dca_compare.py --start 2007-01-01
"""

import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

from strategy_qld import backtest, _nav_metrics, COST_BPS

FEATURES_PATH = "data/qld_features_extended.parquet"
DEFAULT_START = "2007-01-01"
RP_ANNUAL     = 0.0325
TARGET_VOL    = 0.30


def load_ext(start, end=None):
    df = pd.read_parquet(FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"])
    if start: df = df[df["date"] >= start]
    if end:   df = df[df["date"] <= end]
    df = df.set_index("date").sort_index().dropna(subset=["qqq_vs_sma200"])
    df["bil_return_1d"] = RP_ANNUAL / 252.0
    return df


def w_kcm_qld(df):
    """
    KCM QLD baseline strategy:
      1. SMA200 + SMA50 recovery gate
      2. Anti-whipsaw buffer: exit at QQQ < 0.99*SMA200, re-enter at QQQ > 1.01*SMA200
      3. VIX spike re-entry: 100% when VIX > 35 & VIX < VIX_SMA10
      4. RSI dip buy: 100% when RSI < 30 while out of position
         (RSI/VIX are binary overlays; they do not set in_position)
    """
    vs200 = df["qqq_vs_sma200"].fillna(0)
    vs50  = df["qqq_vs_sma50"].fillna(0)
    vix   = df["vix_close"].ffill().fillna(20)
    vix_sma10 = vix.rolling(10).mean()
    rsi   = df["rsi_14"].ffill().fillna(50)

    n = len(df)
    w = np.zeros(n)
    in_position = vs200.iloc[0] >= 0
    w[0] = 1.0 if in_position else 0.0

    for i in range(1, n):
        vix_reentry = (
            vix.iloc[i] > 35 and
            pd.notna(vix_sma10.iloc[i]) and
            vix.iloc[i] < vix_sma10.iloc[i]
        )

        if in_position:
            # Exit: buffer -1% below SMA200 AND below SMA50
            if vs200.iloc[i] < -0.01 and vs50.iloc[i] < 0:
                in_position = False
                if vix_reentry or rsi.iloc[i] < 30:
                    w[i] = 1.0
                else:
                    w[i] = 0.0
            else:
                w[i] = 1.0
        else:
            # Re-enter: +1% above SMA200, or SMA50 recovery
            if vs200.iloc[i] >= 0.01:
                in_position = True
                w[i] = 1.0
            elif vs50.iloc[i] >= 0:
                in_position = True
                w[i] = 1.0
            elif vix_reentry or rsi.iloc[i] < 30:
                w[i] = 1.0
            else:
                w[i] = 0.0

    return pd.Series(w, index=df.index)


def w_kcm_qld_voltarget(df, target_vol=TARGET_VOL):
    """
    KCM QLD + Vol-target sizing:
      Same gate logic as KCM QLD, but position size scaled by vol-target.
    """
    vs200 = df["qqq_vs_sma200"].fillna(0)
    vs50  = df["qqq_vs_sma50"].fillna(0)
    vix   = df["vix_close"].ffill().fillna(20)
    vix_sma10 = vix.rolling(10).mean()
    rsi   = df["rsi_14"].ffill().fillna(50)
    rv    = df["realized_vol_20d"].ffill().fillna(0.20)

    n = len(df)
    w = np.zeros(n)
    in_position = vs200.iloc[0] >= 0
    w[0] = 1.0 if in_position else 0.0

    for i in range(1, n):
        vix_reentry = (
            vix.iloc[i] > 35 and
            pd.notna(vix_sma10.iloc[i]) and
            vix.iloc[i] < vix_sma10.iloc[i]
        )

        # Vol-target sizing
        qld_vol = 2.0 * rv.iloc[i]
        vt_w = min(1.0, target_vol / qld_vol) if qld_vol > 0 else 1.0

        if in_position:
            if vs200.iloc[i] < -0.01 and vs50.iloc[i] < 0:
                in_position = False
                if vix_reentry or rsi.iloc[i] < 30:
                    w[i] = vt_w
                else:
                    w[i] = 0.0
            else:
                w[i] = vt_w
        else:
            if vs200.iloc[i] >= 0.01:
                in_position = True
                w[i] = vt_w
            elif vs50.iloc[i] >= 0:
                in_position = True
                w[i] = vt_w
            elif vix_reentry or rsi.iloc[i] < 30:
                w[i] = vt_w
            else:
                w[i] = 0.0

    return pd.Series(w, index=df.index)


def run(start=DEFAULT_START, end=None):
    print(f"Loading extended features {start} -> {end or 'latest'}...")
    df = load_ext(start, end)
    print(f"  {len(df):,} trading days  {df.index[0].date()} -> {df.index[-1].date()}")

    w_base = w_kcm_qld(df)
    w_vt   = w_kcm_qld_voltarget(df)

    nav_base = backtest(df, w_base, vehicle_open=df["qld_open"], vehicle_close=df["qld_close"])
    nav_vt   = backtest(df, w_vt,   vehicle_open=df["qld_open"], vehicle_close=df["qld_close"])

    m_base = _nav_metrics(nav_base, w_base)
    m_vt   = _nav_metrics(nav_vt,   w_vt)

    # Applied weights (T-1 signal)
    w_base_applied = w_base.shift(1).fillna(0)
    w_vt_applied   = w_vt.shift(1).fillna(0)
    rv = df["realized_vol_20d"].ffill().fillna(0.20)

    # Print table
    print(f"\n{'='*90}")
    print(f"  {'Strategy':<30} {'Ann.Ret':>8} {'Total':>12} {'MDD':>9} {'Sharpe':>7} {'Calmar':>7} {'AvgWt':>6}")
    print(f"  {'-'*80}")
    print(f"  {'KCM QLD':<30} {m_base['ann']:>8.2%} {m_base['total']:>12.1%} {m_base['mdd']:>9.2%} {m_base['sharpe']:>7.3f} {m_base['calmar']:>7.3f} {m_base['avg_w']:>6.1%}")
    print(f"  {'KCM QLD Vol-target 30%':<30} {m_vt['ann']:>8.2%} {m_vt['total']:>12.1%} {m_vt['mdd']:>9.2%} {m_vt['sharpe']:>7.3f} {m_vt['calmar']:>7.3f} {m_vt['avg_w']:>6.1%}")
    print(f"{'='*90}")

    # Current status
    last_w_base = w_base.iloc[-1]
    last_w_vt   = w_vt.iloc[-1]
    last_rv     = rv.iloc[-1]
    last_d      = df.index[-1].date()
    print(f"\n  Current ({last_d}):")
    print(f"    QQQ 20d vol:      {last_rv:.1%}")
    print(f"    QLD eff. vol:     {2*last_rv:.1%}")
    print(f"    KCM QLD weight:   {last_w_base:.0%}")
    print(f"    KCM VT weight:    {last_w_vt:.1%}")

    # ── Chart ────────────────────────────────────────────────────────────────
    BG   = "#0e1117"
    GRID = "#1e1e2e"
    TXT  = "#cccccc"

    fig, axes = plt.subplots(5, 1, figsize=(22, 20),
                             gridspec_kw={"height_ratios": [3.5, 1.0, 1.0, 1.0, 0.8]},
                             sharex=True)
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(BG)
        ax.tick_params(colors=TXT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#333344")

    yr_min = int(df.index[0].year) + 1
    yr_max = int(df.index[-1].year) + 1

    def add_year_lines(ax):
        for yr in range(yr_min, yr_max + 1):
            ax.axvline(pd.Timestamp(f"{yr}-01-01"), color="#333355", lw=0.4, ls=":", alpha=0.4)

    # ── Panel 1: NAV ─────────────────────────────────────────────────────────
    ax1 = axes[0]

    nav_base_r = nav_base / nav_base.iloc[0] * 100
    nav_vt_r   = nav_vt   / nav_vt.iloc[0]   * 100

    # Shade: green = position, intensity = weight
    for i in range(1, len(df)):
        wval = w_base_applied.iloc[i]
        if wval > 0:
            ax1.axvspan(df.index[i-1], df.index[i],
                        alpha=0.03 + 0.05 * wval, color="#00dd88", lw=0)

    ax1.plot(nav_base_r.index, nav_base_r.values, color="#f0c040", lw=2.2,
             label=f"KCM QLD  Ann={m_base['ann']:.1%}  MDD={m_base['mdd']:.1%}  Calmar={m_base['calmar']:.2f}")
    ax1.plot(nav_vt_r.index, nav_vt_r.values, color="#4a9eda", lw=1.5, alpha=0.8,
             label=f"KCM QLD VT30%  Ann={m_vt['ann']:.1%}  MDD={m_vt['mdd']:.1%}  Calmar={m_vt['calmar']:.2f}")

    add_year_lines(ax1)
    ax1.set_ylabel("NAV (100 base)", color=TXT, fontsize=10)
    ax1.set_title(
        f"KCM QLD Strategy   cash=RP 3.25%   cost={COST_BPS:.0f}bps   "
        f"{df.index[0].date()} -> {df.index[-1].date()}",
        color="#eeeeee", fontsize=13, pad=12)
    ax1.legend(loc="upper left", fontsize=9,
               facecolor="#12121f", labelcolor=TXT, edgecolor="#444455", framealpha=0.9)
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax1.grid(axis="y", color=GRID, lw=0.5)

    # ── Panel 2: Drawdown ────────────────────────────────────────────────────
    ax2 = axes[1]
    for nav_s, color, lw, alpha, lbl in [
        (nav_base, "#f0c040", 2.0, 0.9, f"KCM QLD  MDD={m_base['mdd']:.1%}"),
        (nav_vt,   "#4a9eda", 1.2, 0.7, f"KCM QLD VT30%  MDD={m_vt['mdd']:.1%}"),
    ]:
        peak = nav_s.expanding().max()
        dd = (nav_s - peak) / peak
        ax2.plot(dd.index, dd.values, color=color, lw=lw, alpha=alpha, label=lbl)
    ax2.set_ylabel("Drawdown", color=TXT, fontsize=9)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax2.legend(loc="lower left", fontsize=8,
               facecolor="#12121f", labelcolor=TXT, edgecolor="#444455", framealpha=0.9)
    ax2.grid(axis="y", color=GRID, lw=0.5)
    add_year_lines(ax2)

    # ── Panel 3: Weight (applied) ────────────────────────────────────────────
    ax3 = axes[2]

    w_vals = w_base_applied.values
    w_idx  = w_base_applied.index

    ax3.fill_between(w_idx, w_vals, 0, alpha=0.4, color="#f0c040", lw=0)
    ax3.plot(w_idx, w_vals, color="#f0c040", lw=1.2, alpha=0.9, label="KCM QLD weight")

    # Also show VT weight
    ax3.plot(w_vt_applied.index, w_vt_applied.values, color="#4a9eda", lw=0.8,
             alpha=0.6, label="KCM VT30% weight")

    ax3.axhline(1.0, color="#ffffff", lw=0.6, ls="--", alpha=0.3, label="100%")
    ax3.axhline(1.0, color="#ff8844", lw=0.5, ls=":", alpha=0.2, label="100% (RSI/VIX)")

    ax3.set_ylabel("QLD\nWeight", color=TXT, fontsize=9)
    ax3.set_ylim(-0.05, 1.12)
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax3.legend(loc="lower right", fontsize=7.5, ncol=2,
               facecolor="#12121f", labelcolor=TXT, edgecolor="#444455", framealpha=0.9)
    ax3.grid(axis="y", color=GRID, lw=0.5)
    add_year_lines(ax3)

    # ── Panel 4: QQQ Realized Vol + effective QLD vol ────────────────────────
    ax4 = axes[3]

    qld_vol = 2.0 * rv

    ax4.fill_between(rv.index, qld_vol.values, 0, alpha=0.15, color="#ff5577", lw=0)
    ax4.plot(rv.index, rv.values, color="#4a9eda", lw=1.0, alpha=0.8, label="QQQ vol 20d")
    ax4.plot(qld_vol.index, qld_vol.values, color="#ff5577", lw=1.0, alpha=0.7, label="QLD eff. vol (2x)")
    ax4.axhline(TARGET_VOL, color="#f0c040", lw=1.5, ls="--", alpha=0.9,
                label=f"Target = {TARGET_VOL:.0%}")

    for i in range(1, len(df)):
        if qld_vol.iloc[i] > TARGET_VOL:
            ax4.axvspan(df.index[i-1], df.index[i], alpha=0.06, color="#ff5577", lw=0)

    ax4.set_ylabel("Volatility\n(annualized)", color=TXT, fontsize=9)
    ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax4.set_ylim(0, 1.0)
    ax4.legend(loc="upper right", fontsize=8,
               facecolor="#12121f", labelcolor=TXT, edgecolor="#444455", framealpha=0.9)
    ax4.grid(axis="y", color=GRID, lw=0.5)
    add_year_lines(ax4)

    # ── Panel 5: QQQ vs SMA200 (regime) ──────────────────────────────────────
    ax5 = axes[4]
    vs200_s = df["qqq_vs_sma200"].fillna(0)
    ax5.axhline(0, color="#ffffff", lw=0.6, alpha=0.3)
    ax5.fill_between(vs200_s.index, vs200_s.values, 0,
                     where=(vs200_s >= 0), alpha=0.25, color="#00dd88", lw=0)
    ax5.fill_between(vs200_s.index, vs200_s.values, 0,
                     where=(vs200_s < 0), alpha=0.25, color="#ff4444", lw=0)
    ax5.plot(vs200_s.index, vs200_s.values, color="#bbbbbb", lw=0.6, alpha=0.7)
    ax5.set_ylabel("QQQ vs\nSMA200", color=TXT, fontsize=9)
    ax5.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax5.grid(axis="y", color=GRID, lw=0.5)
    add_year_lines(ax5)

    ax5.xaxis.set_major_locator(mdates.YearLocator(2))
    ax5.xaxis.set_minor_locator(mdates.YearLocator())
    ax5.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=0, ha="center", fontsize=9)

    plt.tight_layout(h_pad=0.3)
    out = "chart_dca_compare.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default=DEFAULT_START)
    parser.add_argument("--end",   type=str, default=None)
    args = parser.parse_args()
    run(start=args.start, end=args.end)
