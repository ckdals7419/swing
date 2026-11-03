"""
SMA-VIX-Buffer + RSI Dip Buy strategy video simulation.

Generates an MP4 showing day-by-day portfolio growth with:
  - NAV curve building in real time
  - QLD price panel
  - Weight (position) panel
  - Live stats overlay (NAV, return, weight, RSI, regime)

Usage:
    python video_sma_vix_buffer.py
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.animation as animation

warnings.filterwarnings("ignore")

from strategy_qld import backtest, _nav_metrics
from chart_dca_compare import load_ext

# ── Config ──────────────────────────────────────────────────────────────────
START      = "2021-01-01"
FPS        = 30
OUT_FILE   = "simulation_sma_vix_rsi_2021.mp4"
RP_ANNUAL  = 0.0325

BG    = "#0e1117"
GRID  = "#1e1e2e"
TXT   = "#cccccc"
GOLD  = "#f0c040"
GREEN = "#00dd88"
RED   = "#ff4455"
BLUE  = "#4a9eda"
ORANGE = "#ff8844"


def w_sma_vix_buffer_rsi(df):
    """SMA-VIX-Buffer + RSI/VIX binary overlay (100% or 0%)."""
    vs200 = df["qqq_vs_sma200"].fillna(0).values
    vs50  = df["qqq_vs_sma50"].fillna(0).values
    vix   = df["vix_close"].ffill().fillna(20).values
    vix_sma10 = pd.Series(vix).rolling(10).mean().values
    rsi   = df["rsi_14"].ffill().fillna(50).values

    n = len(df)
    w = np.zeros(n)
    in_position = vs200[0] >= 0
    w[0] = 1.0 if in_position else 0.0

    for i in range(1, n):
        vix_reentry = (
            vix[i] > 35 and
            not np.isnan(vix_sma10[i]) and
            vix[i] < vix_sma10[i]
        )

        if in_position:
            if vs200[i] < -0.01 and vs50[i] < 0:
                in_position = False
                if vix_reentry or rsi[i] < 30:
                    w[i] = 1.0
                else:
                    w[i] = 0.0
            else:
                w[i] = 1.0
        else:
            if vs200[i] >= 0.01:
                in_position = True
                w[i] = 1.0
            elif vs50[i] >= 0:
                in_position = True
                w[i] = 1.0
            elif vix_reentry or rsi[i] < 30:
                w[i] = 1.0
            else:
                w[i] = 0.0

    return pd.Series(w, index=df.index)


def main():
    # ── Load & run full backtest ────────────────────────────────────────────
    print("Loading data...")
    df = load_ext(START)
    print(f"  {len(df):,} days  {df.index[0].date()} -> {df.index[-1].date()}")

    w = w_sma_vix_buffer_rsi(df)
    nav = backtest(df, w, vehicle_open=df["qld_open"], vehicle_close=df["qld_close"])

    # Derived series
    w_applied = w.shift(1).fillna(0)
    nav_vals  = nav.values
    dates     = df.index
    n         = len(df)

    qld_close = df["qld_close"].values
    vs200     = df["qqq_vs_sma200"].fillna(0).values
    vix_vals  = df["vix_close"].ffill().fillna(20).values
    rsi_vals  = df["rsi_14"].ffill().fillna(50).values

    initial_nav = nav_vals[0]

    print(f"  Initial NAV: ${initial_nav:,.0f}")
    print(f"  Final NAV:   ${nav_vals[-1]:,.0f}")
    print(f"  Return:      {(nav_vals[-1]/initial_nav - 1)*100:+.1f}%")
    print(f"  Creating {n} frame animation...")

    # ── Figure setup ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10), facecolor=BG)

    gs = fig.add_gridspec(3, 2, width_ratios=[4, 1.2],
                          height_ratios=[2.5, 1.0, 1.0],
                          hspace=0.25, wspace=0.15,
                          left=0.07, right=0.97, top=0.92, bottom=0.06)

    ax_nav    = fig.add_subplot(gs[0, 0])
    ax_info   = fig.add_subplot(gs[0, 1])
    ax_weight = fig.add_subplot(gs[1, 0])
    ax_rsi_vix = fig.add_subplot(gs[1, 1])
    ax_qld    = fig.add_subplot(gs[2, 0])
    ax_regime = fig.add_subplot(gs[2, 1])

    for ax in [ax_nav, ax_weight, ax_qld]:
        ax.set_facecolor(BG)
        ax.tick_params(colors=TXT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#333344")
        ax.grid(axis="y", color=GRID, lw=0.5, alpha=0.5)

    for ax in [ax_info, ax_rsi_vix, ax_regime]:
        ax.set_facecolor("#0a0d12")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#222233")

    title_text = fig.suptitle("", color="#eeeeee", fontsize=14, fontweight="bold", y=0.97)

    # ── NAV panel ───────────────────────────────────────────────────────────
    ax_nav.set_ylabel("Portfolio NAV ($)", color=TXT, fontsize=9)
    nav_line,  = ax_nav.plot([], [], color=GOLD, lw=2.2)
    nav_dot,   = ax_nav.plot([], [], "o", color=GOLD, ms=6, zorder=5)
    nav_fill   = None

    # ── Weight panel ────────────────────────────────────────────────────────
    ax_weight.set_ylabel("QLD Weight", color=TXT, fontsize=9)
    ax_weight.set_ylim(-0.08, 1.15)
    ax_weight.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax_weight.axhline(1.0, color="#ffffff", lw=0.5, ls="--", alpha=0.2)
    ax_weight.axhline(0.0, color="#ffffff", lw=0.5, ls="--", alpha=0.2)
    weight_line, = ax_weight.plot([], [], color=GREEN, lw=1.5)

    # ── QLD panel ───────────────────────────────────────────────────────────
    ax_qld.set_ylabel("QLD Price ($)", color=TXT, fontsize=9)
    qld_line,  = ax_qld.plot([], [], color=BLUE, lw=1.2)
    qld_dot,   = ax_qld.plot([], [], "o", color=BLUE, ms=5, zorder=5)

    # ── Pre-compute limits ──────────────────────────────────────────────────
    nav_min = nav_vals.min() * 0.98
    nav_max = nav_vals.max() * 1.02
    qld_min = qld_close.min() * 0.95
    qld_max = qld_close.max() * 1.05

    # ── Animation ───────────────────────────────────────────────────────────
    def animate(frame):
        nonlocal nav_fill
        i = frame
        t = i + 1

        d_slice = dates[:t]
        n_slice = nav_vals[:t]
        w_slice = w_applied.values[:t]
        q_slice = qld_close[:t]

        cur_date  = dates[i]
        cur_nav   = nav_vals[i]
        cur_w     = w_applied.iloc[i]
        cur_qld   = qld_close[i]
        cur_vs200 = vs200[i]
        cur_vix   = vix_vals[i]
        cur_rsi   = rsi_vals[i]
        cur_ret   = (cur_nav / initial_nav - 1) * 100

        peak = n_slice.max()
        dd   = (cur_nav - peak) / peak * 100

        title_text.set_text(
            f"SMA-VIX-Buffer + RSI Dip Buy    {cur_date.strftime('%Y-%m-%d')}"
        )

        # ── NAV ─────────────────────────────────────────────────────────
        nav_line.set_data(d_slice, n_slice)
        nav_dot.set_data([cur_date], [cur_nav])
        ax_nav.set_xlim(dates[0], dates[-1])
        ax_nav.set_ylim(nav_min, nav_max)

        if nav_fill is not None:
            nav_fill.remove()
        color = GREEN if cur_ret >= 0 else RED
        nav_fill = ax_nav.fill_between(d_slice, n_slice, initial_nav,
                                       alpha=0.12, color=color, lw=0)

        # ── Weight ──────────────────────────────────────────────────────
        weight_line.set_data(d_slice, w_slice)
        ax_weight.set_xlim(dates[0], dates[-1])

        for coll in ax_weight.collections[:]:
            coll.remove()

        # Color by weight level
        w_arr = np.array(w_slice)
        ax_weight.fill_between(d_slice, w_arr, 0, alpha=0.3, color=GREEN, lw=0)

        # ── QLD ─────────────────────────────────────────────────────────
        qld_line.set_data(d_slice, q_slice)
        qld_dot.set_data([cur_date], [cur_qld])
        ax_qld.set_xlim(dates[0], dates[-1])
        ax_qld.set_ylim(qld_min, qld_max)

        # ── Info panel ──────────────────────────────────────────────────
        ax_info.clear()
        ax_info.set_facecolor("#0a0d12")
        ax_info.set_xticks([]); ax_info.set_yticks([])
        for spine in ax_info.spines.values():
            spine.set_color("#222233")

        info_lines = [
            ("NAV", f"${cur_nav:,.0f}", GOLD),
            ("Return", f"{cur_ret:+.1f}%", GREEN if cur_ret >= 0 else RED),
            ("Drawdown", f"{dd:.1f}%", RED if dd < -5 else TXT),
            ("", "", TXT),
            ("QLD Weight", f"{cur_w:.0%}", GREEN if cur_w >= 1.0 else (ORANGE if cur_w > 0 else "#666666")),
            ("QLD Price", f"${cur_qld:.2f}", BLUE),
            ("", "", TXT),
            ("QQQ vs SMA200", f"{cur_vs200:+.1%}", GREEN if cur_vs200 >= 0 else RED),
            ("RSI 14", f"{cur_rsi:.1f}", RED if cur_rsi < 30 else (ORANGE if cur_rsi > 70 else TXT)),
            ("VIX", f"{cur_vix:.1f}", RED if cur_vix > 30 else TXT),
        ]

        y_pos = 0.95
        for label, value, color in info_lines:
            if label == "":
                y_pos -= 0.03
                continue
            ax_info.text(0.08, y_pos, label, color="#888899", fontsize=9,
                        transform=ax_info.transAxes, va="top", fontfamily="monospace")
            ax_info.text(0.92, y_pos, value, color=color, fontsize=11, fontweight="bold",
                        transform=ax_info.transAxes, va="top", ha="right", fontfamily="monospace")
            y_pos -= 0.10

        # ── RSI / VIX gauges ───────────────────────────────────────────
        ax_rsi_vix.clear()
        ax_rsi_vix.set_facecolor("#0a0d12")
        ax_rsi_vix.set_xticks([]); ax_rsi_vix.set_yticks([])
        for spine in ax_rsi_vix.spines.values():
            spine.set_color("#222233")

        # RSI gauge (top half)
        ax_rsi_vix.text(0.25, 0.92, "RSI", color="#888899", fontsize=9,
                       transform=ax_rsi_vix.transAxes, ha="center", va="top")
        rsi_color = RED if cur_rsi < 30 else (ORANGE if cur_rsi > 70 else GREEN)
        ax_rsi_vix.text(0.25, 0.55, f"{cur_rsi:.0f}", color=rsi_color, fontsize=18,
                       fontweight="bold", transform=ax_rsi_vix.transAxes,
                       ha="center", va="center", fontfamily="monospace")
        rsi_label = "OVERSOLD" if cur_rsi < 30 else ("OVERBOUGHT" if cur_rsi > 70 else "NEUTRAL")
        ax_rsi_vix.text(0.25, 0.25, rsi_label, color=rsi_color, fontsize=7,
                       transform=ax_rsi_vix.transAxes, ha="center", va="center")

        # Highlight RSI<30 dip buy trigger
        if cur_rsi < 30 and cur_w == 0:
            ax_rsi_vix.text(0.25, 0.08, "DIP BUY!", color=RED, fontsize=9,
                           fontweight="bold", transform=ax_rsi_vix.transAxes,
                           ha="center", va="center")

        # VIX gauge (bottom half)
        ax_rsi_vix.text(0.75, 0.92, "VIX", color="#888899", fontsize=9,
                       transform=ax_rsi_vix.transAxes, ha="center", va="top")
        vix_color = GREEN if cur_vix < 15 else (GOLD if cur_vix < 25 else (
                    ORANGE if cur_vix < 35 else RED))
        ax_rsi_vix.text(0.75, 0.55, f"{cur_vix:.0f}", color=vix_color, fontsize=18,
                       fontweight="bold", transform=ax_rsi_vix.transAxes,
                       ha="center", va="center", fontfamily="monospace")
        vix_label = "LOW" if cur_vix < 15 else ("NORMAL" if cur_vix < 25 else (
                    "HIGH" if cur_vix < 35 else "EXTREME"))
        ax_rsi_vix.text(0.75, 0.25, vix_label, color=vix_color, fontsize=7,
                       transform=ax_rsi_vix.transAxes, ha="center", va="center")

        # Divider
        ax_rsi_vix.plot([0.5, 0.5], [0, 1], color="#333344", lw=0.5,
                       transform=ax_rsi_vix.transAxes, clip_on=False)

        # ── Regime panel ────────────────────────────────────────────────
        ax_regime.clear()
        ax_regime.set_facecolor("#0a0d12")
        ax_regime.set_xticks([]); ax_regime.set_yticks([])
        for spine in ax_regime.spines.values():
            spine.set_color("#222233")

        if cur_w >= 1.0:
            regime_txt = "FULL\nPOSITION"
            regime_sub = "100%"
            regime_color = GREEN
        elif cur_w >= 0.75:
            regime_txt = "RSI DIP\nBUY"
            regime_sub = "75%"
            regime_color = ORANGE
        elif cur_w > 0:
            regime_txt = "VIX\nRE-ENTRY"
            regime_sub = "50%"
            regime_color = GOLD
        else:
            regime_txt = "CASH"
            regime_sub = "0%"
            regime_color = "#666677"

        ax_regime.text(0.5, 0.65, regime_txt, color=regime_color, fontsize=14,
                      fontweight="bold", transform=ax_regime.transAxes,
                      ha="center", va="center", linespacing=1.4)
        ax_regime.text(0.5, 0.2, regime_sub, color=regime_color, fontsize=18,
                      fontweight="bold", transform=ax_regime.transAxes,
                      ha="center", va="center", fontfamily="monospace")

        return [nav_line, nav_dot, weight_line, qld_line, qld_dot, title_text]

    # ── Build animation ─────────────────────────────────────────────────────
    anim = animation.FuncAnimation(fig, animate, frames=n,
                                   interval=1000//FPS, blit=False)

    writer = animation.FFMpegWriter(fps=FPS, bitrate=3000,
                                     extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
    print(f"  Rendering {n} frames at {FPS} fps (~{n/FPS:.0f}s video)...")
    anim.save(OUT_FILE, writer=writer, dpi=120)
    plt.close()
    print(f"\nSaved: {OUT_FILE}")


if __name__ == "__main__":
    main()
