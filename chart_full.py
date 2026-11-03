"""
Walk-forward chart: 2015-2026, annual retraining.
Each year's signals use only data available before that year.
Now includes macro features (yield curve, credit spread, NFCI, etc.)
"""
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

from crossval_qld import (
    build_full_dataset,
    ML_FEATURES_BASE, CROSS_ASSET_ADDITIONS, REGIME_ADDITIONS, MACRO_ADDITIONS,
    BTC_XGB_PARAMS, NUM_BOOST_ROUND, EARLY_STOP_ROUNDS,
    w_sma200, backtest, compute_metrics,
)

# Only market-based macro features (0-day lag) are honest; lagged monthly
# indicators (UNRATE 35d, AMTMNO 45d) become stale noise.
# Use only zero-lag macro features alongside existing feature groups.
MACRO_REALTIME = [
    # These are market yields / prices — available same day, no publication lag
    "yc_10y2y", "yc_slope_20d", "yc_inversion_days",
    "cs_hy_level", "cs_hy_change_20d", "cs_hy_zscore_1y",
    "real_rate_10y", "real_rate_change_20d",
    "breakeven_10y", "breakeven_change_20d",
]

FEAT_LIST   = ML_FEATURES_BASE + CROSS_ASSET_ADDITIONS + REGIME_ADDITIONS + MACRO_REALTIME
BUY_THRESH  =  0.25
SELL_THRESH = -0.25
CHART_START = "2015-01-01"

# (predict_from, train_through)  — annual retraining, 30-day purge
RETRAIN_SCHEDULE = [
    ("2015-01-01", "2014-11-30"),
    ("2016-01-01", "2015-11-30"),
    ("2017-01-01", "2016-11-30"),
    ("2018-01-01", "2017-11-30"),
    ("2019-01-01", "2018-11-30"),
    ("2020-01-01", "2019-11-30"),
    ("2021-01-01", "2020-11-30"),
    ("2022-01-01", "2021-11-30"),
    ("2023-01-01", "2022-11-30"),
    ("2024-01-01", "2023-11-30"),
    ("2025-01-01", "2024-11-30"),
    ("2025-07-01", "2025-05-31"),
    ("2026-01-01", "2025-11-30"),
]


def train_for_period(full_df, train_end, feat_list):
    feat_cols   = [c for c in feat_list if c in full_df.columns]
    target_cols = ["target_bull", "target_bear"]
    train_df = full_df.loc[:train_end].dropna(subset=feat_cols + target_cols)
    n_val    = max(60, int(len(train_df) * 0.10))
    val_df   = train_df.iloc[-n_val:]
    tr_df    = train_df.iloc[:-n_val]
    X_tr  = tr_df[feat_cols].ffill().fillna(0.0)
    X_val = val_df[feat_cols].ffill().fillna(0.0)
    boosters = {}
    for target in ("target_bull", "target_bear"):
        y_tr  = tr_df[target]
        y_val = val_df[target]
        spw   = float((y_tr == 0).sum()) / max(1.0, float((y_tr == 1).sum()))
        params = {**BTC_XGB_PARAMS, "scale_pos_weight": spw}
        dtrain = xgb.DMatrix(X_tr,  label=y_tr,  feature_names=feat_cols)
        dval   = xgb.DMatrix(X_val, label=y_val, feature_names=feat_cols)
        booster = xgb.train(
            params, dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            evals=[(dtrain, "train"), (dval, "eval")],
            early_stopping_rounds=EARLY_STOP_ROUNDS,
            verbose_eval=False,
        )
        boosters[target] = booster
    return boosters["target_bull"], boosters["target_bear"], feat_cols


def compute_spread(df, bull_clf, bear_clf, feat_cols):
    X = df[[c for c in feat_cols if c in df.columns]].copy()
    for c in feat_cols:
        if c not in X.columns:
            X[c] = 0.0
    X = X[feat_cols].ffill().fillna(0.0)
    dmat = xgb.DMatrix(X, feature_names=feat_cols)
    bp   = pd.Series(bull_clf.predict(dmat), index=df.index)
    brp  = pd.Series(bear_clf.predict(dmat), index=df.index)
    return bp, brp, bp - brp


def weights_asym(spread, buy_t, sell_t):
    s = spread.values
    w = 1.0 if s[0] >= buy_t else 0.0
    out = np.empty(len(s))
    for i, v in enumerate(s):
        if v >= buy_t:
            w = 1.0
        elif v <= sell_t:
            w = 0.0
        out[i] = w
    return pd.Series(out, index=spread.index)


def main():
    full_df  = build_full_dataset()
    chart_df = full_df.loc[CHART_START:].copy()

    # ── Walk-forward spread computation ──────────────────────────────────────
    all_bull = pd.Series(dtype=float)
    all_bear = pd.Series(dtype=float)

    for i, (pred_from, train_through) in enumerate(RETRAIN_SCHEDULE):
        pred_end = RETRAIN_SCHEDULE[i+1][0] if i+1 < len(RETRAIN_SCHEDULE) else None
        seg = (chart_df.loc[pred_from:pred_end].copy()
               if pred_end else chart_df.loc[pred_from:].copy())
        if len(seg) == 0:
            continue
        print(f"  [{i+1:02d}/{len(RETRAIN_SCHEDULE)}] train<={train_through}  "
              f"predict {pred_from} -> {pred_end or 'latest'} ({len(seg)} rows)")
        bull_clf, bear_clf, feat_cols = train_for_period(full_df, train_through, FEAT_LIST)
        bp, brp, _ = compute_spread(seg, bull_clf, bear_clf, feat_cols)
        all_bull = pd.concat([all_bull, bp])
        all_bear = pd.concat([all_bear, brp])

    all_bull = all_bull[~all_bull.index.duplicated(keep="last")]
    all_bear = all_bear[~all_bear.index.duplicated(keep="last")]
    spread    = all_bull - all_bear
    weights   = weights_asym(spread, BUY_THRESH, SELL_THRESH)

    # ── NAV computation ───────────────────────────────────────────────────────
    nav_ml  = backtest(chart_df, weights)
    nav_sma = backtest(chart_df, w_sma200(chart_df))
    nav_bah = backtest(chart_df, pd.Series(1.0, index=chart_df.index))

    nav_ml  = nav_ml  / nav_ml.iloc[0]  * 100
    nav_sma = nav_sma / nav_sma.iloc[0] * 100
    nav_bah = nav_bah / nav_bah.iloc[0] * 100

    # ── Trade events ──────────────────────────────────────────────────────────
    w_diff     = weights.diff()
    buy_dates  = weights.index[w_diff ==  1.0]
    sell_dates = weights.index[w_diff == -1.0]

    # ── Metrics ───────────────────────────────────────────────────────────────
    m_ml  = compute_metrics(nav_ml  / 100 * 1e6, weights)
    m_sma = compute_metrics(nav_sma / 100 * 1e6, w_sma200(chart_df))
    m_bah = compute_metrics(nav_bah / 100 * 1e6)

    # ── Macro indicators for context panel ───────────────────────────────────
    yc   = chart_df["yc_10y2y"].ffill()          # yield curve 10Y-2Y
    nfci = chart_df["nfci_level"].ffill()         # Chicago Fed financial conditions
    cs   = chart_df["cs_hy_level"].ffill()        # HYG/LQD credit spread proxy
    rr   = chart_df["real_rate_10y"].ffill()      # 10Y TIPS real yield

    # ── Plot ──────────────────────────────────────────────────────────────────
    BG   = "#0e1117"
    GRID = "#1e1e2e"
    TXT  = "#cccccc"

    fig, axes = plt.subplots(4, 1, figsize=(20, 16),
                             gridspec_kw={"height_ratios": [3.5, 1.1, 1.2, 1.0]},
                             sharex=True)
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(BG)
        ax.tick_params(colors=TXT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#333344")

    # ── Panel 1: NAV ──────────────────────────────────────────────────────────
    ax1 = axes[0]

    # Position shading
    in_mask = weights.shift(1).fillna(0) == 1.0
    for i in range(1, len(chart_df)):
        if in_mask.iloc[i]:
            ax1.axvspan(chart_df.index[i-1], chart_df.index[i],
                        alpha=0.06, color="#00dd88", lw=0)

    ax1.plot(nav_bah.index, nav_bah.values, color="#444466", lw=1.0,
             alpha=0.6, label=f"QLD B&H   Ann={m_bah['ann']:.1%}  MDD={m_bah['mdd']:.1%}  Total={nav_bah.iloc[-1]-100:+.0f}%")
    ax1.plot(nav_sma.index, nav_sma.values, color="#4a9eda", lw=1.4,
             alpha=0.85, label=f"SMA200    Ann={m_sma['ann']:.1%}  MDD={m_sma['mdd']:.1%}  Cal={m_sma['calmar']:.2f}")
    ax1.plot(nav_ml.index,  nav_ml.values,  color="#f0c040", lw=2.0,
             label=f"ML (lag-corrected)  Ann={m_ml['ann']:.1%}  MDD={m_ml['mdd']:.1%}  Cal={m_ml['calmar']:.2f}")

    # Buy/sell markers
    b_nav = [nav_ml.loc[d] for d in buy_dates  if d in nav_ml.index]
    s_nav = [nav_ml.loc[d] for d in sell_dates if d in nav_ml.index]
    ax1.scatter([d for d in buy_dates  if d in nav_ml.index], b_nav,
                marker="^", color="#00ee88", s=50, zorder=5, label=f"Buy  ({len(buy_dates)})")
    ax1.scatter([d for d in sell_dates if d in nav_ml.index], s_nav,
                marker="v", color="#ff5555", s=50, zorder=5, label=f"Sell ({len(sell_dates)})")

    for yr in range(2016, 2027):
        ax1.axvline(pd.Timestamp(f"{yr}-01-01"), color="#333355", lw=0.6, ls=":", alpha=0.7)

    ax1.set_ylabel("NAV (rebased to 100)", color=TXT, fontsize=10)
    n_feat = len([c for c in FEAT_LIST if c in full_df.columns])
    ax1.set_title(
        f"QLD Walk-Forward ML  2015-2026   ({n_feat} features, buy>=+{BUY_THRESH} / sell<=-{abs(SELL_THRESH)}, lag-corrected)",
        color="#eeeeee", fontsize=13, pad=10
    )
    ax1.legend(loc="upper left", fontsize=8.5,
               facecolor="#12121f", labelcolor=TXT, edgecolor="#444455", framealpha=0.9)
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax1.grid(axis="y", color=GRID, lw=0.5)

    # ── Panel 2: Spread ───────────────────────────────────────────────────────
    ax2 = axes[1]
    sv  = spread.values
    idx = spread.index
    ax2.axhline(0,           color="#444455", lw=0.8)
    ax2.axhline( BUY_THRESH, color="#00ee88", lw=1.0, ls="--", alpha=0.8,
                 label=f"Buy  +{BUY_THRESH:.2f}")
    ax2.axhline(SELL_THRESH, color="#ff5555", lw=1.0, ls="--", alpha=0.8,
                 label=f"Sell {SELL_THRESH:.2f}")
    ax2.fill_between(idx, sv, 0, where=(sv >= 0), alpha=0.30, color="#00dd88", lw=0)
    ax2.fill_between(idx, sv, 0, where=(sv <  0), alpha=0.30, color="#ff4444", lw=0)
    ax2.plot(idx, sv, color="#bbbbbb", lw=0.8, alpha=0.85)
    ax2.set_ylabel("Bull-Bear\nSpread", color=TXT, fontsize=9)
    ax2.set_ylim(-0.85, 0.85)
    ax2.legend(loc="upper right", fontsize=8,
               facecolor="#12121f", labelcolor=TXT, edgecolor="#444455")
    ax2.grid(axis="y", color=GRID, lw=0.5)
    for yr in range(2016, 2027):
        ax2.axvline(pd.Timestamp(f"{yr}-01-01"), color="#333355", lw=0.6, ls=":", alpha=0.7)

    # ── Panel 3: Macro context (yield curve + NFCI) ───────────────────────────
    ax3  = axes[2]
    ax3b = ax3.twinx()

    # Yield curve (left axis) — inverted = negative = red
    ax3.axhline(0, color="#555566", lw=0.8, ls=":")
    ax3.fill_between(yc.index, yc.values, 0,
                     where=(yc.values >= 0), alpha=0.3, color="#4a9eda", lw=0,
                     label="Yield curve (+)")
    ax3.fill_between(yc.index, yc.values, 0,
                     where=(yc.values < 0), alpha=0.35, color="#ff5555", lw=0,
                     label="Inverted curve (-)")
    ax3.plot(yc.index, yc.values, color="#88aadd", lw=0.9, alpha=0.8)
    ax3.set_ylabel("10Y-2Y\nYield Curve (%)", color="#88aadd", fontsize=9)
    ax3.tick_params(axis="y", colors="#88aadd", labelsize=8)
    ax3.set_ylim(-2.5, 3.5)
    ax3.legend(loc="upper left", fontsize=7.5,
               facecolor="#12121f", labelcolor=TXT, edgecolor="#444455", framealpha=0.85)

    # NFCI (right axis) — positive = tight financial conditions = bearish
    ax3b.plot(nfci.index, nfci.values, color="#f0c040", lw=1.0, alpha=0.7,
              label="NFCI (tight=+)")
    ax3b.axhline(0, color="#444444", lw=0.5, ls=":")
    ax3b.set_ylabel("NFCI", color="#f0c040", fontsize=9)
    ax3b.tick_params(axis="y", colors="#f0c040", labelsize=8)
    ax3b.set_ylim(-1.5, 2.5)
    ax3b.legend(loc="upper right", fontsize=7.5,
                facecolor="#12121f", labelcolor=TXT, edgecolor="#444455", framealpha=0.85)
    for spine in ax3b.spines.values():
        spine.set_color("#333344")
    for yr in range(2016, 2027):
        ax3.axvline(pd.Timestamp(f"{yr}-01-01"), color="#333355", lw=0.6, ls=":", alpha=0.7)

    # ── Panel 4: Real rates + HY credit spread ───────────────────────────────
    ax4  = axes[3]
    ax4b = ax4.twinx()

    ax4.plot(rr.index, rr.values, color="#aa66cc", lw=1.0, alpha=0.85,
             label="Real rate 10Y (TIPS)")
    ax4.axhline(0, color="#555566", lw=0.6, ls=":")
    ax4.set_ylabel("Real Rate\n10Y (%)", color="#aa66cc", fontsize=9)
    ax4.tick_params(axis="y", colors="#aa66cc", labelsize=8)
    ax4.set_ylim(-3, 4)
    ax4.legend(loc="upper left", fontsize=7.5,
               facecolor="#12121f", labelcolor=TXT, edgecolor="#444455", framealpha=0.85)

    # HY credit spread proxy (right axis) — falling = risk-on, rising = risk-off
    cs_chart = cs[cs != 0]
    ax4b.plot(cs_chart.index, cs_chart.values, color="#ff8844", lw=1.0, alpha=0.75,
              label="HYG/LQD (credit risk)")
    ax4b.set_ylabel("HYG/LQD\n(log ratio)", color="#ff8844", fontsize=9)
    ax4b.tick_params(axis="y", colors="#ff8844", labelsize=8)
    ax4b.legend(loc="upper right", fontsize=7.5,
                facecolor="#12121f", labelcolor=TXT, edgecolor="#444455", framealpha=0.85)
    for spine in ax4b.spines.values():
        spine.set_color("#333344")
    for yr in range(2016, 2027):
        ax4.axvline(pd.Timestamp(f"{yr}-01-01"), color="#333355", lw=0.6, ls=":", alpha=0.7)

    # X-axis
    ax4.xaxis.set_major_locator(mdates.YearLocator())
    ax4.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=0, ha="center", fontsize=9)

    plt.tight_layout(h_pad=0.3)
    out = "chart_full.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\nSaved: {out}")

    # ── Trade log ─────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  TRADE LOG  2015-2026  (walk-forward, annual retraining, +Macro)")
    print(f"{'='*70}")
    print(f"  {'Date':<12} {'Action':<6} {'Spread':>8}  {'QLD $':>8}  {'Return':>8}")
    print(f"  {'-'*58}")
    events = (
        [(d, "BUY",  float(spread.loc[d]), float(chart_df.loc[d, "qld_close"])) for d in buy_dates  if d in chart_df.index]
      + [(d, "SELL", float(spread.loc[d]), float(chart_df.loc[d, "qld_close"])) for d in sell_dates if d in chart_df.index]
    )
    events.sort(key=lambda x: x[0])
    last_buy_price = None
    for d, action, sp, price in events:
        if action == "BUY":
            print(f"  {str(d.date()):<12} {'BUY':<6} {sp:>+8.3f}  ${price:>7.2f}")
            last_buy_price = price
        else:
            ret = (price / last_buy_price - 1) if last_buy_price else float("nan")
            print(f"  {str(d.date()):<12} {'SELL':<6} {sp:>+8.3f}  ${price:>7.2f}  {ret:>+7.1%}")
            last_buy_price = None

    last_w = float(weights.iloc[-1])
    last_d = weights.index[-1]
    if last_w == 1.0 and last_buy_price:
        open_ret = float(chart_df["qld_close"].iloc[-1]) / last_buy_price - 1
        print(f"  {'(open)':<12} {'LONG':<6} {'':>8}  ${chart_df['qld_close'].iloc[-1]:>7.2f}  {open_ret:>+7.1%} (unrealised)")

    print(f"\n  Current: {'LONG' if last_w == 1.0 else 'CASH'}  as of {last_d.date()}")
    print(f"\n  {'Strategy':<28} {'Ann':>7}  {'MDD':>8}  {'Calmar':>7}  {'Total':>8}")
    print(f"  {'-'*60}")
    for name, m, nav in [
        (f"ML (buy={BUY_THRESH}/sell={SELL_THRESH})", m_ml,  nav_ml),
        ("SMA200",                                     m_sma, nav_sma),
        ("QLD B&H",                                    m_bah, nav_bah),
    ]:
        tot = nav.iloc[-1] - 100
        print(f"  {name:<32} {m['ann']:>7.1%}  {m['mdd']:>8.1%}  {m['calmar']:>7.3f}  {tot:>+7.0f}%")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
