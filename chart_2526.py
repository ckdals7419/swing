"""
Visualize ML buy/sell signals (buy=0.35, sell=0.45) on QLD for 2025-2026.
Trains on all data before 2025 (with purge), predicts on 2025-present.
"""
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

from crossval_qld import (
    build_full_dataset,
    ML_FEATURES_BASE, CROSS_ASSET_ADDITIONS, REGIME_ADDITIONS,
    BTC_XGB_PARAMS, NUM_BOOST_ROUND, EARLY_STOP_ROUNDS,
    w_sma200, backtest, compute_metrics,
)

FEAT_LIST   = ML_FEATURES_BASE + CROSS_ASSET_ADDITIONS + REGIME_ADDITIONS
BUY_THRESH  =  0.35
SELL_THRESH = -0.45
CHART_START = "2025-01-01"
PURGE_DAYS  = 30

# Walk-forward retraining cutoffs: model retrained at each boundary,
# predicts forward until the next boundary.
# Format: (predict_from, train_through)
RETRAIN_SCHEDULE = [
    ("2025-01-01", "2024-11-30"),   # H1 2025 model
    ("2025-07-01", "2025-05-31"),   # H2 2025 model
    ("2026-01-01", "2025-11-30"),   # H1 2026 model (includes 2025 crashes/recovery)
]


def train_for_period(full_df, train_end, feat_list):
    feat_cols = [c for c in feat_list if c in full_df.columns]
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
    dmat      = xgb.DMatrix(X, feature_names=feat_cols)
    bull_prob = pd.Series(bull_clf.predict(dmat), index=df.index)
    bear_prob = pd.Series(bear_clf.predict(dmat), index=df.index)
    return bull_prob, bear_prob, bull_prob - bear_prob


def weights_asym(spread, buy_t, sell_t):
    s = spread.values
    w = 1.0 if s[0] >= buy_t else 0.0
    weights = np.empty(len(s))
    for i, v in enumerate(s):
        if v >= buy_t:
            w = 1.0
        elif v <= sell_t:
            w = 0.0
        weights[i] = w
    return pd.Series(weights, index=spread.index)


def main():
    full_df = build_full_dataset()
    chart_df = full_df.loc[CHART_START:].copy()

    # Walk-forward: train at each cutoff, predict the subsequent segment
    all_bull = pd.Series(dtype=float)
    all_bear = pd.Series(dtype=float)

    for i, (pred_from, train_through) in enumerate(RETRAIN_SCHEDULE):
        pred_end = RETRAIN_SCHEDULE[i+1][0] if i+1 < len(RETRAIN_SCHEDULE) else None
        seg = chart_df.loc[pred_from:pred_end].copy() if pred_end else chart_df.loc[pred_from:].copy()
        if len(seg) == 0:
            continue

        print(f"Training through {train_through}, predicting {pred_from} → {pred_end or 'latest'}...")
        bull_clf, bear_clf, feat_cols = train_for_period(full_df, train_through, FEAT_LIST)
        bp, brp, _ = compute_spread(seg, bull_clf, bear_clf, feat_cols)
        all_bull = pd.concat([all_bull, bp])
        all_bear = pd.concat([all_bear, brp])

    all_bull = all_bull[~all_bull.index.duplicated(keep="last")]
    all_bear = all_bear[~all_bear.index.duplicated(keep="last")]
    spread   = all_bull - all_bear
    bull_prob, bear_prob = all_bull, all_bear

    # Generate weights & find transitions
    weights = weights_asym(spread, BUY_THRESH, SELL_THRESH)

    # Detect buy/sell events (weight transitions)
    w_diff = weights.diff()
    buy_dates  = weights.index[w_diff ==  1.0]
    sell_dates = weights.index[w_diff == -1.0]

    # NAV series for ML and SMA200
    nav_ml  = backtest(chart_df, weights)
    nav_sma = backtest(chart_df, w_sma200(chart_df))
    nav_bah = backtest(chart_df, pd.Series(1.0, index=chart_df.index))

    # Normalise all to 100 at start
    nav_ml  = nav_ml  / nav_ml.iloc[0]  * 100
    nav_sma = nav_sma / nav_sma.iloc[0] * 100
    nav_bah = nav_bah / nav_bah.iloc[0] * 100
    qld     = chart_df["qld_close"] / chart_df["qld_close"].iloc[0] * 100

    # ── Plot ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(16, 12),
                             gridspec_kw={"height_ratios": [3, 1.2, 1]},
                             sharex=True)
    fig.patch.set_facecolor("#0e1117")
    for ax in axes:
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="#cccccc", labelsize=9)
        ax.spines["bottom"].set_color("#333333")
        ax.spines["top"].set_color("#333333")
        ax.spines["left"].set_color("#333333")
        ax.spines["right"].set_color("#333333")

    # ── Panel 1: NAV comparison ──────────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(qld.index,    qld.values,     color="#555577", lw=1.0, alpha=0.5, label="QLD B&H")
    ax1.plot(nav_sma.index, nav_sma.values, color="#4a9eda", lw=1.3, alpha=0.8, label="SMA200")
    ax1.plot(nav_ml.index,  nav_ml.values,  color="#f0c040", lw=1.8, label="ML (buy=0.35, sell=0.45)")

    # Shade position: green=in, transparent=out
    in_mask  = weights.shift(1).fillna(0) == 1.0
    out_mask = ~in_mask
    for i in range(1, len(chart_df)):
        if in_mask.iloc[i]:
            ax1.axvspan(chart_df.index[i-1], chart_df.index[i],
                        alpha=0.07, color="#00dd88", lw=0)

    # Buy markers on NAV
    for d in buy_dates:
        if d in nav_ml.index:
            ax1.axvline(d, color="#00ee88", lw=0.8, alpha=0.7, ls="--")
            ax1.annotate("B", xy=(d, nav_ml.loc[d]),
                         fontsize=7, color="#00ee88", fontweight="bold",
                         ha="center", va="bottom",
                         xytext=(0, 4), textcoords="offset points")

    # Sell markers on NAV
    for d in sell_dates:
        if d in nav_ml.index:
            ax1.axvline(d, color="#ff5555", lw=0.8, alpha=0.7, ls="--")
            ax1.annotate("S", xy=(d, nav_ml.loc[d]),
                         fontsize=7, color="#ff5555", fontweight="bold",
                         ha="center", va="top",
                         xytext=(0, -4), textcoords="offset points")

    ax1.set_ylabel("NAV (rebased to 100)", color="#cccccc", fontsize=10)
    ax1.set_title("QLD Strategy Signals — 2025–2026  (buy≥+0.35, sell≤−0.45)",
                  color="#eeeeee", fontsize=13, pad=10)
    ax1.legend(loc="upper left", fontsize=9,
               facecolor="#1a1a2e", labelcolor="#cccccc", edgecolor="#444444")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax1.grid(axis="y", color="#222233", lw=0.5)

    # ── Panel 2: Spread + thresholds ─────────────────────────────────────────
    ax2 = axes[1]
    ax2.axhline(0,           color="#444455", lw=0.8)
    ax2.axhline(BUY_THRESH,  color="#00ee88", lw=1.0, ls="--", alpha=0.8, label=f"Buy  +{BUY_THRESH:.2f}")
    ax2.axhline(SELL_THRESH, color="#ff5555", lw=1.0, ls="--", alpha=0.8, label=f"Sell {SELL_THRESH:.2f}")

    # Colour spread above/below zero
    spread_vals = spread.values
    dates_arr   = spread.index

    ax2.fill_between(dates_arr, spread_vals, 0,
                     where=(spread_vals >= 0), alpha=0.35, color="#00dd88", lw=0)
    ax2.fill_between(dates_arr, spread_vals, 0,
                     where=(spread_vals <  0), alpha=0.35, color="#ff4444", lw=0)
    ax2.plot(dates_arr, spread_vals, color="#dddddd", lw=0.9, alpha=0.7)

    ax2.set_ylabel("Spread (bull−bear)", color="#cccccc", fontsize=10)
    ax2.set_ylim(-0.8, 0.8)
    ax2.legend(loc="upper right", fontsize=8,
               facecolor="#1a1a2e", labelcolor="#cccccc", edgecolor="#444444")
    ax2.grid(axis="y", color="#222233", lw=0.5)

    # ── Panel 3: Bull/Bear probabilities ────────────────────────────────────
    ax3 = axes[2]
    ax3.plot(bull_prob.index, bull_prob.values, color="#00dd88", lw=1.1, label="Bull prob")
    ax3.plot(bear_prob.index, bear_prob.values, color="#ff5555", lw=1.1, label="Bear prob")
    ax3.axhline(0.5, color="#444455", lw=0.8, ls=":")
    ax3.set_ylabel("Probability", color="#cccccc", fontsize=10)
    ax3.set_ylim(0, 1)
    ax3.legend(loc="upper right", fontsize=8,
               facecolor="#1a1a2e", labelcolor="#cccccc", edgecolor="#444444")
    ax3.grid(axis="y", color="#222233", lw=0.5)

    # X-axis formatting
    ax3.xaxis.set_major_locator(mdates.MonthLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # Performance annotation
    ml_ret  = nav_ml.iloc[-1]  - 100
    sma_ret = nav_sma.iloc[-1] - 100
    bah_ret = nav_bah.iloc[-1] - 100
    ann_str = (f"Period return:  ML {ml_ret:+.1f}%   SMA200 {sma_ret:+.1f}%   B&H {bah_ret:+.1f}%\n"
               f"Trades: {len(buy_dates)} buys, {len(sell_dates)} sells")
    ax1.text(0.99, 0.04, ann_str, transform=ax1.transAxes,
             fontsize=8, color="#aaaaaa", ha="right", va="bottom",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e", edgecolor="#333333"))

    plt.tight_layout(h_pad=0.4)
    out_path = "chart_2526.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0e1117")
    print(f"Saved: {out_path}")

    # Print trade log
    print(f"\n{'='*60}")
    print(f"  TRADE LOG  (2025-present)")
    print(f"{'='*60}")
    print(f"  {'Date':<12} {'Action':<6} {'Spread':>8} {'QLD $':>8}")
    print(f"  {'-'*42}")
    events = (
        [(d, "BUY",  float(spread.loc[d]), float(chart_df.loc[d, 'qld_close'])) for d in buy_dates]
      + [(d, "SELL", float(spread.loc[d]), float(chart_df.loc[d, 'qld_close'])) for d in sell_dates]
    )
    events.sort(key=lambda x: x[0])
    for d, action, sp, price in events:
        print(f"  {str(d.date()):<12} {action:<6} {sp:>+8.3f} ${price:>7.2f}")

    # Current status
    last_w   = float(weights.iloc[-1])
    last_sp  = float(spread.iloc[-1])
    last_date = spread.index[-1]
    status = "LONG (invested)" if last_w == 1.0 else "CASH (out)"
    print(f"\n  Current position ({last_date.date()}): {status}")
    print(f"  Current spread: {last_sp:+.3f}")
    print(f"{'='*60}")
    print(f"\nPeriod return:  ML {ml_ret:+.1f}%   SMA200 {sma_ret:+.1f}%   B&H {bah_ret:+.1f}%")


if __name__ == "__main__":
    main()
