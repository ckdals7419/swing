"""
Diagnose ML model failures: identify bad trade periods, analyze feature
behavior during those periods, and recommend improvements.
"""
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path

warnings.filterwarnings("ignore")

from crossval_qld import (
    build_full_dataset, WINDOWS, BTC_PERIODS,
    ML_FEATURES_BASE, CROSS_ASSET_ADDITIONS,
    train_ml_for_window, w_sma200, backtest, compute_metrics,
    weights_from_spread, BUY_THRESH, SELL_THRESH,
)

FEAT_LIST = ML_FEATURES_BASE + CROSS_ASSET_ADDITIONS


def get_spread_and_weights(full_df, label, start, end, thresh=0.10):
    """Train model for window, return (window_df, spread, ml_weights, sma_weights)."""
    start_dt = pd.Timestamp(start)
    end_dt   = pd.Timestamp(end) if end else full_df.index.max()
    window   = full_df.loc[start_dt:end_dt].copy()

    ml_result = train_ml_for_window(full_df, start_dt, end_dt,
                                     feature_override=FEAT_LIST)
    if ml_result is None:
        return None

    bull_clf, bear_clf, feat_cols = ml_result

    X = window[[c for c in feat_cols if c in window.columns]].copy()
    for c in feat_cols:
        if c not in X.columns:
            X[c] = 0.0
    X = X[feat_cols].ffill().fillna(0.0)

    dmat      = xgb.DMatrix(X, feature_names=feat_cols)
    bull_prob = pd.Series(bull_clf.predict(dmat), index=window.index)
    bear_prob = pd.Series(bear_clf.predict(dmat), index=window.index)
    spread    = bull_prob - bear_prob

    ml_w  = weights_from_spread(spread, thresh)
    sma_w = w_sma200(window)

    # Feature importances
    bull_imp = bull_clf.get_score(importance_type="gain")
    bear_imp = bear_clf.get_score(importance_type="gain")

    return window, spread, ml_w, sma_w, bull_prob, bear_prob, feat_cols, bull_imp, bear_imp


def daily_pnl(window, weights):
    """Simple daily P&L: weight * QLD daily return."""
    ret = window["qld_close"].pct_change().fillna(0)
    w   = weights.shift(1).fillna(0)  # signal at T, applied T+1
    return w * ret


def find_bad_periods(window, ml_w, sma_w, min_streak=5):
    """
    Find periods where ML significantly underperforms SMA200.
    Returns DataFrame of bad episodes.
    """
    ml_pnl  = daily_pnl(window, ml_w)
    sma_pnl = daily_pnl(window, sma_w)
    diff    = ml_pnl - sma_pnl  # negative = ML worse

    # Rolling 20-day relative P&L
    roll_diff = diff.rolling(20).sum()

    qld_ret = window["qld_close"].pct_change().fillna(0)

    episodes = []

    # Type 1: ML is IN (w=1) while market crashes (>5% drawdown in 20 days)
    ml_in = ml_w == 1.0
    sma_out = sma_w == 0.0
    roll_ret = qld_ret.rolling(20).sum()

    crash_while_in = ml_in & (roll_ret < -0.05)
    missed_exit = crash_while_in & sma_out

    # Type 2: ML is OUT (w=0) while market rallies (>5% gain in 20 days)
    ml_out = ml_w == 0.0
    sma_in = sma_w == 1.0
    rally_while_out = ml_out & (roll_ret > 0.05)
    missed_entry = rally_while_out & sma_in

    return {
        "crash_while_in": missed_exit,       # ML in, SMA out, market crashing
        "rally_while_out": missed_entry,     # ML out, SMA in, market rallying
        "ml_pnl": ml_pnl,
        "sma_pnl": sma_pnl,
        "diff": diff,
        "roll_diff_20d": roll_diff,
    }


def analyze_window(full_df, label, start, end, thresh=0.10):
    """Full diagnostic for one window."""
    result = get_spread_and_weights(full_df, label, start, end, thresh)
    if result is None:
        print(f"  [{label}] Training failed")
        return

    window, spread, ml_w, sma_w, bull_prob, bear_prob, feat_cols, bull_imp, bear_imp = result

    # Basic metrics
    ml_nav  = backtest(window, ml_w)
    sma_nav = backtest(window, sma_w)
    bah_nav = backtest(window, pd.Series(1.0, index=window.index))
    ml_m  = compute_metrics(ml_nav, ml_w)
    sma_m = compute_metrics(sma_nav, sma_w)

    print(f"\n{'='*90}")
    print(f"  DIAGNOSIS: {label} ({start} to {end or 'latest'})")
    print(f"{'='*90}")
    print(f"  ML:    Ann={ml_m['ann']:>7.1%}  MDD={ml_m['mdd']:>7.1%}  Cal={ml_m['calmar']:.3f}  AvgW={ml_m['avg_w']:.1%}")
    print(f"  SMA200:Ann={sma_m['ann']:>7.1%}  MDD={sma_m['mdd']:>7.1%}  Cal={sma_m['calmar']:.3f}  AvgW={sma_m['avg_w']:.1%}")

    # --- Signal analysis ---
    print(f"\n  --- Signal Statistics ---")
    print(f"  Spread: mean={spread.mean():.3f}  std={spread.std():.3f}  "
          f"min={spread.min():.3f}  max={spread.max():.3f}")
    print(f"  Bull prob: mean={bull_prob.mean():.3f}  Bear prob: mean={bear_prob.mean():.3f}")
    print(f"  ML weight: mean={ml_w.mean():.1%}  (in={int((ml_w==1).sum())}d, out={int((ml_w==0).sum())}d)")
    print(f"  SMA weight: mean={sma_w.mean():.1%}  (in={int((sma_w==1).sum())}d, out={int((sma_w==0).sum())}d)")

    # Agreement / disagreement analysis
    agree    = ((ml_w == 1) & (sma_w == 1)) | ((ml_w == 0) & (sma_w == 0))
    disagree = ~agree
    ml_in_sma_out  = (ml_w == 1) & (sma_w == 0)
    ml_out_sma_in  = (ml_w == 0) & (sma_w == 1)

    qld_ret = window["qld_close"].pct_change().fillna(0)
    print(f"\n  --- Agreement Analysis ---")
    print(f"  Agree:  {agree.sum()}d ({agree.mean():.0%})  "
          f"Disagree: {disagree.sum()}d ({disagree.mean():.0%})")
    print(f"  ML=IN,  SMA=OUT: {ml_in_sma_out.sum()}d ({ml_in_sma_out.mean():.0%})")
    print(f"  ML=OUT, SMA=IN:  {ml_out_sma_in.sum()}d ({ml_out_sma_in.mean():.0%})")

    # P&L in each regime
    if ml_in_sma_out.sum() > 0:
        ret_ml_in_sma_out = qld_ret[ml_in_sma_out].sum()
        print(f"    -> When ML=IN, SMA=OUT: QLD cumulative return = {ret_ml_in_sma_out:.1%} "
              f"({'GOOD' if ret_ml_in_sma_out > 0 else 'BAD'} for ML)")
    if ml_out_sma_in.sum() > 0:
        ret_ml_out_sma_in = qld_ret[ml_out_sma_in].sum()
        print(f"    -> When ML=OUT, SMA=IN: QLD cumulative return = {ret_ml_out_sma_in:.1%} "
              f"({'GOOD' if ret_ml_out_sma_in < 0 else 'BAD'} for ML)")

    # --- Bad period analysis ---
    bad = find_bad_periods(window, ml_w, sma_w)

    crash_in_days  = bad["crash_while_in"].sum()
    rally_out_days = bad["rally_while_out"].sum()
    print(f"\n  --- Error Patterns ---")
    print(f"  Crash-while-IN  (ML=1, SMA=0, 20d ret<-5%): {crash_in_days} days")
    print(f"  Rally-while-OUT (ML=0, SMA=1, 20d ret>+5%): {rally_out_days} days")

    # Identify the worst specific episodes
    roll_diff = bad["roll_diff_20d"]

    # Find the 5 worst 20-day underperformance periods
    worst_dates = roll_diff.nsmallest(5)
    if len(worst_dates) > 0:
        print(f"\n  --- Worst 5 Underperformance Periods (20d rolling ML-SMA) ---")
        for date, val in worst_dates.items():
            d20_start = date - pd.Timedelta(days=30)
            d20_end   = date
            mask = (window.index >= d20_start) & (window.index <= d20_end)
            if mask.sum() == 0:
                continue

            sub = window.loc[mask]
            qld_ret_period = sub["qld_close"].iloc[-1] / sub["qld_close"].iloc[0] - 1
            ml_w_avg  = ml_w.loc[mask].mean()
            sma_w_avg = sma_w.loc[mask].mean()
            vix_avg   = sub["vix_close"].mean()
            dd_avg    = sub["qqq_drawdown"].mean() if "qqq_drawdown" in sub.columns else float("nan")

            print(f"    {date.date()}: ML-SMA gap={val:+.1%}  QLD={qld_ret_period:+.1%}  "
                  f"ML_w={ml_w_avg:.0%}  SMA_w={sma_w_avg:.0%}  "
                  f"VIX={vix_avg:.0f}  DD={dd_avg:.1%}")

    # --- Feature behavior during errors ---
    print(f"\n  --- Feature Values During Errors vs Normal ---")

    key_features = [
        "vix_close", "vix_percentile", "qqq_drawdown", "days_in_drawdown",
        "volatility_20d", "return_20d", "rsi_20d", "rsi_60d",
        "tlt_return_20d", "hyg_return_20d", "qqq_vs_spy_20d",
        "adx", "macd_hist",
    ]
    key_features = [f for f in key_features if f in window.columns]

    # Regime: ML wrong and losing vs ML correct
    ml_pnl = daily_pnl(window, ml_w)
    sma_pnl = daily_pnl(window, sma_w)

    # Days where ML underperforms SMA by >0.5%
    ml_bad_days = (ml_pnl - sma_pnl) < -0.005
    ml_good_days = (ml_pnl - sma_pnl) > 0.005

    if ml_bad_days.sum() > 5 and ml_good_days.sum() > 5:
        print(f"  {'Feature':<25} {'ML-bad':>10} {'ML-good':>10} {'All':>10} {'Delta':>10}")
        print(f"  {'-'*65}")
        for feat in key_features:
            if feat in window.columns:
                bad_val  = window.loc[ml_bad_days, feat].mean()
                good_val = window.loc[ml_good_days, feat].mean()
                all_val  = window[feat].mean()
                delta    = bad_val - good_val
                print(f"  {feat:<25} {bad_val:>10.3f} {good_val:>10.3f} {all_val:>10.3f} {delta:>+10.3f}")

    # --- Transition analysis: what triggers false buys/sells ---
    print(f"\n  --- False Signal Analysis ---")

    # Find buy signals that led to losses (bought, then 10d return < -3%)
    buy_signals = (ml_w.diff() == 1)  # 0->1 transitions
    sell_signals = (ml_w.diff() == -1)  # 1->0 transitions
    fwd_10d = window["qld_close"].shift(-10) / window["qld_close"] - 1

    false_buys  = buy_signals & (fwd_10d < -0.03)
    good_buys   = buy_signals & (fwd_10d > 0.03)
    false_sells = sell_signals & (fwd_10d > 0.03)
    good_sells  = sell_signals & (fwd_10d < -0.03)

    n_buys  = buy_signals.sum()
    n_sells = sell_signals.sum()
    print(f"  Total buy signals:  {n_buys}  (false={false_buys.sum()}, good={good_buys.sum()}, neutral={n_buys - false_buys.sum() - good_buys.sum()})")
    print(f"  Total sell signals: {n_sells}  (false={false_sells.sum()}, good={good_sells.sum()}, neutral={n_sells - false_sells.sum() - good_sells.sum()})")

    if false_buys.sum() > 0 and good_buys.sum() > 0:
        print(f"\n  Feature values at FALSE BUYS vs GOOD BUYS:")
        print(f"  {'Feature':<25} {'False Buy':>10} {'Good Buy':>10} {'Delta':>10}")
        print(f"  {'-'*55}")
        for feat in key_features:
            if feat in window.columns:
                fb_val = window.loc[false_buys, feat].mean()
                gb_val = window.loc[good_buys, feat].mean()
                print(f"  {feat:<25} {fb_val:>10.3f} {gb_val:>10.3f} {fb_val - gb_val:>+10.3f}")

    if false_sells.sum() > 0 and good_sells.sum() > 0:
        print(f"\n  Feature values at FALSE SELLS vs GOOD SELLS:")
        print(f"  {'Feature':<25} {'False Sell':>10} {'Good Sell':>10} {'Delta':>10}")
        print(f"  {'-'*55}")
        for feat in key_features:
            if feat in window.columns:
                fs_val = window.loc[false_sells, feat].mean()
                gs_val = window.loc[good_sells, feat].mean()
                print(f"  {feat:<25} {fs_val:>10.3f} {gs_val:>10.3f} {fs_val - gs_val:>+10.3f}")

    # --- Spread distribution during different market regimes ---
    print(f"\n  --- Spread by Market Regime ---")
    dd = window["qqq_drawdown"] if "qqq_drawdown" in window.columns else pd.Series(0, index=window.index)
    vix = window["vix_close"]

    regimes = {
        "Bull (DD>-5%)":        dd > -0.05,
        "Correction (-5%~-15%)": (dd <= -0.05) & (dd > -0.15),
        "Bear (DD<-15%)":       dd <= -0.15,
        "Low VIX (<20)":        vix < 20,
        "Med VIX (20-30)":      (vix >= 20) & (vix < 30),
        "High VIX (>30)":       vix >= 30,
    }

    print(f"  {'Regime':<25} {'Days':>6} {'Spread':>8} {'ML_w':>6} {'SMA_w':>6} {'QLD ret':>8}")
    print(f"  {'-'*65}")
    for name, mask in regimes.items():
        if mask.sum() == 0:
            continue
        s_mean = spread.loc[mask].mean()
        ml_mean = ml_w.loc[mask].mean()
        sma_mean = sma_w.loc[mask].mean()
        qld_r = qld_ret.loc[mask].sum()
        print(f"  {name:<25} {mask.sum():>6} {s_mean:>8.3f} {ml_mean:>6.0%} {sma_mean:>6.0%} {qld_r:>+8.1%}")


def main():
    full_df = build_full_dataset()

    for label, start, end, bear_type in WINDOWS:
        analyze_window(full_df, label, start, end, thresh=0.10)

    # --- Cross-window summary of failure patterns ---
    print(f"\n\n{'='*90}")
    print(f"  CROSS-WINDOW FAILURE PATTERN SUMMARY")
    print(f"{'='*90}")
    print("""
  Key questions for feature engineering:
  1. Can the model distinguish "V-recovery" from "dead cat bounce"?
  2. Can the model detect regime transitions (bull->bear, bear->bull)?
  3. Does the model over-rely on recent momentum (which works in trends but fails at turns)?
  4. Are cross-asset signals arriving too late (lagging the equity move)?

  Run complete. Check output above for per-window diagnostics.
""")


if __name__ == "__main__":
    main()
