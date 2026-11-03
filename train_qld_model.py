"""
Phase D — Train QLD weight model.

Trains two models on daily QQQ/VIX/macro features:
  1. Rule-based regime model (transparent, no ML)
  2. XGBoost classifier + regressor (ML)

Both are evaluated on validation set (2019-03-01 to 2021-12-31).
The better model on val Sharpe is saved as the "primary" model.

Train  : 2007-01-01 → 2018-12-31  (~3,000 samples, includes GFC + bull)
Val    : 2019-03-01 → 2021-12-31  (~700 samples, includes COVID)
Test   : 2022-02-01 → present     (evaluated in strategy_qld.py)

20-trading-day purge gaps between sets.

Output:
  models/qld/clf_10d.ubj      XGBoost classifier
  models/qld/reg_10d.ubj      XGBoost regressor
  models/qld/feature_cols.json
  models/qld/results.json     val/test metrics for both models
"""

import warnings
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

FEATURES_PATH = "data/qld_features.parquet"
MODELS_DIR    = Path("models/qld")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_END  = "2018-12-31"
VAL_START  = "2019-03-01"    # ~40-day purge
VAL_END    = "2021-12-31"
TEST_START = "2022-02-01"    # ~30-day purge

TARGET_CLF = "fwd_up_10d"
TARGET_REG = "fwd_return_10d"

# Columns that are not features
NON_FEAT = {
    "date", "fwd_return_5d", "fwd_return_10d", "fwd_return_20d",
    "fwd_up_5d", "fwd_up_10d", "fwd_up_20d",
    "qld_close", "qld_open", "qqq_close", "qqq_open",
    "bil_return_1d",
}

CLF_PARAMS = dict(
    objective="binary:logistic",
    eval_metric="auc",
    n_estimators=500,
    learning_rate=0.03,
    max_depth=4,
    min_child_weight=20,
    subsample=0.8,
    colsample_bytree=0.7,
    gamma=0.5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=30,
)

REG_PARAMS = dict(
    objective="reg:squarederror",
    eval_metric="mae",
    n_estimators=500,
    learning_rate=0.03,
    max_depth=4,
    min_child_weight=20,
    subsample=0.8,
    colsample_bytree=0.7,
    gamma=0.5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=30,
)


# ── Rule-based regime model ────────────────────────────────────────────────────

def rule_weight(row) -> float:
    """
    Deterministic regime → QLD weight.

    Rules (evaluated top-to-bottom, first match wins):
      Crisis    : vix > 30 OR qqq_drawdown < -15%   → 0%
      Bear      : below SMA200 AND vix > 22           → 25%
      Cautious  : below SMA200 OR vix > 22            → 50%
      Pullback  : below SMA50 (but above SMA200)      → 75%
      Bull      : everything else                     → 100%
    """
    vix   = row.get("vix_close",    20.0)
    dd    = row.get("qqq_drawdown",  0.0)
    vs200 = row.get("qqq_vs_sma200", 0.0)
    vs50  = row.get("qqq_vs_sma50",  0.0)

    if pd.isna(vix):   vix   = 20.0
    if pd.isna(dd):    dd    = 0.0
    if pd.isna(vs200): vs200 = 0.0
    if pd.isna(vs50):  vs50  = 0.0

    if vix > 30 or dd < -0.15:
        return 0.0
    if vs200 < 0 and vix > 22:
        return 0.25
    if vs200 < 0 or vix > 22:
        return 0.50
    if vs50 < 0:
        return 0.75
    return 1.0


# ── Backtest helper ────────────────────────────────────────────────────────────

def backtest_weights(df: pd.DataFrame, weights: pd.Series,
                     cost_bps: float = 5.0) -> pd.Series:
    """
    Simulate NAV given a daily weight series for QLD.

    - Signal at T close → trade at T+1 open
    - Cash earns bil_return_1d
    - Transaction cost: cost_bps per side on the weight change × NAV

    Returns: NAV series indexed by date (starting at 1.0)
    """
    cost_rate = cost_bps / 10_000.0

    # Align
    df = df.set_index("date") if "date" in df.columns else df.copy()
    weights = weights.reindex(df.index).ffill().fillna(1.0)

    qld_open   = df["qld_open"]
    qld_close  = df["qld_close"]
    cash_daily = df["bil_return_1d"].fillna(0.0)

    # Compute QLD daily return (open-to-close same day for fill-next-day)
    # We fill at T+1 open → hold to T+1 close
    # qld_ret[t] = (qld_close[t] / qld_open[t]) - 1  is T+1 within-day
    # Better: weight decided at T close, filled at T+1 open, closed at T+1 close
    # qld_hold_ret[t+1] = qld_close[t+1] / qld_open[t+1] - 1
    qld_hold_ret = qld_close / qld_open - 1   # row t is the return if filled at open[t]

    # Shift weights by 1: signal at T → weight applied from T+1
    w_applied = weights.shift(1).fillna(0.0)   # start in cash day 1
    w_prev    = w_applied.shift(1).fillna(0.0)
    w_change  = (w_applied - w_prev).abs()

    nav = pd.Series(index=df.index, dtype=float)
    nav_val = 1.0
    for t in df.index:
        w  = w_applied.loc[t]
        dw = w_change.loc[t]
        # Transaction cost on the traded notional
        tc = dw * nav_val * cost_rate
        # Return for this day
        port_ret = w * qld_hold_ret.loc[t] + (1 - w) * cash_daily.loc[t]
        nav_val  = nav_val * (1 + port_ret) - tc
        nav.loc[t] = nav_val

    return nav


def metrics(nav: pd.Series) -> dict:
    ret   = nav.pct_change().dropna()
    total = nav.iloc[-1] / nav.iloc[0] - 1
    nyrs  = len(nav) / 252.0
    ann   = (1 + total) ** (1 / nyrs) - 1 if nyrs > 0 else 0
    vol   = ret.std() * np.sqrt(252)
    rf    = 0.03   # rough avg over period
    sharpe = (ann - rf) / vol if vol > 0 else 0.0
    peak  = nav.expanding().max()
    dd    = (nav - peak) / peak
    mdd   = dd.min()
    calmar = ann / abs(mdd) if mdd != 0 else 0.0
    return {
        "total_return": round(total, 4),
        "ann_return":   round(ann,   4),
        "volatility":   round(vol,   4),
        "sharpe":       round(sharpe, 3),
        "max_drawdown": round(mdd,   4),
        "calmar":       round(calmar, 3),
    }


# ── ML score → weight ──────────────────────────────────────────────────────────

def score_to_weight(clf_prob: float, reg_pred_z: float,
                    thresholds=(0.35, 0.45, 0.55)) -> float:
    """
    Composite score → 4-level weight: 0, 0.33, 0.67, 1.0
    """
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

    score = 0.6 * clf_prob + 0.4 * sigmoid(reg_pred_z)
    t0, t1, t2 = thresholds
    if score < t0:
        return 0.0
    elif score < t1:
        return 0.33
    elif score < t2:
        return 0.67
    else:
        return 1.0


def ml_weights(clf, reg, X: pd.DataFrame, thresholds=(0.35, 0.45, 0.55)) -> pd.Series:
    clf_prob = clf.predict_proba(X.fillna(0))[:, 1]
    reg_pred = reg.predict(X.fillna(0))
    reg_std  = reg_pred.std()
    reg_z    = (reg_pred - reg_pred.mean()) / reg_std if reg_std > 1e-6 else np.zeros(len(reg_pred))

    weights = pd.Series(
        [score_to_weight(cp, rz, thresholds) for cp, rz in zip(clf_prob, reg_z)],
        index=X.index
    )
    return weights


# ── Threshold tuning ───────────────────────────────────────────────────────────

def tune_thresholds(clf, reg, X_val: pd.DataFrame, val_df: pd.DataFrame) -> tuple:
    """Grid search over threshold pairs to maximize val Sharpe."""
    clf_prob = clf.predict_proba(X_val.fillna(0))[:, 1]
    reg_pred = reg.predict(X_val.fillna(0))
    reg_std  = reg_pred.std()
    reg_z    = (reg_pred - reg_pred.mean()) / reg_std if reg_std > 1e-6 else np.zeros(len(reg_pred))
    score    = 0.6 * clf_prob + 0.4 / (1 + np.exp(-reg_z.clip(-10, 10)))

    score_s  = pd.Series(score, index=X_val.index)

    best_sharpe = -np.inf
    best_thr    = (0.35, 0.45, 0.55)

    # Grid over (t0, t2) pairs; keep t1 = midpoint
    for t0 in np.arange(0.30, 0.50, 0.025):
        for t2 in np.arange(0.50, 0.70, 0.025):
            if t2 <= t0 + 0.05:
                continue
            t1 = (t0 + t2) / 2.0
            weights = score_s.apply(
                lambda s, a=t0, b=t1, c=t2:
                    0.0 if s < a else 0.33 if s < b else 0.67 if s < c else 1.0
            )
            nav = backtest_weights(val_df.copy(), weights)
            m   = metrics(nav)
            if m["sharpe"] > best_sharpe:
                best_sharpe = m["sharpe"]
                best_thr    = (round(t0, 3), round(t1, 3), round(t2, 3))

    print(f"  Best thresholds: {best_thr}  val Sharpe={best_sharpe:.3f}")
    return best_thr


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading {FEATURES_PATH}...")
    df = pd.read_parquet(FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"])

    feat_cols = [c for c in df.columns if c not in NON_FEAT]
    print(f"  {len(df):,} rows, {len(feat_cols)} features")
    print(f"  Date range: {df['date'].min().date()} -> {df['date'].max().date()}")

    # Drop rows with all-null features (warm-up period)
    null_ratio = df[feat_cols].isnull().mean(axis=1)
    df = df[null_ratio < 0.5].copy()
    df = df[df[TARGET_CLF].notna() & df[TARGET_REG].notna()].copy()
    df = df.set_index("date")

    # ── Split ──────────────────────────────────────────────────────────────────
    train = df[df.index <= TRAIN_END]
    val   = df[(df.index >= VAL_START) & (df.index <= VAL_END)]
    test  = df[df.index >= TEST_START]

    print(f"\nSplit: train={len(train)}, val={len(val)}, test={len(test)}")

    Xtr, ytr_clf, ytr_reg = train[feat_cols], train[TARGET_CLF], train[TARGET_REG]
    Xv,  yv_clf,  yv_reg  = val[feat_cols],   val[TARGET_CLF],   val[TARGET_REG]

    # ── Train XGBoost ──────────────────────────────────────────────────────────
    print("\nTraining XGBoost classifier...")
    pos_ratio = (ytr_clf == 0).sum() / max((ytr_clf == 1).sum(), 1)
    clf = xgb.XGBClassifier(**{**CLF_PARAMS, "scale_pos_weight": pos_ratio})
    clf.fit(Xtr.fillna(0), ytr_clf, eval_set=[(Xv.fillna(0), yv_clf)], verbose=False)
    clf.save_model(str(MODELS_DIR / "clf_10d.ubj"))

    print("Training XGBoost regressor...")
    reg = xgb.XGBRegressor(**REG_PARAMS)
    reg.fit(Xtr.fillna(0), ytr_reg, eval_set=[(Xv.fillna(0), yv_reg)], verbose=False)
    reg.save_model(str(MODELS_DIR / "reg_10d.ubj"))

    # ── Tune thresholds on val ─────────────────────────────────────────────────
    print("\nTuning weight thresholds on validation set...")
    best_thr = tune_thresholds(clf, reg, Xv, val.reset_index())

    with open(MODELS_DIR / "thresholds.json", "w") as f:
        json.dump({"thresholds": list(best_thr)}, f)

    # ── Evaluate: rule-based vs ML ─────────────────────────────────────────────
    print("\n" + "="*55)
    print("  Validation: Rule-based vs ML vs QLD B&H")
    print("="*55)

    # Rule-based weights on val
    val_reset = val.reset_index()
    rule_w = val_reset.apply(rule_weight, axis=1)
    rule_w.index = val.index

    # ML weights on val
    ml_w = ml_weights(clf, reg, Xv, best_thr)

    # QLD B&H weight (always 1.0)
    bah_w = pd.Series(1.0, index=val.index)

    for label, weights in [("Rule-based", rule_w), ("ML", ml_w), ("QLD B&H", bah_w)]:
        nav = backtest_weights(val_reset.copy(), weights)
        m   = metrics(nav)
        time_in = (weights > 0).mean()
        print(f"\n  {label}:")
        print(f"    Total return : {m['total_return']:>8.2%}")
        print(f"    Ann. return  : {m['ann_return']:>8.2%}")
        print(f"    Max drawdown : {m['max_drawdown']:>8.2%}")
        print(f"    Sharpe       : {m['sharpe']:>8.3f}")
        print(f"    Calmar       : {m['calmar']:>8.3f}")
        print(f"    Time in mkt  : {time_in:>8.1%}")

    # ── Feature importance ─────────────────────────────────────────────────────
    scores = clf.get_booster().get_score(importance_type="gain")
    imp = (pd.DataFrame(list(scores.items()), columns=["feature", "gain"])
           .sort_values("gain", ascending=False)
           .reset_index(drop=True))
    imp.to_csv(MODELS_DIR / "importance_clf.csv", index=False)
    print(f"\nTop 10 clf features:")
    for _, row in imp.head(10).iterrows():
        print(f"  {row['feature']:<30}  gain={row['gain']:.1f}")

    # ── Save metadata ──────────────────────────────────────────────────────────
    with open(MODELS_DIR / "feature_cols.json", "w") as f:
        json.dump(feat_cols, f)

    results = {
        "train_end":   TRAIN_END,
        "val_start":   VAL_START,
        "val_end":     VAL_END,
        "test_start":  TEST_START,
        "n_train":     len(train),
        "n_val":       len(val),
        "n_features":  len(feat_cols),
        "thresholds":  list(best_thr),
    }
    with open(MODELS_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nModels saved to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
