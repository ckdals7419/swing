"""
Real-time prediction for today using the full trained model.
Trains on all available data, predicts on the latest row.
"""
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")

from crossval_qld import (
    build_full_dataset,
    ML_FEATURES_BASE, CROSS_ASSET_ADDITIONS, REGIME_ADDITIONS,
    BTC_XGB_PARAMS, NUM_BOOST_ROUND, EARLY_STOP_ROUNDS,
)
from sklearn.metrics import roc_auc_score

FEAT_LIST = ML_FEATURES_BASE + CROSS_ASSET_ADDITIONS + REGIME_ADDITIONS
BUY_THRESH  =  0.35
SELL_THRESH = -0.45


def train_on_all(full_df):
    feat_cols   = [c for c in FEAT_LIST if c in full_df.columns]
    target_cols = ["target_bull", "target_bear"]

    train_df = full_df.dropna(subset=feat_cols + target_cols)
    n_val    = max(60, int(len(train_df) * 0.10))
    val_df   = train_df.iloc[-n_val:]
    tr_df    = train_df.iloc[:-n_val]

    X_tr  = tr_df[feat_cols].ffill().fillna(0.0)
    X_val = val_df[feat_cols].ffill().fillna(0.0)

    boosters = {}
    for target in ("target_bull", "target_bear"):
        y_tr  = tr_df[target]
        y_val = val_df[target]

        n_neg = (y_tr == 0).sum()
        n_pos = (y_tr == 1).sum()
        spw   = float(n_neg) / max(1.0, float(n_pos))

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
        val_pred = booster.predict(dval)
        auc = roc_auc_score(y_val, val_pred) if y_val.sum() > 0 else float("nan")
        n_trees = booster.num_boosted_rounds()
        print(f"  {target:<15} AUC={auc:.3f}  trees={n_trees}")
        boosters[target] = booster

    return boosters["target_bull"], boosters["target_bear"], feat_cols


def main():
    full_df = build_full_dataset()

    latest_date = full_df.index.max()
    print(f"\nLatest data: {latest_date.date()}  (predicting 5-day forward from this close)")

    print("\nTraining on all available data...")
    bull_clf, bear_clf, feat_cols = train_on_all(full_df)

    # Predict on last available row
    last = full_df.iloc[[-1]]
    X = last[feat_cols].ffill().fillna(0.0)
    dmat = xgb.DMatrix(X, feature_names=feat_cols)

    bull_prob = float(bull_clf.predict(dmat)[0])
    bear_prob = float(bear_clf.predict(dmat)[0])
    spread    = bull_prob - bear_prob

    # Current QLD price
    qld_price = float(last["qld_close"].iloc[0])
    qqq_price = float(last["qqq_close"].iloc[0])
    vix       = float(last["vix_close"].iloc[0])
    dd        = float(last["qqq_drawdown"].iloc[0])

    # Decision
    if spread >= BUY_THRESH:
        decision = "BUY / HOLD LONG"
    elif spread <= SELL_THRESH:
        decision = "SELL / MOVE TO CASH"
    else:
        decision = "HOLD (no change)"

    print(f"\n{'='*60}")
    print(f"  PREDICTION AS OF {latest_date.date()}")
    print(f"{'='*60}")
    print(f"  QLD close:      ${qld_price:.2f}")
    print(f"  QQQ close:      ${qqq_price:.2f}")
    print(f"  VIX:            {vix:.1f}")
    print(f"  QQQ drawdown:   {dd:.1%}")
    print()
    print(f"  Bull prob (>+2% in 5d):  {bull_prob:.3f}")
    print(f"  Bear prob (<-2% in 5d):  {bear_prob:.3f}")
    print(f"  Spread (bull-bear):      {spread:+.3f}")
    print()
    print(f"  Buy threshold:  >= +{BUY_THRESH:.2f}")
    print(f"  Sell threshold: <= {SELL_THRESH:.2f}")
    print()
    print(f"  >>> SIGNAL: {decision} <<<")
    print(f"{'='*60}")

    # Context: recent spread history
    print(f"\n  Recent spread history (last 10 trading days):")
    recent = full_df.iloc[-10:]
    X_rec = recent[feat_cols].ffill().fillna(0.0)
    dmat_rec = xgb.DMatrix(X_rec, feature_names=feat_cols)
    bp = bull_clf.predict(dmat_rec)
    brp = bear_clf.predict(dmat_rec)
    sp = bp - brp
    for date, b, br, s in zip(recent.index, bp, brp, sp):
        sig = "BUY" if s >= BUY_THRESH else ("SELL" if s <= SELL_THRESH else "hold")
        print(f"    {date.date()}  bull={b:.3f}  bear={br:.3f}  spread={s:+.3f}  [{sig}]")


if __name__ == "__main__":
    main()
