"""
XGBoost Swing Trading Model - Training Script

Targets (10-day horizon):
  - Regression : fwd_return_10d
  - Classification: fwd_up_10d  (1 if return > +2%, else 0)

Split:
  Train      : 1990-01-01 ~ 2021-12-31
  Validation : 2022-01-01 ~ 2023-12-31   (20-day purge gap)
  Test       : 2024-01-01 ~ 2026-05-06   (20-day purge gap)

Usage:
    python train_model.py
"""

import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import json
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, classification_report, mean_absolute_error
)
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
INPUT          = "data/features_phase4.parquet"
MODELS_DIR     = Path("models/v2")
MODELS_DIR.mkdir(exist_ok=True)

TRAIN_END      = "2019-12-31"
VAL_START      = "2020-02-01"   # ~20 trading-day purge
VAL_END        = "2021-12-31"
TEST_START     = "2022-02-01"   # ~20 trading-day purge — includes 2022 bear market

TARGET_CLF     = "fwd_up_10d"
TARGET_REG     = "fwd_return_10d"
FORWARD_DAYS   = 10             # used to strip last N rows (no valid target)

META_COLS      = ["ticker", "date", "Open", "High", "Low", "Close", "Volume", "sector"]

# Features removed after screening (zero gain or Tier 4)
DROP_FEATURES  = [
    # DROP — zero importance in both models
    "close_position", "consecutive_up_days", "consecutive_down_days",
    "breakout_high_20d", "breakout_high_50d", "breakout_low_50d",
    "ichimoku_above_cloud", "ichimoku_tk_cross", "sma_cross_5_20",
    "return_20d", "momentum_20d", "rate_change_direction",
    "hy_credit_spread", "credit_spread_change_20d",
    # Tier 4 — low value in both models
    "return_5d", "entropy_20d", "lower_shadow", "williams_r_14",
    "sector_vol_rank", "return_vs_sector_1d", "obv_slope_5",
    "momentum_5d", "roc_5", "price_vs_sma_20",
]
TARGET_COLS    = [c for c in ["fwd_return_5d","fwd_label_5d","fwd_up_5d","fwd_down_5d",
                               "fwd_return_10d","fwd_label_10d","fwd_up_10d","fwd_down_10d",
                               "fwd_return_20d","fwd_label_20d","fwd_up_20d","fwd_down_20d",
                               "fwd_sharpe_10d"]]

# XGBoost params (classifier)
CLF_PARAMS = dict(
    objective        = "binary:logistic",
    eval_metric      = "auc",
    n_estimators     = 1000,
    learning_rate    = 0.05,
    max_depth        = 6,
    min_child_weight = 50,
    subsample        = 0.8,
    colsample_bytree = 0.6,
    gamma            = 1.0,
    reg_alpha        = 0.1,
    reg_lambda       = 1.0,
    tree_method      = "hist",
    random_state     = 42,
    n_jobs           = -1,
    early_stopping_rounds = 50,
)

# XGBoost params (regressor)
REG_PARAMS = dict(
    objective        = "reg:squarederror",
    eval_metric      = "mae",
    n_estimators     = 1000,
    learning_rate    = 0.05,
    max_depth        = 6,
    min_child_weight = 50,
    subsample        = 0.8,
    colsample_bytree = 0.6,
    gamma            = 1.0,
    reg_alpha        = 0.1,
    reg_lambda       = 1.0,
    tree_method      = "hist",
    random_state     = 42,
    n_jobs           = -1,
    early_stopping_rounds = 50,
)


# ── Load & prepare ─────────────────────────────────────────────────────────────
def load_data():
    print(f"Loading {INPUT}...")
    df = pd.read_parquet(INPUT)
    df["date"] = pd.to_datetime(df["date"])

    # Drop rows with no valid forward target (last FORWARD_DAYS per ticker)
    df = df[df[TARGET_CLF].notna() & df[TARGET_REG].notna()].copy()

    # Feature columns = all non-meta, non-target, non-dropped
    feat_cols = [c for c in df.columns
                 if c not in META_COLS and c not in TARGET_COLS
                 and not c.startswith("fwd_")
                 and c not in DROP_FEATURES]

    # Drop rows with >50% null features (warmup period)
    null_ratio = df[feat_cols].isnull().mean(axis=1)
    df = df[null_ratio <= 0.5].copy()

    print(f"  Rows after cleaning: {len(df):,}")
    print(f"  Features: {len(feat_cols)}")
    print(f"  Date range: {df['date'].min().date()} -> {df['date'].max().date()}")
    return df, feat_cols


def split_data(df, feat_cols):
    train = df[df["date"] <= TRAIN_END]
    val   = df[(df["date"] >= VAL_START) & (df["date"] <= VAL_END)]
    test  = df[df["date"] >= TEST_START]

    print(f"\nSplit sizes:")
    print(f"  Train : {len(train):>10,}  ({train['date'].min().date()} ~ {train['date'].max().date()})")
    print(f"  Val   : {len(val):>10,}  ({val['date'].min().date()} ~ {val['date'].max().date()})")
    print(f"  Test  : {len(test):>10,}  ({test['date'].min().date()} ~ {test['date'].max().date()})")

    Xtr, ytr_clf, ytr_reg = train[feat_cols], train[TARGET_CLF], train[TARGET_REG]
    Xv,  yv_clf,  yv_reg  = val[feat_cols],   val[TARGET_CLF],   val[TARGET_REG]
    Xte, yte_clf, yte_reg = test[feat_cols],  test[TARGET_CLF],  test[TARGET_REG]

    return (Xtr, ytr_clf, ytr_reg,
            Xv,  yv_clf,  yv_reg,
            Xte, yte_clf, yte_reg,
            val, test, feat_cols)


# ── Training ──────────────────────────────────────────────────────────────────
def train_classifier(Xtr, ytr, Xv, yv):
    print("\n=== Training Classifier (fwd_up_10d) ===")
    pos_ratio = (ytr == 0).sum() / (ytr == 1).sum()
    params = {**CLF_PARAMS, "scale_pos_weight": pos_ratio}

    model = xgb.XGBClassifier(**params)
    model.fit(
        Xtr, ytr,
        eval_set=[(Xv, yv)],
        verbose=100,
    )
    print(f"  Best iteration: {model.best_iteration}")
    return model


def train_regressor(Xtr, ytr, Xv, yv):
    print("\n=== Training Regressor (fwd_return_10d) ===")
    model = xgb.XGBRegressor(**REG_PARAMS)
    model.fit(
        Xtr, ytr,
        eval_set=[(Xv, yv)],
        verbose=100,
    )
    print(f"  Best iteration: {model.best_iteration}")
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_classifier(model, X, y, split_name):
    prob  = model.predict_proba(X)[:, 1]
    pred  = (prob >= 0.5).astype(int)
    auc   = roc_auc_score(y, prob)
    ap    = average_precision_score(y, prob)
    acc   = accuracy_score(y, pred)
    print(f"\n[{split_name}] Classifier:")
    print(f"  AUC-ROC : {auc:.4f}")
    print(f"  Avg Prec: {ap:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Base rate: {y.mean():.4f}")
    print(classification_report(y, pred, digits=4))
    return {"auc": auc, "avg_precision": ap, "accuracy": acc}


def evaluate_regressor(model, X, y, df_slice, split_name):
    pred = model.predict(X)
    mae  = mean_absolute_error(y, pred)
    ic, _ = spearmanr(pred, y)
    # Top-quintile realized return (strategy signal)
    df_eval = df_slice.copy()
    df_eval["pred"] = pred
    df_eval["actual"] = y.values
    top_q = df_eval.groupby("date").apply(
        lambda g: g.nlargest(max(1, len(g)//5), "pred")["actual"].mean()
    ).mean()
    bot_q = df_eval.groupby("date").apply(
        lambda g: g.nsmallest(max(1, len(g)//5), "pred")["actual"].mean()
    ).mean()
    print(f"\n[{split_name}] Regressor:")
    print(f"  MAE            : {mae:.6f}")
    print(f"  IC (Spearman)  : {ic:.4f}")
    print(f"  Top-20%% mean return : {top_q:.4f} ({top_q*100:.2f}%%)")
    print(f"  Bot-20%% mean return : {bot_q:.4f} ({bot_q*100:.2f}%%)")
    print(f"  Long-short spread   : {(top_q - bot_q)*100:.2f}%%")
    return {"mae": mae, "ic": ic, "top_quintile": top_q, "long_short": top_q - bot_q}


# ── Feature Importance ────────────────────────────────────────────────────────
def save_feature_importance(model, feat_cols, filename):
    scores = model.get_booster().get_score(importance_type="gain")
    imp = pd.DataFrame(list(scores.items()), columns=["feature", "importance_gain"])
    imp = imp.sort_values("importance_gain", ascending=False).reset_index(drop=True)
    imp.to_csv(MODELS_DIR / filename, index=False)
    print(f"\nTop 20 features ({filename}):")
    print(imp.head(20).to_string(index=False))
    return imp


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    import sys
    resume = "--resume" in sys.argv

    df, feat_cols = load_data()

    (Xtr, ytr_clf, ytr_reg,
     Xv,  yv_clf,  yv_reg,
     Xte, yte_clf, yte_reg,
     val_df, test_df, feat_cols) = split_data(df, feat_cols)

    results = {}

    # ── Classifier ────────────────────────────────────────────────────────────
    clf_path = MODELS_DIR / "clf_10d.ubj"
    if resume and clf_path.exists():
        print("\n=== Loading saved Classifier ===")
        clf = xgb.XGBClassifier()
        clf.load_model(str(clf_path))
    else:
        clf = train_classifier(Xtr, ytr_clf, Xv, yv_clf)
        clf.save_model(str(clf_path))

    results["clf_val"]  = evaluate_classifier(clf, Xv,  yv_clf,  "Validation")
    results["clf_test"] = evaluate_classifier(clf, Xte, yte_clf, "Test")
    imp_clf = save_feature_importance(clf, feat_cols, "importance_clf.csv")

    # ── Regressor ─────────────────────────────────────────────────────────────
    reg = train_regressor(Xtr, ytr_reg, Xv, yv_reg)
    reg.save_model(str(MODELS_DIR / "reg_10d.ubj"))

    results["reg_val"]  = evaluate_regressor(reg, Xv,  yv_reg,  val_df.reset_index(drop=True),  "Validation")
    results["reg_test"] = evaluate_regressor(reg, Xte, yte_reg, test_df.reset_index(drop=True), "Test")
    imp_reg = save_feature_importance(reg, feat_cols, "importance_reg.csv")

    # ── Save results ──────────────────────────────────────────────────────────
    with open(MODELS_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Summary ===")
    print(f"Classifier  AUC  - Val: {results['clf_val']['auc']:.4f}  Test: {results['clf_test']['auc']:.4f}")
    print(f"Regressor   IC   - Val: {results['reg_val']['ic']:.4f}  Test: {results['reg_test']['ic']:.4f}")
    print(f"L/S spread (10d) - Val: {results['reg_val']['long_short']*100:.2f}%  Test: {results['reg_test']['long_short']*100:.2f}%")
    print(f"\nModels saved to: {MODELS_DIR}/")


if __name__ == "__main__":
    main()
