"""
XGBoost Swing Trading Model v3 - Sector-by-Sector Training

v3 changes vs v2:
  - 8 additional features dropped (141 total, down from 149)
  - Trains one classifier + regressor per sector (11 sectors)
  - Models saved under models/v3/{sector_slug}/

Usage:
    python train_model_v3.py
    python train_model_v3.py --resume   # skip sectors that already have saved models
"""

import warnings
import sys
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import roc_auc_score, mean_absolute_error
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
INPUT      = "data/features_phase4.parquet"
MODELS_DIR = Path("models/v3")
MODELS_DIR.mkdir(exist_ok=True)

TRAIN_END  = "2019-12-31"
VAL_START  = "2020-02-01"   # ~20-day purge
VAL_END    = "2021-12-31"
TEST_START = "2022-02-01"   # includes 2022 bear market

TARGET_CLF = "fwd_up_10d"
TARGET_REG = "fwd_return_10d"

META_COLS  = ["ticker", "date", "Open", "High", "Low", "Close", "Volume", "sector"]

# All 32 dropped features (24 from v2 + 8 new in v3)
DROP_FEATURES = [
    # v2 — zero importance in both models (14)
    "close_position", "consecutive_up_days", "consecutive_down_days",
    "breakout_high_20d", "breakout_high_50d", "breakout_low_50d",
    "ichimoku_above_cloud", "ichimoku_tk_cross", "sma_cross_5_20",
    "return_20d", "momentum_20d", "rate_change_direction",
    "hy_credit_spread", "credit_spread_change_20d",
    # v2 — Tier 4, low value in both models (10)
    "return_5d", "entropy_20d", "lower_shadow", "williams_r_14",
    "sector_vol_rank", "return_vs_sector_1d", "obv_slope_5",
    "momentum_5d", "roc_5", "price_vs_sma_20",
    # v3 — near-zero in both models after v2 pruning (8)
    "return_10d", "channel_position_20d", "upper_shadow",
    "price_vs_ema_20", "stoch_d_14", "realized_vol_10d",
    "return_vs_sector_20d", "bb_upper",
]

TARGET_COLS = [
    "fwd_return_5d", "fwd_label_5d", "fwd_up_5d", "fwd_down_5d",
    "fwd_return_10d", "fwd_label_10d", "fwd_up_10d", "fwd_down_10d",
    "fwd_return_20d", "fwd_label_20d", "fwd_up_20d", "fwd_down_20d",
    "fwd_sharpe_10d",
]

CLF_PARAMS = dict(
    objective="binary:logistic",
    eval_metric="auc",
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=50,
    subsample=0.8,
    colsample_bytree=0.6,
    gamma=1.0,
    reg_alpha=0.1,
    reg_lambda=1.0,
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50,
)

REG_PARAMS = dict(
    objective="reg:squarederror",
    eval_metric="mae",
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=50,
    subsample=0.8,
    colsample_bytree=0.6,
    gamma=1.0,
    reg_alpha=0.1,
    reg_lambda=1.0,
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50,
)


# ── Data ───────────────────────────────────────────────────────────────────────
def load_data():
    print(f"Loading {INPUT}...")
    df = pd.read_parquet(INPUT)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df[TARGET_CLF].notna() & df[TARGET_REG].notna()].copy()

    feat_cols = [
        c for c in df.columns
        if c not in META_COLS
        and c not in TARGET_COLS
        and not c.startswith("fwd_")
        and c not in DROP_FEATURES
    ]

    null_ratio = df[feat_cols].isnull().mean(axis=1)
    df = df[null_ratio <= 0.5].copy()
    df = df[df["sector"].notna()].copy()

    print(f"  Rows: {len(df):,}  |  Features: {len(feat_cols)}")
    print(f"  Date range: {df['date'].min().date()} -> {df['date'].max().date()}")
    return df, feat_cols


def split_sector(df, feat_cols, sector):
    sdf = df[df["sector"] == sector]
    train = sdf[sdf["date"] <= TRAIN_END]
    val   = sdf[(sdf["date"] >= VAL_START) & (sdf["date"] <= VAL_END)]
    test  = sdf[sdf["date"] >= TEST_START]
    return (
        train[feat_cols], train[TARGET_CLF], train[TARGET_REG],
        val[feat_cols],   val[TARGET_CLF],   val[TARGET_REG],
        test[feat_cols],  test[TARGET_CLF],  test[TARGET_REG],
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


# ── Training ───────────────────────────────────────────────────────────────────
def sector_params(n_train):
    """Scale regularization with training set size.
    Smaller sectors need lighter regularization to learn anything."""
    mcw   = max(10, min(50, n_train // 4000))
    gamma = 0.0 if n_train < 150_000 else 1.0
    return mcw, gamma


def train_classifier(Xtr, ytr, Xv, yv):
    mcw, gamma = sector_params(len(Xtr))
    pos_ratio = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    params = {**CLF_PARAMS, "scale_pos_weight": pos_ratio,
              "min_child_weight": mcw, "gamma": gamma}
    model = xgb.XGBClassifier(**params)
    model.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False)
    return model


def train_regressor(Xtr, ytr, Xv, yv):
    mcw, gamma = sector_params(len(Xtr))
    params = {**REG_PARAMS, "min_child_weight": mcw, "gamma": gamma}
    model = xgb.XGBRegressor(**params)
    model.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False)
    return model


# ── Evaluation ─────────────────────────────────────────────────────────────────
def eval_clf(model, X, y):
    prob = model.predict_proba(X)[:, 1]
    auc  = roc_auc_score(y, prob)
    return auc


def eval_reg(model, X, y, df_slice):
    pred = model.predict(X)
    ic, _ = spearmanr(pred, y)
    df_e = df_slice.copy()
    df_e["pred"]   = pred
    df_e["actual"] = y.values
    top_q = df_e.groupby("date").apply(
        lambda g: g.nlargest(max(1, len(g) // 5), "pred")["actual"].mean()
    ).mean()
    bot_q = df_e.groupby("date").apply(
        lambda g: g.nsmallest(max(1, len(g) // 5), "pred")["actual"].mean()
    ).mean()
    return ic, top_q - bot_q


# ── Feature importance ─────────────────────────────────────────────────────────
def save_importance(model, slug, name):
    scores = model.get_booster().get_score(importance_type="gain")
    imp = (
        pd.DataFrame(list(scores.items()), columns=["feature", "importance_gain"])
        .sort_values("importance_gain", ascending=False)
        .reset_index(drop=True)
    )
    imp.to_csv(MODELS_DIR / slug / f"importance_{name}.csv", index=False)
    return imp


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    resume = "--resume" in sys.argv

    df, feat_cols = load_data()
    sectors = sorted(df["sector"].unique())

    print(f"\nTraining {len(sectors)} sectors x 2 models each\n")
    hdr = (f"{'Sector':<26} {'TrainN':>8} {'ValN':>6} {'TestN':>7} "
           f"{'ValAUC':>8} {'TstAUC':>8} {'ValIC':>7} {'TstIC':>7} {'TstL/S':>8}")
    print(hdr)
    print("-" * len(hdr))

    all_results = {}

    for sector in sectors:
        slug = sector.lower().replace(" ", "_")
        sdir = MODELS_DIR / slug
        sdir.mkdir(exist_ok=True)

        (Xtr, ytr_clf, ytr_reg,
         Xv,  yv_clf,  yv_reg,
         Xte, yte_clf, yte_reg,
         val_df, test_df) = split_sector(df, feat_cols, sector)

        clf_path = sdir / "clf_10d.ubj"
        reg_path = sdir / "reg_10d.ubj"

        if resume and clf_path.exists() and reg_path.exists():
            clf = xgb.XGBClassifier(); clf.load_model(str(clf_path))
            reg = xgb.XGBRegressor();  reg.load_model(str(reg_path))
        else:
            clf = train_classifier(Xtr, ytr_clf, Xv, yv_clf)
            clf.save_model(str(clf_path))
            reg = train_regressor(Xtr, ytr_reg, Xv, yv_reg)
            reg.save_model(str(reg_path))
            save_importance(clf, slug, "clf")
            save_importance(reg, slug, "reg")

        val_auc  = eval_clf(clf, Xv,  yv_clf)
        test_auc = eval_clf(clf, Xte, yte_clf)
        val_ic,  val_ls  = eval_reg(reg, Xv,  yv_reg,  val_df)
        test_ic, test_ls = eval_reg(reg, Xte, yte_reg, test_df)

        all_results[sector] = {
            "n_train": int(len(Xtr)), "n_val": int(len(Xv)), "n_test": int(len(Xte)),
            "val_auc": round(val_auc, 4), "test_auc": round(test_auc, 4),
            "val_ic":  round(val_ic,  4), "test_ic":  round(test_ic,  4),
            "val_ls":  round(float(val_ls),  4),
            "test_ls": round(float(test_ls), 4),
        }

        print(f"  {sector:<24} {len(Xtr):>8,} {len(Xv):>6,} {len(Xte):>7,} "
              f"{val_auc:>8.4f} {test_auc:>8.4f} "
              f"{val_ic:>7.4f} {test_ic:>7.4f} {test_ls*100:>7.2f}%")

    with open(MODELS_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n=== Summary ===")
    best_auc = max(all_results, key=lambda s: all_results[s]["test_auc"])
    best_ic  = max(all_results, key=lambda s: all_results[s]["test_ic"])
    best_ls  = max(all_results, key=lambda s: all_results[s]["test_ls"])
    print(f"Best Test AUC : {best_auc} ({all_results[best_auc]['test_auc']:.4f})")
    print(f"Best Test IC  : {best_ic}  ({all_results[best_ic]['test_ic']:.4f})")
    print(f"Best L/S spread: {best_ls} ({all_results[best_ls]['test_ls']*100:.2f}%)")
    print(f"\nModels saved to: {MODELS_DIR}/  ({len(feat_cols)} features per sector)")


if __name__ == "__main__":
    main()
