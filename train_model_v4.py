"""
Walk-Forward XGBoost Training — Phase 5

Root cause of Phase 4's weak signal (test IC ≈ 0.02-0.05): all v3 models
were trained through 2019-12-31, leaving them 3+ years stale by the time
the 2022 bear market hit. Walk-forward retraining fixes this by deploying
a fresh vintage every ~6 months.

Vintage schedule (train_end → deploy window):
  v0: ≤ 2021-06-30   2022-02-01 → 2022-07-31
  v1: ≤ 2021-12-31   2022-08-01 → 2023-01-31
  v2: ≤ 2022-06-30   2023-02-01 → 2023-07-31
  v3: ≤ 2022-12-31   2023-08-01 → 2024-01-31
  v4: ≤ 2023-06-30   2024-02-01 → 2024-07-31
  v5: ≤ 2023-12-31   2024-08-01 → 2025-01-31
  v6: ≤ 2024-06-30   2025-02-01 → 2025-07-31
  v7: ≤ 2024-12-31   2025-08-01 → ∞

Each vintage trains one clf + reg per sector (11 sectors × 2 = 22 models).
Models saved to models/v4/{vintage_id}/{sector_slug}/clf_10d.ubj + reg_10d.ubj
Manifest saved to models/v4/manifest.json

Usage:
    python train_model_v4.py                  # train all vintages
    python train_model_v4.py --resume         # skip completed vintages
    python train_model_v4.py --vintage v3     # train one vintage only
"""

import warnings
import sys
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
INPUT      = "data/features_phase4.parquet"
MODELS_DIR = Path("models/v4")
MODELS_DIR.mkdir(exist_ok=True)

TARGET_CLF = "fwd_up_10d"
TARGET_REG = "fwd_return_10d"

META_COLS = ["ticker", "date", "Open", "High", "Low", "Close", "Volume", "sector"]

TARGET_COLS = [
    "fwd_return_5d", "fwd_label_5d", "fwd_up_5d", "fwd_down_5d",
    "fwd_return_10d", "fwd_label_10d", "fwd_up_10d", "fwd_down_10d",
    "fwd_return_20d", "fwd_label_20d", "fwd_up_20d", "fwd_down_20d",
    "fwd_sharpe_10d",
]

# Same feature drops as v3 (32 total)
DROP_FEATURES = [
    "close_position", "consecutive_up_days", "consecutive_down_days",
    "breakout_high_20d", "breakout_high_50d", "breakout_low_50d",
    "ichimoku_above_cloud", "ichimoku_tk_cross", "sma_cross_5_20",
    "return_20d", "momentum_20d", "rate_change_direction",
    "hy_credit_spread", "credit_spread_change_20d",
    "return_5d", "entropy_20d", "lower_shadow", "williams_r_14",
    "sector_vol_rank", "return_vs_sector_1d", "obv_slope_5",
    "momentum_5d", "roc_5", "price_vs_sma_20",
    "return_10d", "channel_position_20d", "upper_shadow",
    "price_vs_ema_20", "stoch_d_14", "realized_vol_10d",
    "return_vs_sector_20d", "bb_upper",
]

# Walk-forward vintage schedule
# (id, train_end, val_start, val_end, deploy_start, deploy_end)
# 20-trading-day purge gap between train_end and val_start
VINTAGES = [
    ("v0", "2021-06-30", "2021-08-15", "2022-01-31", "2022-02-01", "2022-07-31"),
    ("v1", "2021-12-31", "2022-02-15", "2022-07-31", "2022-08-01", "2023-01-31"),
    ("v2", "2022-06-30", "2022-08-15", "2023-01-31", "2023-02-01", "2023-07-31"),
    ("v3", "2022-12-31", "2023-02-15", "2023-07-31", "2023-08-01", "2024-01-31"),
    ("v4", "2023-06-30", "2023-08-15", "2024-01-31", "2024-02-01", "2024-07-31"),
    ("v5", "2023-12-31", "2024-02-15", "2024-07-31", "2024-08-01", "2025-01-31"),
    ("v6", "2024-06-30", "2024-08-15", "2025-01-31", "2025-02-01", "2025-07-31"),
    ("v7", "2024-12-31", "2025-02-15", "2025-07-31", "2025-08-01", "2099-12-31"),
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


def split_sector_vintage(df, feat_cols, sector, train_end, val_start, val_end):
    sdf = df[df["sector"] == sector]
    train = sdf[sdf["date"] <= train_end]
    val   = sdf[(sdf["date"] >= val_start) & (sdf["date"] <= val_end)]
    return (
        train[feat_cols], train[TARGET_CLF], train[TARGET_REG],
        val[feat_cols],   val[TARGET_CLF],   val[TARGET_REG],
        val.reset_index(drop=True),
    )


# ── Training ───────────────────────────────────────────────────────────────────
def sector_params(n_train):
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
    return roc_auc_score(y, prob)


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
    return float(ic), float(top_q - bot_q)


# ── Vintage training ───────────────────────────────────────────────────────────
def train_vintage(vintage_id, train_end, val_start, val_end,
                  deploy_start, deploy_end,
                  df, feat_cols, sectors, resume):

    vdir = MODELS_DIR / vintage_id
    vdir.mkdir(exist_ok=True)

    done_path = vdir / "results.json"
    if resume and done_path.exists():
        print(f"  Vintage {vintage_id}: skipping (already complete)")
        with open(done_path) as f:
            return json.load(f)

    print(f"\n{'='*70}")
    print(f"  Vintage {vintage_id}: train ≤ {train_end}  |  "
          f"val {val_start} → {val_end}  |  deploy {deploy_start} → {deploy_end}")
    print(f"{'='*70}")

    train_end_dt = pd.Timestamp(train_end)
    val_start_dt = pd.Timestamp(val_start)
    val_end_dt   = pd.Timestamp(val_end)

    hdr = (f"  {'Sector':<26} {'TrainN':>8} {'ValN':>6} "
           f"{'ValAUC':>8} {'ValIC':>7}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    vintage_results = {}

    for sector in sectors:
        slug = sector.lower().replace(" ", "_")
        sdir = vdir / slug
        sdir.mkdir(exist_ok=True)

        clf_path = sdir / "clf_10d.ubj"
        reg_path = sdir / "reg_10d.ubj"

        (Xtr, ytr_clf, ytr_reg,
         Xv,  yv_clf,  yv_reg,
         val_df) = split_sector_vintage(
            df, feat_cols, sector,
            train_end_dt, val_start_dt, val_end_dt
        )

        if len(Xtr) < 500 or len(Xv) < 50:
            print(f"  {sector:<26} {'SKIP':>8} (insufficient data: "
                  f"train={len(Xtr)}, val={len(Xv)})")
            continue

        if resume and clf_path.exists() and reg_path.exists():
            clf = xgb.XGBClassifier(); clf.load_model(str(clf_path))
            reg = xgb.XGBRegressor();  reg.load_model(str(reg_path))
        else:
            clf = train_classifier(Xtr, ytr_clf, Xv, yv_clf)
            clf.save_model(str(clf_path))
            reg = train_regressor(Xtr, ytr_reg, Xv, yv_reg)
            reg.save_model(str(reg_path))

        val_auc = eval_clf(clf, Xv, yv_clf)
        val_ic, val_ls = eval_reg(reg, Xv, yv_reg, val_df)

        vintage_results[sector] = {
            "n_train": int(len(Xtr)), "n_val": int(len(Xv)),
            "val_auc": round(val_auc, 4),
            "val_ic":  round(val_ic,  4),
            "val_ls":  round(val_ls,  4),
        }

        print(f"  {sector:<26} {len(Xtr):>8,} {len(Xv):>6,} "
              f"{val_auc:>8.4f} {val_ic:>7.4f}")

    # Save vintage results
    output = {
        "vintage_id":    vintage_id,
        "train_end":     train_end,
        "val_start":     val_start,
        "val_end":       val_end,
        "deploy_start":  deploy_start,
        "deploy_end":    deploy_end,
        "n_features":    len(feat_cols),
        "sectors":       vintage_results,
    }
    with open(done_path, "w") as f:
        json.dump(output, f, indent=2)

    avg_auc = np.mean([v["val_auc"] for v in vintage_results.values()])
    avg_ic  = np.mean([v["val_ic"]  for v in vintage_results.values()])
    print(f"\n  Vintage {vintage_id} summary: avg AUC={avg_auc:.4f}  avg IC={avg_ic:.4f}")

    return output


# ── Manifest ───────────────────────────────────────────────────────────────────
def build_manifest(all_results):
    manifest = []
    for res in all_results:
        if res is None:
            continue
        manifest.append({
            "vintage_id":   res["vintage_id"],
            "train_end":    res["train_end"],
            "deploy_start": res["deploy_start"],
            "deploy_end":   res["deploy_end"],
            "n_features":   res.get("n_features", 141),
            "sector_val_ic": {
                s: v["val_ic"]
                for s, v in res.get("sectors", {}).items()
            },
        })

    with open(MODELS_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {MODELS_DIR}/manifest.json  ({len(manifest)} vintages)")
    return manifest


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    resume = "--resume" in sys.argv

    # Optional: train a single vintage
    target_vintage = None
    if "--vintage" in sys.argv:
        idx = sys.argv.index("--vintage")
        target_vintage = sys.argv[idx + 1]

    df, feat_cols = load_data()
    sectors = sorted(df["sector"].unique())

    vintages_to_run = VINTAGES
    if target_vintage:
        vintages_to_run = [v for v in VINTAGES if v[0] == target_vintage]
        if not vintages_to_run:
            print(f"Unknown vintage '{target_vintage}'. "
                  f"Valid: {[v[0] for v in VINTAGES]}")
            sys.exit(1)

    print(f"\nTraining {len(vintages_to_run)} vintage(s) × "
          f"{len(sectors)} sectors × 2 models each\n")

    all_results = []
    for vintage in vintages_to_run:
        vid, train_end, val_start, val_end, deploy_start, deploy_end = vintage
        res = train_vintage(
            vid, train_end, val_start, val_end, deploy_start, deploy_end,
            df, feat_cols, sectors, resume
        )
        all_results.append(res)

    # (Re)build manifest from all completed vintages
    completed = []
    for v in VINTAGES:
        done_path = MODELS_DIR / v[0] / "results.json"
        if done_path.exists():
            with open(done_path) as f:
                completed.append(json.load(f))
    build_manifest(completed)

    print("\nAll done.")
    print(f"Run scorer_v2.py / strategy_v4.py to backtest with walk-forward models.")


if __name__ == "__main__":
    main()
