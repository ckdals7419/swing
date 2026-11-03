"""
Signal Generator — Phase 2

Loads XGBoost models (v2 global + v3 sector) and scores every stock
for a given date using a composite clf + reg signal.

Composite score:
    score = w_clf * clf_prob + w_reg * sigmoid(reg_z)

    clf_prob : sector classifier P(up 2%+ in 10d)            [0, 1]
    reg_z    : sector/global regressor prediction, z-scored    [0, 1]
               within sector then squashed via sigmoid

Routing:
    Classifier  : always use sector model (all have val AUC 0.57-0.63)
    Regressor   : use sector model if sector val_ic >= 0.07
                  use global v2 otherwise
    Financial   : w_reg = 0.0  (sector reg inverted, -0.087 val IC)
    Utilities   : w_reg = 0.2  (sector reg unreliable, 0.06 val IC)
"""

import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")


# ── Feature sets ───────────────────────────────────────────────────────────────

# v2 global model: trained with 149 features (24 dropped)
_DROP_V2 = [
    "close_position", "consecutive_up_days", "consecutive_down_days",
    "breakout_high_20d", "breakout_high_50d", "breakout_low_50d",
    "ichimoku_above_cloud", "ichimoku_tk_cross", "sma_cross_5_20",
    "return_20d", "momentum_20d", "rate_change_direction",
    "hy_credit_spread", "credit_spread_change_20d",
    "return_5d", "entropy_20d", "lower_shadow", "williams_r_14",
    "sector_vol_rank", "return_vs_sector_1d", "obv_slope_5",
    "momentum_5d", "roc_5", "price_vs_sma_20",
]

# v3 sector models: trained with 141 features (32 dropped)
_DROP_V3 = _DROP_V2 + [
    "return_10d", "channel_position_20d", "upper_shadow",
    "price_vs_ema_20", "stoch_d_14", "realized_vol_10d",
    "return_vs_sector_20d", "bb_upper",
]

_META    = ["ticker", "date", "Open", "High", "Low", "Close", "Volume", "sector"]
_TARGETS = [
    "fwd_return_5d", "fwd_label_5d", "fwd_up_5d", "fwd_down_5d",
    "fwd_return_10d", "fwd_label_10d", "fwd_up_10d", "fwd_down_10d",
    "fwd_return_20d", "fwd_label_20d", "fwd_up_20d", "fwd_down_20d",
    "fwd_sharpe_10d",
]


def _feat_cols(drop_list, df_columns):
    return [c for c in df_columns
            if c not in _META
            and c not in _TARGETS
            and not c.startswith("fwd_")
            and c not in drop_list]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))


# ── Sector routing config ─────────────────────────────────────────────────────

# val_ic from models/v3/results.json  (computed on 2020-2021 validation set)
_SECTOR_VAL_IC = {
    "Basic Materials":        0.1612,
    "Communication Services": 0.1152,
    "Consumer Cyclical":      0.1446,
    "Consumer Defensive":     0.0600,
    "Energy":                 0.0972,
    "Financial Services":    -0.0865,
    "Healthcare":             0.1287,
    "Industrials":            0.0881,
    "Real Estate":            0.1801,
    "Technology":             0.1584,
    "Utilities":              0.0600,
}

# Weight given to the regressor signal per sector
# Low or negative val IC → reduce/zero out regressor contribution
_REG_WEIGHT = {
    "Financial Services":    0.00,   # broken regressor (negative IC)
    "Utilities":             0.20,   # unreliable
    "Consumer Defensive":    0.25,   # below threshold
    "Communication Services": 0.30,
    "Consumer Cyclical":     0.35,
    # default: 0.40 for sectors with strong val IC
}
_REG_WEIGHT_DEFAULT = 0.40

# Use sector regressor if val_ic >= this threshold, else fall back to global v2
_SECTOR_REG_THRESHOLD = 0.07


def _sector_slug(sector: str) -> str:
    return sector.lower().replace(" ", "_")


# ── SignalGenerator ────────────────────────────────────────────────────────────

class SignalGenerator:
    """
    Scores all available stocks on a given date.

    Parameters
    ----------
    features_path : path to features_phase4.parquet
    prices_path   : path to sp500_clean_ohlcv.parquet (for ticker column)
    preload_start : only keep features from this date onward (saves memory)
    """

    def __init__(
        self,
        features_path: str = "data/features_phase4.parquet",
        prices_path:   str = "data/sp500_clean_ohlcv.parquet",
        preload_start: str = "2020-01-01",   # include val period for warm-up
    ):
        print("Loading models...")
        self._load_models()

        print("Loading features...")
        self._load_features(features_path, prices_path, preload_start)
        print(f"  Features ready: {len(self._features):,} rows, "
              f"{self._features['date'].nunique()} dates")

    # ── Model loading ──────────────────────────────────────────────────────────
    def _load_models(self):
        # Global v2 models
        self.clf_global = xgb.XGBClassifier()
        self.clf_global.load_model("models/v2/clf_10d.ubj")
        self.reg_global = xgb.XGBRegressor()
        self.reg_global.load_model("models/v2/reg_10d.ubj")

        # Sector v3 models
        self.sector_clfs: dict = {}
        self.sector_regs: dict = {}
        for sector in _SECTOR_VAL_IC:
            slug = _sector_slug(sector)
            clf_path = Path(f"models/v3/{slug}/clf_10d.ubj")
            reg_path = Path(f"models/v3/{slug}/reg_10d.ubj")
            if clf_path.exists():
                c = xgb.XGBClassifier(); c.load_model(str(clf_path))
                self.sector_clfs[sector] = c
            if reg_path.exists():
                r = xgb.XGBRegressor();  r.load_model(str(reg_path))
                self.sector_regs[sector] = r

        print(f"  Loaded: global v2 clf+reg  |  "
              f"{len(self.sector_clfs)} sector clfs  |  "
              f"{len(self.sector_regs)} sector regs")

    # ── Feature loading ────────────────────────────────────────────────────────
    def _load_features(self, feat_path, price_path, preload_start):
        feat = pd.read_parquet(feat_path)
        raw  = pd.read_parquet(price_path, columns=["ticker"])
        feat["ticker"] = raw["ticker"].values
        feat["date"]   = pd.to_datetime(feat["date"])

        # Keep only from preload_start
        feat = feat[feat["date"] >= preload_start].copy()

        # Drop rows with > 50% null features (warm-up period edge)
        all_feat_v3 = _feat_cols(_DROP_V3, feat.columns)
        null_ratio  = feat[all_feat_v3].isnull().mean(axis=1)
        feat = feat[null_ratio <= 0.5].copy()

        self._features      = feat
        self._feat_cols_v2  = _feat_cols(_DROP_V2, feat.columns)
        self._feat_cols_v3  = _feat_cols(_DROP_V3, feat.columns)

        # Build date index for fast lookup
        self._date_index = feat.groupby("date").indices   # date -> array of row indices

    # ── Scoring ───────────────────────────────────────────────────────────────
    def score(self, date: pd.Timestamp) -> Optional[pd.DataFrame]:
        """
        Score all active tickers on `date`.

        Returns
        -------
        DataFrame with columns [ticker, sector, clf_prob, reg_pred, score]
        sorted by score descending, or None if no data for that date.
        """
        if date not in self._date_index:
            return None

        idx  = self._date_index[date]
        rows = self._features.iloc[idx].copy()

        if len(rows) == 0:
            return None

        # ── Score each sector ──────────────────────────────────────────────────
        frames = []

        for sector, grp in rows.groupby("sector"):
            if len(grp) == 0:
                continue

            # Classifier (always sector model)
            clf = self.sector_clfs.get(sector, None)
            if clf is None:
                clf = self.clf_global
                X_clf = grp[self._feat_cols_v2]
            else:
                X_clf = grp[self._feat_cols_v3]
            clf_prob = clf.predict_proba(X_clf.fillna(0))[:, 1]

            # Regressor (sector if val_ic >= threshold, else global)
            val_ic    = _SECTOR_VAL_IC.get(sector, 0.0)
            use_sec_r = val_ic >= _SECTOR_REG_THRESHOLD
            reg_w     = _REG_WEIGHT.get(sector, _REG_WEIGHT_DEFAULT)

            if reg_w > 0:
                if use_sec_r and sector in self.sector_regs:
                    reg   = self.sector_regs[sector]
                    X_reg = grp[self._feat_cols_v3]
                else:
                    reg   = self.reg_global
                    X_reg = grp[self._feat_cols_v2]
                reg_pred = reg.predict(X_reg.fillna(0))
            else:
                reg_pred = np.zeros(len(grp))

            # Z-score regressor within sector (keeps regressor as within-sector
            # ranker; classifier already handles cross-sector comparison)
            reg_std = float(np.std(reg_pred))
            if reg_std > 1e-6:
                reg_z = (reg_pred - np.mean(reg_pred)) / reg_std
            else:
                reg_z = np.zeros(len(grp))

            # Composite score
            clf_w = 1.0 - reg_w
            score = clf_w * clf_prob + reg_w * _sigmoid(reg_z)

            sec_df = pd.DataFrame({
                "ticker":   grp["ticker"].values,
                "sector":   sector,
                "clf_prob": clf_prob,
                "reg_pred": reg_pred,
                "score":    score,
            })
            frames.append(sec_df)

        if not frames:
            return None

        out = (pd.concat(frames, ignore_index=True)
               .sort_values("score", ascending=False)
               .reset_index(drop=True))
        return out

    # ── Batch scoring (for analysis) ──────────────────────────────────────────
    def score_range(self, dates) -> pd.DataFrame:
        """Score all tickers for a list of dates. Returns concatenated DataFrame."""
        frames = []
        for d in dates:
            s = self.score(d)
            if s is not None:
                s["date"] = d
                frames.append(s)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def precompute(
        self,
        start: str = "2022-01-01",
        end:   str = "2026-05-01",
        cache_path: str = "models/scores_cache.parquet",
    ) -> pd.DataFrame:
        """
        Pre-compute scores for all dates in [start, end] and cache to disk.

        Subsequent calls with the same cache_path skip computation entirely.
        Returns a DataFrame with columns [date, ticker, sector, clf_prob, reg_pred, score].
        """
        from pathlib import Path
        import time

        cache = Path(cache_path)
        if cache.exists():
            print(f"Loading cached scores from {cache_path}...")
            df = pd.read_parquet(cache_path)
            print(f"  {len(df):,} rows, {df['date'].nunique()} dates")
            return df

        start_dt = pd.Timestamp(start)
        end_dt   = pd.Timestamp(end)
        all_dates = sorted(
            d for d in self._date_index.keys() if start_dt <= d <= end_dt
        )
        print(f"Pre-computing scores for {len(all_dates)} dates ({start} -> {end})...")
        t0 = time.time()
        frames = []
        for i, d in enumerate(all_dates):
            s = self.score(d)
            if s is not None:
                s["date"] = d
                frames.append(s)
            if (i + 1) % 200 == 0:
                elapsed = time.time() - t0
                remaining = elapsed / (i + 1) * (len(all_dates) - i - 1)
                print(f"  {i+1}/{len(all_dates)}  elapsed {elapsed:.0f}s  "
                      f"remaining ~{remaining:.0f}s")

        result = pd.concat(frames, ignore_index=True)
        result.to_parquet(cache_path)
        print(f"Scores saved to {cache_path}  ({len(result):,} rows, "
              f"{time.time()-t0:.0f}s total)")
        return result
