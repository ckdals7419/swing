"""
Signal Generator v2 — Walk-Forward (Phase 5)

Uses walk-forward model vintages (models/v4/) instead of static v3 models.
Each vintage covers a specific deploy period; the scorer automatically
swaps to the correct vintage for each scoring date.

Composite score (same formula as scorer.py):
    score = w_clf * clf_prob + w_reg * sigmoid(reg_z)

Routing:
    Classifier  : always use sector model
    Regressor   : use sector model if sector val_ic >= 0.07 (from vintage metadata)
                  use regressor weight 0.0 for Financial Services (broken IC per vintage)
"""

import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import json
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")


# ── Feature sets (same as v3) ──────────────────────────────────────────────────
_DROP = [
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

_META    = ["ticker", "date", "Open", "High", "Low", "Close", "Volume", "sector"]
_TARGETS = [
    "fwd_return_5d", "fwd_label_5d", "fwd_up_5d", "fwd_down_5d",
    "fwd_return_10d", "fwd_label_10d", "fwd_up_10d", "fwd_down_10d",
    "fwd_return_20d", "fwd_label_20d", "fwd_up_20d", "fwd_down_20d",
    "fwd_sharpe_10d",
]

# Static reg weight overrides (carry over from scorer.py analysis)
_REG_WEIGHT_OVERRIDE = {
    "Utilities": 0.20,
    "Consumer Defensive": 0.25,
    "Communication Services": 0.30,
    "Consumer Cyclical": 0.35,
}
_REG_WEIGHT_DEFAULT  = 0.40
_REG_IC_THRESHOLD    = 0.07   # use sector reg only if val_ic >= this


def _feat_cols(df_columns):
    return [c for c in df_columns
            if c not in _META
            and c not in _TARGETS
            and not c.startswith("fwd_")
            and c not in _DROP]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))


def _sector_slug(sector: str) -> str:
    return sector.lower().replace(" ", "_")


# ── Vintage manifest ───────────────────────────────────────────────────────────
def _load_manifest(models_dir: Path) -> list:
    manifest_path = models_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Walk-forward manifest not found at {manifest_path}.\n"
            f"Run: python train_model_v4.py"
        )
    with open(manifest_path) as f:
        manifest = json.load(f)
    # Sort by deploy_start so we can binary-search or iterate
    manifest.sort(key=lambda v: v["deploy_start"])
    return manifest


def _find_vintage(manifest: list, date: pd.Timestamp) -> Optional[dict]:
    """Return the vintage whose deploy window contains `date`."""
    date_str = date.strftime("%Y-%m-%d")
    for v in manifest:
        if v["deploy_start"] <= date_str <= v["deploy_end"]:
            return v
    # Fallback: use the earliest vintage if date is before any deploy window,
    # or the latest vintage if date is after all deploy windows
    if date_str < manifest[0]["deploy_start"]:
        return manifest[0]
    if date_str > manifest[-1]["deploy_end"]:
        return manifest[-1]
    return None


# ── SignalGeneratorV2 ──────────────────────────────────────────────────────────
class SignalGeneratorV2:
    """
    Walk-forward scorer. Uses the correct vintage of sector models for each
    scoring date. Models are loaded lazily and cached in memory; a cache
    miss only occurs when the date crosses a vintage boundary.

    Parameters
    ----------
    features_path : path to features_phase4.parquet
    models_dir    : directory containing models/v4/ walk-forward vintages
    preload_start : load features from this date (saves memory)
    """

    def __init__(
        self,
        features_path: str = "data/features_phase4.parquet",
        models_dir:    str = "models/v4",
        preload_start: str = "2020-01-01",
    ):
        self._models_dir   = Path(models_dir)
        self._manifest     = _load_manifest(self._models_dir)
        self._active_vid   = None        # currently loaded vintage id
        self._sector_clfs  = {}          # sector -> XGBClassifier
        self._sector_regs  = {}          # sector -> XGBRegressor | None
        self._sector_val_ic = {}         # sector -> val_ic for routing
        self._reg_w        = {}          # sector -> reg weight (cached)

        print(f"Walk-forward manifest: {len(self._manifest)} vintages loaded")
        for v in self._manifest:
            print(f"  {v['vintage_id']}  deploy {v['deploy_start']} → {v['deploy_end']}")

        print("\nLoading features...")
        self._load_features(features_path, preload_start)
        print(f"  Features ready: {len(self._features):,} rows, "
              f"{self._features['date'].nunique()} dates")

    # ── Feature loading ────────────────────────────────────────────────────────
    def _load_features(self, feat_path: str, preload_start: str):
        feat = pd.read_parquet(feat_path)
        # Restore ticker from prices file (same approach as scorer.py)
        raw  = pd.read_parquet("data/sp500_clean_ohlcv.parquet", columns=["ticker"])
        feat["ticker"] = raw["ticker"].values
        feat["date"]   = pd.to_datetime(feat["date"])

        feat = feat[feat["date"] >= preload_start].copy()

        all_feat = _feat_cols(feat.columns)
        null_ratio = feat[all_feat].isnull().mean(axis=1)
        feat = feat[null_ratio <= 0.5].copy()

        self._features     = feat
        self._feat_cols    = _feat_cols(feat.columns)
        self._date_index   = feat.groupby("date").indices

    # ── Vintage loading ────────────────────────────────────────────────────────
    def _load_vintage(self, vintage: dict):
        vid   = vintage["vintage_id"]
        vdir  = self._models_dir / vid
        val_ic_map = vintage.get("sector_val_ic", {})

        clfs = {}
        regs = {}
        reg_w = {}

        for sector in val_ic_map:
            slug = _sector_slug(sector)
            clf_path = vdir / slug / "clf_10d.ubj"
            reg_path = vdir / slug / "reg_10d.ubj"

            if not clf_path.exists():
                continue

            c = xgb.XGBClassifier()
            c.load_model(str(clf_path))
            clfs[sector] = c

            val_ic = val_ic_map.get(sector, 0.0)
            # Reg weight: override for problem sectors, else by IC
            if sector == "Financial Services" or val_ic < -0.02:
                rw = 0.0
            else:
                rw = _REG_WEIGHT_OVERRIDE.get(sector, _REG_WEIGHT_DEFAULT)

            reg_w[sector] = rw

            if rw > 0 and val_ic >= _REG_IC_THRESHOLD and reg_path.exists():
                r = xgb.XGBRegressor()
                r.load_model(str(reg_path))
                regs[sector] = r
            else:
                regs[sector] = None  # will fall back to clf-only

        self._sector_clfs   = clfs
        self._sector_regs   = regs
        self._sector_val_ic = val_ic_map
        self._reg_w         = reg_w
        self._active_vid    = vid

        print(f"  Loaded vintage {vid}: "
              f"{len(clfs)} sector clfs, "
              f"{sum(1 for r in regs.values() if r is not None)} sector regs")

    def _ensure_vintage(self, date: pd.Timestamp):
        """Load the correct vintage if not already loaded."""
        v = _find_vintage(self._manifest, date)
        if v is None:
            return  # keep whatever is loaded
        if v["vintage_id"] != self._active_vid:
            self._load_vintage(v)

    # ── Scoring ───────────────────────────────────────────────────────────────
    def score(self, date: pd.Timestamp) -> Optional[pd.DataFrame]:
        """
        Score all active tickers on `date` using the appropriate vintage.

        Returns DataFrame with [ticker, sector, clf_prob, reg_pred, score]
        sorted by score descending, or None if no data.
        """
        self._ensure_vintage(date)

        if date not in self._date_index:
            return None

        idx  = self._date_index[date]
        rows = self._features.iloc[idx].copy()

        if len(rows) == 0:
            return None

        frames = []
        for sector, grp in rows.groupby("sector"):
            clf = self._sector_clfs.get(sector)
            if clf is None:
                continue

            X = grp[self._feat_cols]
            clf_prob = clf.predict_proba(X.fillna(0))[:, 1]

            reg_w  = self._reg_w.get(sector, _REG_WEIGHT_DEFAULT)
            reg    = self._sector_regs.get(sector)

            if reg_w > 0 and reg is not None:
                reg_pred = reg.predict(X.fillna(0))
                reg_std  = float(np.std(reg_pred))
                reg_z    = (reg_pred - np.mean(reg_pred)) / reg_std if reg_std > 1e-6 else np.zeros(len(grp))
            else:
                reg_pred = np.zeros(len(grp))
                reg_z    = np.zeros(len(grp))
                reg_w    = 0.0

            clf_w = 1.0 - reg_w
            score = clf_w * clf_prob + reg_w * _sigmoid(reg_z)

            frames.append(pd.DataFrame({
                "ticker":   grp["ticker"].values,
                "sector":   sector,
                "clf_prob": clf_prob,
                "reg_pred": reg_pred,
                "score":    score,
            }))

        if not frames:
            return None

        return (pd.concat(frames, ignore_index=True)
                .sort_values("score", ascending=False)
                .reset_index(drop=True))

    # ── Batch / cache ─────────────────────────────────────────────────────────
    def precompute(
        self,
        start: str = "2022-01-01",
        end:   str = "2026-05-01",
        cache_path: str = "models/scores_cache_v2.parquet",
    ) -> pd.DataFrame:
        """Pre-compute and cache scores for all dates in [start, end]."""
        import time
        from pathlib import Path as _Path

        cache = _Path(cache_path)
        if cache.exists():
            print(f"Loading cached scores from {cache_path}...")
            df = pd.read_parquet(cache_path)
            print(f"  {len(df):,} rows, {df['date'].nunique()} dates")
            return df

        start_dt = pd.Timestamp(start)
        end_dt   = pd.Timestamp(end)
        all_dates = sorted(d for d in self._date_index.keys()
                           if start_dt <= d <= end_dt)

        print(f"Pre-computing scores for {len(all_dates)} dates ({start} → {end})...")
        t0 = time.time()
        frames = []
        for i, d in enumerate(all_dates):
            s = self.score(d)
            if s is not None:
                s["date"] = d
                frames.append(s)
            if (i + 1) % 200 == 0:
                elapsed   = time.time() - t0
                remaining = elapsed / (i + 1) * (len(all_dates) - i - 1)
                print(f"  {i+1}/{len(all_dates)}  "
                      f"elapsed {elapsed:.0f}s  remaining ~{remaining:.0f}s")

        result = pd.concat(frames, ignore_index=True)
        result.to_parquet(cache_path)
        print(f"Scores saved to {cache_path}  "
              f"({len(result):,} rows, {time.time()-t0:.0f}s total)")
        return result
