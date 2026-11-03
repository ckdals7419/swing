"""
Cross-Fold Validation — QLD SMA200 Strategy Across Market Regimes

Tests the SMA200 rule (and benchmark strategies) against each major bear market
+ 5 years of recovery, using synthetic 2x QLD for pre-2006 data.

Test windows (bear start → bear trough + 5 years):
  2000 dot-com   2000-01-03 → 2007-10-31  (tech crash + recovery)
  2008 GFC       2007-10-01 → 2014-03-31  (systemic crisis + recovery)
  2015 chop      2015-01-01 → 2021-02-28  (sideways + COVID spike)
  2018 QT        2018-01-01 → 2023-12-31  (fast correction + bull)
  2020 COVID     2020-01-01 → 2025-03-31  (volatility shock + bull)
  2022 inflation 2022-01-03 → latest      (rate regime + bull)

Strategies evaluated:
  SMA200         QLD weight = 1.0 when QQQ > SMA200, else 0.0
  SMA200+VIX     SMA200 plus VIX > 30 hard stop
  Rules-5lvl     5-level VIX/trend regime model
  VolTarget      Vol-targeting at 18% annualized QQQ vol
  QLD B&H        100% allocation, no rebalancing (baseline)

Synthetic QLD:
  For dates before 2006-06-19, QLD is reconstructed as:
    QLD[t] = QLD[t-1] × (1 + 2 × QQQ_daily_return[t])
  This approximates the 2x daily reset mechanism (ignores fee drag ~0.95%/yr).

Usage:
    python crossval_qld.py
    python crossval_qld.py --download    # re-download QQQ history
"""

import warnings
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
import xgboost as xgb
from pathlib import Path

warnings.filterwarnings("ignore")

QQQ_CACHE         = Path("data/qqq_full.parquet")
CROSS_ASSET_CACHE = Path("data/cross_asset.parquet")
MACRO_CACHE       = Path("data/macro_features.parquet")
VIX_PATH          = "data/vix_daily.csv"
UNIV_PATH         = "data/qld_universe.parquet"
COST_BPS          = 9.0   # per side

# ── Test windows ───────────────────────────────────────────────────────────────

WINDOWS = [
    # (label, test_start, test_end, bear_type)
    ("2000 dot-com",    "2000-01-03", "2007-10-31", "tech crash"),
    ("2008 GFC",        "2007-10-01", "2014-03-31", "systemic crisis"),
    ("2015 chop",       "2015-01-01", "2021-02-28", "sideways"),
    ("2018 QT",         "2018-01-01", "2023-12-31", "fast correction"),
    ("2020 COVID",      "2020-01-01", "2025-03-31", "volatility shock"),
    ("2022 inflation",  "2022-01-03", None,          "rate regime"),
]


# ── Data loading ───────────────────────────────────────────────────────────────

def download_qqq(force: bool = False) -> pd.DataFrame:
    """Download QQQ daily OHLCV back to 1999. Cache to disk."""
    if QQQ_CACHE.exists() and not force:
        df = pd.read_parquet(QQQ_CACHE)
        print(f"QQQ cache: {df['date'].min().date()} -> {df['date'].max().date()} "
              f"({len(df)} rows)")
        return df

    print("Downloading QQQ history (1999-present)...")
    raw = yf.download("QQQ", start="1999-03-10", progress=False, auto_adjust=True)
    raw.index = pd.to_datetime(raw.index)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]
    raw = raw.rename(columns={"Adj Close": "Adj_Close"})
    if "Adj_Close" not in raw.columns:
        raw["Adj_Close"] = raw["Close"]
    raw.index.name = "date"
    raw = raw.reset_index()
    if hasattr(raw["date"].dt, "tz") and raw["date"].dt.tz is not None:
        raw["date"] = raw["date"].dt.tz_localize(None)
    raw.to_parquet(QQQ_CACHE, index=False)
    print(f"  Saved to {QQQ_CACHE}: {raw['date'].min().date()} -> {raw['date'].max().date()}")
    return raw


def build_full_dataset(force_download: bool = False) -> pd.DataFrame:
    """
    Build a daily dataset from 1999 to present with:
      date, qqq_open, qqq_close, qqq_high, qqq_low,
      qld_open, qld_close,
      vix_close, bil_return_1d

    QLD is synthetic (2x QQQ returns) before 2006-06-19, real afterwards.
    """
    qqq_raw = download_qqq(force=force_download)
    qqq = qqq_raw.sort_values("date").set_index("date")

    # ── Actual QLD from universe file ─────────────────────────────────────────
    univ = pd.read_parquet(UNIV_PATH)
    univ["date"] = pd.to_datetime(univ["date"])
    qld_actual = (univ[univ["ticker"] == "QLD"]
                  .sort_values("date").set_index("date"))

    # ── BIL from universe (short-term T-bill return) ──────────────────────────
    bil_df = univ[univ["ticker"] == "BIL"].sort_values("date").set_index("date")
    bil_close = bil_df["Close"]

    # ── Synthetic QLD for pre-inception dates ─────────────────────────────────
    # Start synthetic QLD at the same price as actual QLD on its first day
    qld_start = qld_actual.index.min()
    qld_start_price = float(qld_actual.loc[qld_start, "Close"])

    # QQQ daily returns
    qqq_close = qqq["Close"]
    qqq_ret   = qqq_close.pct_change().fillna(0.0)

    # Build synthetic QLD from a base, anchored at inception
    # Work backwards from QLD inception
    dates_pre = qqq_close.index[qqq_close.index < qld_start]

    # Forward simulate from first available QQQ date
    first_qqq = qqq_close.index.min()
    if first_qqq in dates_pre or first_qqq < qld_start:
        # Normalise so that synthetic QLD on qld_start == qld_start_price
        synth_ret = qqq_ret * 2.0  # 2x daily return
        synth_cumret = (1 + synth_ret).cumprod()
        # Scale to match QLD price at inception
        scale = qld_start_price / float(synth_cumret.loc[qld_start])
        synth_close = synth_cumret * scale
        # Synthetic open: approximate as close × (qqq_open/qqq_close)
        qqq_oc_ratio = (qqq["Open"] / qqq["Close"]).fillna(1.0)
        synth_open   = synth_close * qqq_oc_ratio
    else:
        synth_close = pd.Series(dtype=float)
        synth_open  = pd.Series(dtype=float)

    # ── Splice synthetic + actual ─────────────────────────────────────────────
    qld_close = pd.concat([
        synth_close[synth_close.index < qld_start],
        qld_actual["Close"].rename("Close")
    ]).sort_index()

    qld_open = pd.concat([
        synth_open[synth_open.index < qld_start],
        qld_actual["Open"].rename("Open")
    ]).sort_index()

    # ── VIX ───────────────────────────────────────────────────────────────────
    vix_raw = pd.read_csv(VIX_PATH, parse_dates=["Date"])
    vix_raw = vix_raw.rename(columns={"Date": "date", "vix_close": "vix_close"})
    vix = vix_raw.set_index("date")["vix_close"]

    # ── Assemble master DataFrame ──────────────────────────────────────────────
    idx = qqq_close.index   # use QQQ trading calendar

    df = pd.DataFrame(index=idx)
    df["qqq_close"] = qqq_close.reindex(idx)
    df["qqq_open"]  = qqq["Open"].reindex(idx)
    df["qqq_high"]  = qqq["High"].reindex(idx)
    df["qqq_low"]   = qqq["Low"].reindex(idx)
    df["qld_close"] = qld_close.reindex(idx, method="ffill")
    df["qld_open"]  = qld_open.reindex(idx, method="ffill")
    df["vix_close"] = vix.reindex(idx, method="ffill")

    # BIL return: use actual BIL when available, else risk-free approximation
    bil_ret = bil_close.pct_change().reindex(idx, method="ffill")
    # Pre-BIL: approximate as fed_funds / 252 (use 0 as floor)
    df["bil_return_1d"] = bil_ret.fillna(0.0001)   # ~2.5% annualised floor

    # SMA200 and related features
    df["sma200"]        = df["qqq_close"].rolling(200, min_periods=200).mean()
    df["sma50"]         = df["qqq_close"].rolling(50,  min_periods=50).mean()
    df["qqq_vs_sma200"] = df["qqq_close"] / df["sma200"] - 1
    df["qqq_vs_sma50"]  = df["qqq_close"] / df["sma50"]  - 1
    df["qqq_drawdown"]  = df["qqq_close"] / df["qqq_close"].rolling(252, min_periods=1).max() - 1

    rv20 = (np.log(df["qqq_close"] / df["qqq_close"].shift(1))
            .rolling(20).std() * np.sqrt(252))
    df["realized_vol_20d"] = rv20

    # Add QQQ volume for BTC-style volume features
    df["qqq_volume"] = qqq["Volume"].reindex(idx)

    df["is_synthetic_qld"] = df.index < qld_start

    df = df.dropna(subset=["qqq_close", "qld_close"])
    df.index.name = "date"

    # ── Feature engineering ───────────────────────────────────────────────────
    df = add_btc_style_features(df)
    df = add_cross_asset_features(df, force=force_download)
    df = add_second_order_features(df)
    df = add_regime_features(df)
    df = add_macro_features(df, force=force_download)

    # ── Dual-model targets: 5-day forward ±2% thresholds ────────────────────
    fwd_5d = df["qld_close"].shift(-5) / df["qld_close"] - 1
    df["target_bull"] = (fwd_5d >  0.02).astype(float)  # P(return > +2% in 5 days)
    df["target_bear"] = (fwd_5d < -0.02).astype(float)  # P(return < -2% in 5 days)

    print(f"Full dataset: {df.index.min().date()} -> {df.index.max().date()} "
          f"({len(df)} rows, QLD synthetic before {qld_start.date()})")
    synth_count = df["is_synthetic_qld"].sum()
    print(f"  Synthetic QLD rows: {synth_count} ({synth_count/len(df):.0%})")
    bull_rate = df["target_bull"].dropna().mean()
    bear_rate = df["target_bear"].dropna().mean()
    print(f"  Bull target rate (>+2% in 5d): {bull_rate:.1%}   "
          f"Bear target rate (<-2% in 5d): {bear_rate:.1%}")
    return df


# ── BTC-style feature engineering ────────────────────────────────────────────
# Mirrors feature_extractor.py from BTC project, adapted for daily QQQ bars.
# Lookback periods translated from BTC's minute-bars [3,5,10,20,30,60,120,240]
# to equivalent trading days.

BTC_PERIODS = [3, 5, 10, 20, 30, 60, 120, 240]


def _rsi_btc(prices: pd.Series, period: int) -> pd.Series:
    """RSI as used in BTC feature_extractor: ewm alpha=1/period."""
    delta = prices.diff()
    gain  = delta.where(delta > 0, 0.0).ewm(alpha=1/period, adjust=False).mean()
    loss  = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _btc_features_from_series(close: pd.Series, high: pd.Series,
                               low: pd.Series, volume: pd.Series,
                               vix: pd.Series) -> pd.DataFrame:
    """
    Generic BTC-style feature extraction. Works for any daily price series
    (QQQ, SPY, or any other equity). All price features are normalised to
    dimensionless ratios so they generalise across price levels and eras.
    """
    out  = pd.DataFrame(index=close.index)
    vol  = volume.replace(0, np.nan)
    ret1 = close.pct_change()

    # ── Per-period features (8 periods x 9 = 72) ────────────────────────────
    for p in BTC_PERIODS:
        s = f"{p}d"

        out[f"return_{s}"]       = close.pct_change(p)
        out[f"volatility_{s}"]   = ret1.rolling(p).std()
        out[f"volume_ratio_{s}"] = vol / vol.rolling(p).mean()

        roll_vol  = vol.rolling(p).sum().replace(0, np.nan)
        roll_cvol = (close * vol).rolling(p).sum()
        vwap_p    = roll_cvol / roll_vol
        out[f"vwap_{s}"]         = close / vwap_p.replace(0, np.nan) - 1

        sma_p = close.rolling(p).mean()
        std_p = close.rolling(p).std().replace(0, np.nan)
        out[f"sma_{s}"]          = close / sma_p.replace(0, np.nan) - 1
        out[f"hlrange_{s}"]      = (high.rolling(p).max() -
                                    low.rolling(p).min()) / close
        out[f"rsi_{s}"]          = _rsi_btc(close, p)
        bb_upper_p = sma_p + 2 * std_p
        bb_lower_p = sma_p - 2 * std_p
        out[f"bb_upper_{s}"]     = (close - bb_upper_p) / close
        out[f"bb_lower_{s}"]     = (close - bb_lower_p) / close

    # ── MACD ────────────────────────────────────────────────────────────────
    exp12 = close.ewm(span=12, adjust=False).mean()
    exp26 = close.ewm(span=26, adjust=False).mean()
    out["macd"]        = exp12 - exp26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"]   = out["macd"] - out["macd_signal"]

    # ── DMI / ADX ────────────────────────────────────────────────────────────
    hi_diff  = high.diff()
    lo_diff  = low.diff()
    plus_dm  = hi_diff.where(hi_diff > lo_diff, 0.0).clip(lower=0)
    minus_dm = (-lo_diff).where(lo_diff > hi_diff, 0.0).clip(lower=0)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    p14   = 14
    spdm  = plus_dm.ewm(alpha=1/p14, adjust=False).mean()
    smdm  = minus_dm.ewm(alpha=1/p14, adjust=False).mean()
    str_  = tr.ewm(alpha=1/p14, adjust=False).mean().replace(0, np.nan)
    out["dmi_plus"]  = 100 * spdm / str_
    out["dmi_minus"] = 100 * smdm / str_
    out["adx"]       = (out["dmi_plus"] - out["dmi_minus"]).abs().ewm(
                           alpha=1/p14, adjust=False).mean()

    # ── VIX features ─────────────────────────────────────────────────────────
    out["vix_close"]      = vix
    out["vix_sma20"]      = vix.rolling(20, min_periods=5).mean()
    out["vix_percentile"] = vix.rolling(252, min_periods=50).rank(pct=True)
    out["vix_change_5d"]  = vix.pct_change(5)

    # ── Drawdown ─────────────────────────────────────────────────────────────
    rolling_max = close.rolling(252, min_periods=1).max()
    at_high     = (close >= rolling_max).astype(float)
    days_dd     = at_high.groupby((at_high != at_high.shift()).cumsum()).cumcount()
    out["qqq_drawdown"]      = close / rolling_max - 1
    out["days_in_drawdown"]  = days_dd * (1 - at_high)

    return out


def add_btc_style_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply BTC-style features to the main QQQ/QLD dataset."""
    feats = _btc_features_from_series(
        df["qqq_close"], df["qqq_high"], df["qqq_low"],
        df["qqq_volume"], df["vix_close"],
    )
    for col in feats.columns:
        df[col] = feats[col]
    return df


def _dl_etf(ticker: str, start: str, idx: pd.Index) -> pd.DataFrame:
    """Download a single ETF and reindex to QQQ calendar. Returns OHLCV df."""
    raw = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    raw.index = pd.to_datetime(raw.index)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]
    if hasattr(raw.index, "tz") and raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)
    raw.index.name = "date"
    return raw.reindex(idx).ffill()


def add_cross_asset_features(df: pd.DataFrame,
                              force: bool = False) -> pd.DataFrame:
    """
    Download TLT, HYG, GLD, SPY, IWM and add cross-asset features to df.
    For dates before each ETF's inception, features are filled with 0.

    Features added (22 total):
        TLT (bonds):         tlt_return_5d, tlt_return_20d, tlt_vs_sma50, tlt_vol_20d
        HYG (hi-yield):      hyg_return_5d, hyg_return_20d, hyg_vs_sma50
        GLD (gold):          gld_return_5d, gld_return_20d
        SPY ratio:           qqq_vs_spy_5d, qqq_vs_spy_20d, qqq_vs_spy_sma20
        Credit proxy:        hyg_minus_tlt_5d
        [NEW] Flight safety: tlt_qqq_corr_20d, tlt_qqq_corr_60d,
                             hyg_tlt_spread_accel, risk_off_score
        [NEW] Breadth:       iwm_return_5d, iwm_return_20d,
                             qqq_spy_corr_20d, iwm_vs_spy_20d
    """
    # Cache invalidation: rebuild if new columns are missing
    _new_cols = {"tlt_qqq_corr_20d", "iwm_return_5d"}
    if CROSS_ASSET_CACHE.exists() and not force:
        ca_check = pd.read_parquet(CROSS_ASSET_CACHE)
        if _new_cols.issubset(set(ca_check.columns)):
            ca = ca_check.reindex(df.index)
            for col in ca.columns:
                df[col] = ca[col].fillna(0.0)
            print(f"Cross-asset cache loaded ({len(ca.columns)} features)")
            return df
        print("Cross-asset cache stale (missing new columns) - rebuilding...")

    print("Downloading cross-asset ETFs (TLT/HYG/GLD/SPY/IWM)...")
    idx = df.index

    tlt = _dl_etf("TLT", "2002-07-01", idx)["Close"]
    hyg = _dl_etf("HYG", "2007-04-01", idx)["Close"]
    gld = _dl_etf("GLD", "2004-11-01", idx)["Close"]
    spy = _dl_etf("SPY", "1999-01-01", idx)["Close"]
    iwm = _dl_etf("IWM", "2000-05-01", idx)["Close"]
    qqq = df["qqq_close"]

    qqq_ret1 = qqq.pct_change()
    tlt_ret1 = tlt.pct_change()
    spy_ret1 = spy.pct_change()

    ca = pd.DataFrame(index=idx)

    # TLT features
    ca["tlt_return_5d"]   = tlt.pct_change(5)
    ca["tlt_return_20d"]  = tlt.pct_change(20)
    ca["tlt_vs_sma50"]    = tlt / tlt.rolling(50).mean() - 1
    ca["tlt_vol_20d"]     = tlt_ret1.rolling(20).std()

    # HYG features
    ca["hyg_return_5d"]   = hyg.pct_change(5)
    ca["hyg_return_20d"]  = hyg.pct_change(20)
    ca["hyg_vs_sma50"]    = hyg / hyg.rolling(50).mean() - 1

    # GLD features
    ca["gld_return_5d"]   = gld.pct_change(5)
    ca["gld_return_20d"]  = gld.pct_change(20)

    # QQQ vs SPY (growth vs value rotation)
    qqq_r5  = qqq.pct_change(5)
    spy_r5  = spy.pct_change(5)
    qqq_r20 = qqq.pct_change(20)
    spy_r20 = spy.pct_change(20)
    ratio   = (qqq / spy).replace(0, np.nan)
    ca["qqq_vs_spy_5d"]    = qqq_r5  - spy_r5
    ca["qqq_vs_spy_20d"]   = qqq_r20 - spy_r20
    ca["qqq_vs_spy_sma20"] = ratio / ratio.rolling(20).mean() - 1

    # Credit stress proxy
    ca["hyg_minus_tlt_5d"] = ca["hyg_return_5d"] - ca["tlt_return_5d"]

    # ── Group 3: Flight-to-Safety / Risk Regime ──────────────────────────────
    # Rolling QQQ-TLT correlation: negative = risk-off (bonds up, stocks down)
    ca["tlt_qqq_corr_20d"] = qqq_ret1.rolling(20).corr(tlt_ret1)
    ca["tlt_qqq_corr_60d"] = qqq_ret1.rolling(60).corr(tlt_ret1)

    # HYG-TLT spread acceleration: credit stress worsening
    hyg_tlt_spread = hyg.pct_change(20) - tlt.pct_change(20)
    ca["hyg_tlt_spread_accel"] = hyg_tlt_spread - hyg_tlt_spread.shift(10)

    # Risk-off composite: TLT rising + negative TLT-QQQ correlation = active flight
    # Positive value = risk-off pressure increasing
    ca["risk_off_score"] = -ca["tlt_qqq_corr_20d"] * ca["tlt_return_5d"]

    # ── Group 5: Market Breadth Proxy ────────────────────────────────────────
    # IWM (Russell 2000): small-cap risk appetite
    iwm_ret1 = iwm.pct_change()
    ca["iwm_return_5d"]  = iwm.pct_change(5)
    ca["iwm_return_20d"] = iwm.pct_change(20)

    # QQQ vs SPY rolling correlation: low/negative = sector divergence, risk rising
    ca["qqq_spy_corr_20d"] = qqq_ret1.rolling(20).corr(spy_ret1)

    # IWM vs SPY 20d: small-cap outperformance = broad risk appetite
    ca["iwm_vs_spy_20d"] = iwm.pct_change(20) - spy_r20

    # Fill pre-inception NaNs with 0 (neutral signal)
    ca = ca.fillna(0.0)
    ca.to_parquet(CROSS_ASSET_CACHE)
    print(f"  Saved cross-asset features ({len(ca.columns)} cols, "
          f"{(ca != 0).any(axis=1).sum()} non-zero rows)")

    for col in ca.columns:
        df[col] = ca[col]
    return df


def add_second_order_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Second-order momentum features derived entirely from existing df columns.
    No new downloads required.

    Features added (12 total):
        RSI acceleration:     rsi_accel_5_20, rsi_accel_5_60, rsi_accel_20_60
        Vol regime:           vol_expand_5_20, vol_expand_10_60, vol_expand_20_120
        Return acceleration:  ret_accel_5_20, ret_accel_20_60
        Trend consistency:    up_days_20, up_days_60
        Volume confirmation:  vol_confirm_5d, vol_confirm_20d
    """
    ret1 = df["qqq_close"].pct_change()

    # RSI acceleration (short RSI - long RSI: positive = momentum building)
    df["rsi_accel_5_20"]  = df["rsi_5d"]  - df["rsi_20d"]
    df["rsi_accel_5_60"]  = df["rsi_5d"]  - df["rsi_60d"]
    df["rsi_accel_20_60"] = df["rsi_20d"] - df["rsi_60d"]

    # Volatility regime (short vol / long vol: > 0 = vol expanding)
    df["vol_expand_5_20"]   = df["volatility_5d"]   / df["volatility_20d"].replace(0, np.nan)  - 1
    df["vol_expand_10_60"]  = df["volatility_10d"]  / df["volatility_60d"].replace(0, np.nan)  - 1
    df["vol_expand_20_120"] = df["volatility_20d"]  / df["volatility_120d"].replace(0, np.nan) - 1

    # Return acceleration (recent return vs longer-period normalised return)
    df["ret_accel_5_20"]  = df["return_5d"]  - df["return_20d"]  / 4
    df["ret_accel_20_60"] = df["return_20d"] - df["return_60d"]  / 3

    # Trend consistency (fraction of last N days with positive close-to-close)
    up = (ret1 > 0).astype(float)
    df["up_days_20"] = up.rolling(20).mean()
    df["up_days_60"] = up.rolling(60).mean()

    # Volume-confirmed momentum (price move × relative volume)
    df["vol_confirm_5d"]  = df["return_5d"]  * df["volume_ratio_5d"]
    df["vol_confirm_20d"] = df["return_20d"] * df["volume_ratio_20d"]

    return df


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Regime detection features using only existing QQQ/SMA/VIX columns.
    No new downloads required.

    Group 1 - Drawdown Regime (5 features):
        drawdown_velocity_10d, drawdown_velocity_20d, log_days_in_dd,
        dd_depth_x_duration, recovery_ratio_20d

    Group 2 - Trend Structure (5 features):
        sma200_slope_20d, sma50_slope_20d, sma50_vs_sma200,
        price_dist_sma200, trend_quality_60d

    Group 4 - VIX Regime (4 features):
        vix_velocity_5d, vix_persistence_20d,
        vix_regime_shift, vix_above_sma20_pct
    """
    close  = df["qqq_close"]
    dd     = df["qqq_drawdown"]
    days   = df["days_in_drawdown"]
    vix    = df["vix_close"]
    sma200 = df["sma200"]
    sma50  = df["sma50"]

    # ── Group 1: Drawdown Regime ─────────────────────────────────────────────
    # How fast is the drawdown deepening? (negative = worsening)
    df["drawdown_velocity_10d"] = dd - dd.shift(10)
    df["drawdown_velocity_20d"] = dd - dd.shift(20)

    # Log-transformed days in drawdown: emphasises long bear regimes
    df["log_days_in_dd"] = np.log1p(days)

    # Depth × duration interaction: large negative = deep AND prolonged bear
    df["dd_depth_x_duration"] = dd * np.log1p(days)

    # Where in the 20-day high-low range is price? 0=at 20d low, 1=at 20d high
    hi20 = close.rolling(20).max()
    lo20 = close.rolling(20).min()
    rng  = (hi20 - lo20).replace(0, np.nan)
    df["recovery_ratio_20d"] = (close - lo20) / rng

    # ── Group 2: Trend Structure ─────────────────────────────────────────────
    # SMA slope: positive = uptrend in progress, negative = trend turning
    df["sma200_slope_20d"] = sma200 / sma200.shift(20) - 1
    df["sma50_slope_20d"]  = sma50  / sma50.shift(20)  - 1

    # Golden/death cross proximity: positive = golden cross, negative = death cross
    df["sma50_vs_sma200"]  = sma50 / sma200 - 1

    # Absolute distance from SMA200: large = overextended (mean-reversion risk)
    df["price_dist_sma200"] = (close / sma200 - 1).abs()

    # Trend quality: risk-adjusted momentum (60-day Sharpe-like)
    ret1 = close.pct_change()
    roll_ret60 = ret1.rolling(60).mean()
    roll_std60 = ret1.rolling(60).std().replace(0, np.nan)
    df["trend_quality_60d"] = roll_ret60 / roll_std60

    # ── Group 4: VIX Regime ──────────────────────────────────────────────────
    # VIX rate of change: spiking vs persistent
    df["vix_velocity_5d"] = vix.pct_change(5)

    # Fraction of last 20 days where VIX > vix_sma20: persistent high vol = bear regime
    vix_sma20 = df["vix_sma20"]
    df["vix_persistence_20d"] = (vix > vix_sma20).rolling(20).mean()

    # Structural VIX level shift: 60-day SMA change (regime, not spike)
    vix_sma60 = vix.rolling(60, min_periods=10).mean()
    df["vix_regime_shift"] = vix_sma60 / vix_sma60.shift(60) - 1

    # % of days above sma20 over past 60 days (persistently elevated = danger)
    df["vix_above_sma20_pct"] = (vix > vix_sma20).rolling(60).mean()

    return df


# ── Macro / economic feature engineering ─────────────────────────────────────

_FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# (fred_id, col_name, publication_lag_days)
# lag_days = how many calendar days AFTER the observation_date the data
# is actually publicly released.  We shift the index forward by this
# many days so the model never sees data before it was available.
#   Market yields / prices -> 0  (real-time)
#   Weekly releases        -> 6-10 days
#   Monthly releases       -> 30-45 days
_FRED_SERIES = [
    # id             col_name                lag   desc
    ("T10Y2Y",       "fred_yield_curve",       0),  # market yield, real-time
    ("DFII10",       "fred_real_rate_10y",      0),  # TIPS yield, real-time
    ("T10YIE",       "fred_breakeven_10y",      0),  # breakeven, real-time
    ("ICSA",         "fred_init_claims",        6),  # weekly, ~6d lag
    ("UNRATE",       "fred_unemployment",      35),  # monthly, 1st Fri next month
    ("WALCL",        "fred_fed_bs",             8),  # weekly, ~8d lag
    ("AMTMNO",       "fred_mfg_orders",        45),  # monthly, ~6 wk lag
    ("NFCI",         "fred_nfci",              10),  # weekly, ~10d lag
    # HY credit spread: computed from HYG/LQD ratio below (market prices, 0 lag)
]

MACRO_ADDITIONS = [
    # Yield curve
    "yc_10y2y", "yc_slope_20d", "yc_inversion_days",
    # Credit spread
    "cs_hy_level", "cs_hy_change_20d", "cs_hy_zscore_1y",
    # Real rates / inflation expectations
    "real_rate_10y", "real_rate_change_20d",
    "breakeven_10y", "breakeven_change_20d",
    # Labor market
    "claims_log", "claims_trend_13w", "claims_4w_chg",
    "unemp_level", "unemp_trend_3m",
    # Fed balance sheet / liquidity
    "fed_bs_pct_4w", "fed_bs_pct_13w",
    # Manufacturing new orders (PMI proxy)
    "mfg_orders_trend_3m", "mfg_orders_trend_12m",
    # Financial conditions
    "nfci_level", "nfci_trend_4w",
]


def _dl_fred(series_id: str, idx: pd.Index,
             lag_days: int = 0,
             observation_start: str = "1998-01-01") -> pd.Series:
    """
    Download a FRED series and reindex to trading-day calendar idx via ffill.

    lag_days: publication lag in calendar days.  The observation_date in FRED
    is the *reference period* (e.g. "2024-05-01" for May unemployment), but
    the data is only publicly available lag_days later (e.g. early June).
    We shift every observation forward by lag_days so the model never
    uses data before its actual release.
    """
    import io
    try:
        import requests as _req
        url = (f"{_FRED_BASE}?id={series_id}"
               f"&observation_start={observation_start}")
        resp = _req.get(url, timeout=30)
        resp.raise_for_status()
        raw_df = pd.read_csv(io.StringIO(resp.text), na_values=[".", ""])
        date_col = raw_df.columns[0]
        val_col  = raw_df.columns[1]
        raw_df[date_col] = pd.to_datetime(raw_df[date_col])
        raw_df = raw_df.set_index(date_col)
        s = raw_df[val_col].astype(float).dropna()
        if len(s) == 0:
            raise ValueError("no valid observations")

        # Apply publication lag: shift observation dates forward
        if lag_days > 0:
            s.index = s.index + pd.Timedelta(days=lag_days)

        # Merge idx with FRED dates, ffill across union, then select idx only
        combined = s.reindex(s.index.union(idx)).ffill()
        result = combined.reindex(idx)
        lag_str = f" (lag={lag_days}d)" if lag_days > 0 else ""
        print(f"    FRED {series_id}{lag_str}: {s.index.min().date()} -> {s.index.max().date()} "
              f"({s.notna().sum()} obs, {result.notna().sum()} trading days)")
        return result
    except Exception as e:
        print(f"    FRED {series_id} failed: {e}")
        return pd.Series(np.nan, index=idx)


def add_macro_features(df: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    """
    Download macroeconomic data from FRED and engineer features:
      Yield curve, credit spread, real rates, breakeven inflation,
      initial claims, unemployment, Fed balance sheet, ISM PMI,
      financial conditions index.

    All series are forward-filled (low-frequency -> daily).
    Pre-inception periods filled with 0 (neutral/unknown).
    """
    # Cache key: include "v2" to force rebuild with publication-lag correction
    _check_col = "macro_v2_lagged"
    if MACRO_CACHE.exists() and not force:
        cached = pd.read_parquet(MACRO_CACHE)
        if _check_col in cached.columns:
            macro = cached.reindex(df.index).ffill().fillna(0.0)
            for c in macro.columns:
                if c != _check_col:
                    df[c] = macro[c]
            print(f"Macro feature cache loaded ({len(macro.columns)-1} features, lag-corrected)")
            return df
        print("Macro cache stale (needs lag correction) - rebuilding...")

    print("Downloading macro features from FRED + yfinance (with publication lags)...")
    idx = df.index

    raw = {col: _dl_fred(sid, idx, lag_days=lag)
           for sid, col, lag in _FRED_SERIES}

    # ── HY credit spread proxy: HYG / LQD log-ratio (yfinance, full history) ─
    print("    Computing HY spread proxy (HYG/LQD)...")
    try:
        lqd = _dl_etf("LQD", "2002-07-01", idx)["Close"]
        hyg_close = _dl_etf("HYG", "2007-04-01", idx)["Close"]
        # Log ratio: higher = wider spread (more risk premium) = more bearish signal
        hy_lqd_ratio = np.log(hyg_close / lqd)
        raw["fred_hy_spread"] = hy_lqd_ratio
        print(f"    HYG/LQD ratio: {hy_lqd_ratio.notna().sum()} obs")
    except Exception as e:
        print(f"    HYG/LQD failed: {e}")
        raw["fred_hy_spread"] = pd.Series(np.nan, index=idx)

    macro = pd.DataFrame(index=idx)

    # ── Yield curve ──────────────────────────────────────────────────────────
    yc = raw["fred_yield_curve"]
    macro["yc_10y2y"]         = yc
    macro["yc_slope_20d"]     = yc.diff(20)
    # Consecutive days the curve has been inverted (negative spread)
    inv = (yc < 0).astype(float)
    macro["yc_inversion_days"] = inv.groupby(
        (inv != inv.shift()).cumsum()).cumcount() * inv

    # ── Credit spread proxy (HYG/LQD log-ratio) ──────────────────────────────
    # More negative = HY underperforming IG = wider credit spread = risk-off
    cs = raw["fred_hy_spread"]
    macro["cs_hy_level"]      = cs
    macro["cs_hy_change_20d"] = cs.diff(20)
    cs_mean = cs.rolling(252, min_periods=60).mean()
    cs_std  = cs.rolling(252, min_periods=60).std().replace(0, np.nan)
    macro["cs_hy_zscore_1y"]  = (cs - cs_mean) / cs_std

    # ── Real rates / inflation expectations ───────────────────────────────────
    rr = raw["fred_real_rate_10y"]
    macro["real_rate_10y"]      = rr
    macro["real_rate_change_20d"] = rr.diff(20)

    be = raw["fred_breakeven_10y"]
    macro["breakeven_10y"]      = be
    macro["breakeven_change_20d"] = be.diff(20)

    # ── Initial jobless claims ────────────────────────────────────────────────
    ic = raw["fred_init_claims"]
    macro["claims_log"]       = np.log(ic.clip(lower=1))
    macro["claims_4w_chg"]    = ic.pct_change(20).clip(-1, 1)   # ~4 weeks biz days
    macro["claims_trend_13w"] = ic.pct_change(65).clip(-1, 1)   # ~13 weeks biz days

    # ── Unemployment ──────────────────────────────────────────────────────────
    ur = raw["fred_unemployment"]
    macro["unemp_level"]    = ur
    macro["unemp_trend_3m"] = ur.diff(63)    # ~3 months

    # ── Fed balance sheet ─────────────────────────────────────────────────────
    fb = raw["fred_fed_bs"]
    macro["fed_bs_pct_4w"]  = fb.pct_change(20).clip(-0.5, 0.5)
    macro["fed_bs_pct_13w"] = fb.pct_change(65).clip(-0.5, 0.5)

    # ── Manufacturing new orders (PMI proxy) ──────────────────────────────────
    mo = raw["fred_mfg_orders"].replace(0, np.nan)
    macro["mfg_orders_trend_3m"]  = mo.pct_change(63).clip(-1, 1)   # 3-month growth
    macro["mfg_orders_trend_12m"] = mo.pct_change(252).clip(-1, 1)  # 12-month growth

    # ── Financial conditions (NFCI) ───────────────────────────────────────────
    nfci = raw["fred_nfci"]
    macro["nfci_level"]   = nfci
    macro["nfci_trend_4w"] = nfci.diff(20)

    # Forward-fill any remaining gaps, then fill pre-inception with 0 (neutral)
    macro = macro.ffill().fillna(0.0)
    macro["macro_v2_lagged"] = 1.0   # cache version marker
    macro.to_parquet(MACRO_CACHE)
    print(f"  Saved macro features ({len(macro.columns)-1} cols, lag-corrected)")

    for c in macro.columns:
        df[c] = macro[c]
    return df


# ── Strategy weight functions ──────────────────────────────────────────────────

def w_sma200(df: pd.DataFrame) -> pd.Series:
    return (df["qqq_vs_sma200"] >= 0).astype(float)


def w_sma200_vix(df: pd.DataFrame) -> pd.Series:
    vs200 = df["qqq_vs_sma200"]
    vix   = df["vix_close"]
    return ((vs200 >= 0) & (vix < 30)).astype(float)


def w_rules5(df: pd.DataFrame) -> pd.Series:
    """5-level VIX/trend regime model."""
    vix   = df["vix_close"].fillna(20.0)
    vs200 = df["qqq_vs_sma200"].fillna(0.0)
    vs50  = df["qqq_vs_sma50"].fillna(0.0)
    dd    = df["qqq_drawdown"].fillna(0.0)

    w = pd.Series(1.0, index=df.index)
    w = w.where(~(vs50 < 0), 0.75)
    w = w.where(~((vs200 < 0) | (vix > 22)), 0.50)
    w = w.where(~((vs200 < 0) & (vix > 22)), 0.25)
    w = w.where(~((vix > 30) | (dd < -0.15)), 0.0)
    return w


def w_voltarget(df: pd.DataFrame, target_vol: float = 0.18) -> pd.Series:
    rv = df["realized_vol_20d"].replace(0, np.nan).ffill().fillna(0.20)
    return (target_vol / rv).clip(0.0, 1.0)


def w_bah(_df: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=_df.index)


# ── ML multi-timeframe model ────────────────────────────────────────────────

ML_FEATURES_BASE = [
    # Per-period features (8 periods x 9 features = 72)
    *[f"return_{p}d"       for p in BTC_PERIODS],
    *[f"volatility_{p}d"   for p in BTC_PERIODS],
    *[f"volume_ratio_{p}d" for p in BTC_PERIODS],
    *[f"vwap_{p}d"         for p in BTC_PERIODS],
    *[f"sma_{p}d"          for p in BTC_PERIODS],
    *[f"hlrange_{p}d"      for p in BTC_PERIODS],
    *[f"rsi_{p}d"          for p in BTC_PERIODS],
    *[f"bb_upper_{p}d"     for p in BTC_PERIODS],
    *[f"bb_lower_{p}d"     for p in BTC_PERIODS],
    # Global technical indicators (6)
    "macd", "macd_signal", "macd_hist",
    "dmi_plus", "dmi_minus", "adx",
    # VIX features (4)
    "vix_close", "vix_sma20", "vix_percentile", "vix_change_5d",
    # Drawdown (2)
    "qqq_drawdown", "days_in_drawdown",
]

CROSS_ASSET_ADDITIONS = [
    "tlt_return_5d", "tlt_return_20d", "tlt_vs_sma50", "tlt_vol_20d",
    "hyg_return_5d", "hyg_return_20d", "hyg_vs_sma50",
    "gld_return_5d", "gld_return_20d",
    "qqq_vs_spy_5d", "qqq_vs_spy_20d", "qqq_vs_spy_sma20",
    "hyg_minus_tlt_5d",
    # Group 3: Flight-to-Safety
    "tlt_qqq_corr_20d", "tlt_qqq_corr_60d",
    "hyg_tlt_spread_accel", "risk_off_score",
    # Group 5: Market Breadth
    "iwm_return_5d", "iwm_return_20d",
    "qqq_spy_corr_20d", "iwm_vs_spy_20d",
]

SECOND_ORDER_ADDITIONS = [
    "rsi_accel_5_20", "rsi_accel_5_60", "rsi_accel_20_60",
    "vol_expand_5_20", "vol_expand_10_60", "vol_expand_20_120",
    "ret_accel_5_20", "ret_accel_20_60",
    "up_days_20", "up_days_60",
    "vol_confirm_5d", "vol_confirm_20d",
]

REGIME_ADDITIONS = [
    # Group 1: Drawdown Regime
    "drawdown_velocity_10d", "drawdown_velocity_20d",
    "log_days_in_dd", "dd_depth_x_duration", "recovery_ratio_20d",
    # Group 2: Trend Structure
    "sma200_slope_20d", "sma50_slope_20d", "sma50_vs_sma200",
    "price_dist_sma200", "trend_quality_60d",
    # Group 4: VIX Regime
    "vix_velocity_5d", "vix_persistence_20d",
    "vix_regime_shift", "vix_above_sma20_pct",
]

# Active feature set (overridden per run in main)
ML_FEATURES = ML_FEATURES_BASE

# BTC-style native XGBoost parameters (mirrors analyze_patterns.py from BTC project)
BTC_XGB_PARAMS = dict(
    objective="binary:logistic",
    eval_metric=["auc", "error"],
    max_depth=6,
    eta=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    tree_method="hist",
    seed=42,
)
NUM_BOOST_ROUND    = 1000
EARLY_STOP_ROUNDS  = 10

BUY_THRESH  =  0.15   # spread threshold to enter full position
SELL_THRESH = -0.15   # spread threshold to exit to cash


def train_ml_for_window(full_df: pd.DataFrame,
                        test_start: pd.Timestamp,
                        test_end: pd.Timestamp,
                        feature_override: list | None = None,
                        min_train_rows: int = 500,
                        purge_days: int = 30):
    """
    TRUE k-fold CV: train on ALL data EXCEPT the test window (with purge gaps).

    Two independent XGBoost boosters (native API, BTC-style):
      bull_booster: P(QLD next-day return > +1%)
      bear_booster: P(QLD next-day return < -1%)

    scale_pos_weight = neg_count / pos_count (computed per target, per fold).

    Signal: spread = bull_prob - bear_prob
      spread >= BUY_THRESH  -> weight = 1.0 (buy)
      spread <= SELL_THRESH -> weight = 0.0 (sell)
      otherwise             -> hold previous weight
    """
    from sklearn.metrics import roc_auc_score

    purge      = pd.Timedelta(days=purge_days)
    excl_start = test_start - purge
    excl_end   = test_end   + purge

    feat_list   = feature_override if feature_override is not None else ML_FEATURES
    feat_cols   = [c for c in feat_list if c in full_df.columns]
    target_cols = ["target_bull", "target_bear"]

    train_mask = (full_df.index < excl_start) | (full_df.index > excl_end)
    train_df   = full_df.loc[train_mask].dropna(subset=feat_cols + target_cols)

    if len(train_df) < min_train_rows:
        print(f"    ML: only {len(train_df)} training rows (need {min_train_rows}), skipping")
        return None

    # Val: rows adjacent to the test window (15% of training data)
    pre_df  = train_df.loc[train_df.index < excl_start]
    post_df = train_df.loc[train_df.index > excl_end]
    n_val      = max(50, int(len(train_df) * 0.15))
    n_pre_val  = min(n_val // 2, len(pre_df))
    n_post_val = min(n_val - n_pre_val, len(post_df))
    val_idx    = pd.Index(
        list(pre_df.index[-n_pre_val:]) + list(post_df.index[:n_post_val])
    )
    val_df = train_df.loc[train_df.index.isin(val_idx)]
    tr_df  = train_df.loc[~train_df.index.isin(val_idx)]

    X_tr  = tr_df[feat_cols].ffill().fillna(0.0)
    X_val = val_df[feat_cols].ffill().fillna(0.0)

    boosters = {}
    aucs     = {}
    for target in ("target_bull", "target_bear"):
        y_tr  = tr_df[target]
        y_val = val_df[target]

        # BTC key: scale_pos_weight corrects class imbalance
        n_neg = (y_tr == 0).sum()
        n_pos = (y_tr == 1).sum()
        spw   = float(n_neg) / max(1.0, float(n_pos))

        params = {**BTC_XGB_PARAMS, "scale_pos_weight": spw}

        dtrain = xgb.DMatrix(X_tr,  label=y_tr,  feature_names=feat_cols)
        dval   = xgb.DMatrix(X_val, label=y_val, feature_names=feat_cols)

        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            evals=[(dtrain, "train"), (dval, "eval")],
            early_stopping_rounds=EARLY_STOP_ROUNDS,
            verbose_eval=False,
        )
        boosters[target] = booster

        val_pred = booster.predict(dval)
        aucs[target] = roc_auc_score(y_val, val_pred) if y_val.sum() > 0 else float("nan")

    pre_rows  = (full_df.index < excl_start).sum()
    post_rows = (full_df.index > excl_end).sum()
    print(f"    ML trained: {len(tr_df)} rows "
          f"({pre_rows} pre + {post_rows} post), "
          f"{len(val_df)} val  "
          f"bull_AUC={aucs['target_bull']:.3f}  bear_AUC={aucs['target_bear']:.3f}")

    return boosters["target_bull"], boosters["target_bear"], feat_cols


def ml_weight_from_model(df: pd.DataFrame,
                         bull_clf, bear_clf,
                         feat_cols: list,
                         buy_thresh: float = BUY_THRESH,
                         sell_thresh: float = SELL_THRESH) -> pd.Series:
    """
    Stateful dual-threshold weight logic:
      spread = bull_prob - bear_prob
      >= buy_thresh  -> weight = 1.0 (enter)
      <= sell_thresh -> weight = 0.0 (exit)
      otherwise      -> hold previous weight (no trade)
    """
    X = df[[c for c in feat_cols if c in df.columns]].copy()
    for c in feat_cols:
        if c not in X.columns:
            X[c] = 0.0
    X = X[feat_cols].ffill().fillna(0.0)

    dmat      = xgb.DMatrix(X, feature_names=feat_cols)
    bull_prob = bull_clf.predict(dmat)
    bear_prob = bear_clf.predict(dmat)
    spread    = bull_prob - bear_prob

    # Stateful hold: maintain position when neither threshold is met
    weights = np.empty(len(df))
    w = 1.0 if spread[0] >= buy_thresh else 0.0   # initialise from first signal
    n_buy = n_sell = 0
    for i, s in enumerate(spread):
        if s >= buy_thresh:
            if w != 1.0:
                n_buy += 1
            w = 1.0
        elif s <= sell_thresh:
            if w != 0.0:
                n_sell += 1
            w = 0.0
        # else: hold (w unchanged)
        weights[i] = w

    pct_buy  = (spread >= buy_thresh).mean()
    pct_sell = (spread <= sell_thresh).mean()
    print(f"      spread: buy signal {pct_buy:.1%} of days, "
          f"sell signal {pct_sell:.1%} of days  "
          f"(transitions: {n_buy} buys, {n_sell} sells)")

    return pd.Series(weights, index=df.index)


STRATEGIES = {
    "SMA200":     w_sma200,
    "SMA200+VIX": w_sma200_vix,
    "Rules-5lvl": w_rules5,
    "VolTarget":  w_voltarget,
    "QLD B&H":    w_bah,
}


# ── Backtest ──────────────────────────────────────────────────────────────────

def backtest(df: pd.DataFrame, weights: pd.Series,
             cost_bps: float = COST_BPS,
             initial_cash: float = 1_000_000.0) -> pd.Series:
    """Close-to-close return model. Signal at T close → fill at T+1 open."""
    cost_rate = cost_bps / 10_000.0

    w = weights.reindex(df.index).ffill().fillna(0.0)
    v_close    = df["qld_close"].ffill()
    v_open     = df["qld_open"].ffill()
    cash_daily = df["bil_return_1d"].fillna(0.0)

    cc_ret    = (v_close / v_close.shift(1) - 1).fillna(0.0)
    intra_ret = (v_close / v_open - 1).fillna(0.0)

    w_applied = w.shift(1).fillna(0.0)
    w_prev    = w_applied.shift(1).fillna(0.0)
    w_delta   = (w_applied - w_prev)
    w_abs_chg = w_delta.abs()

    nav_vals = np.empty(len(df))
    nav_val  = float(initial_cash)

    for i in range(len(df)):
        unchanged = min(float(w_applied.iloc[i]), float(w_prev.iloc[i]))
        bought    = max(0.0, float(w_delta.iloc[i]))
        tc        = float(w_abs_chg.iloc[i]) * nav_val * cost_rate

        daily_ret = (unchanged * cc_ret.iloc[i]
                     + bought  * intra_ret.iloc[i]
                     + (1 - float(w_applied.iloc[i])) * cash_daily.iloc[i])

        nav_val   = nav_val * (1 + daily_ret) - tc
        nav_vals[i] = nav_val

    return pd.Series(nav_vals, index=df.index, name="nav")


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(nav: pd.Series, weights: pd.Series = None) -> dict:
    ret    = nav.pct_change().dropna()
    total  = nav.iloc[-1] / nav.iloc[0] - 1
    nyrs   = len(nav) / 252.0
    ann    = (1 + total) ** (1 / nyrs) - 1 if nyrs > 0 else 0.0
    vol    = ret.std() * np.sqrt(252)
    peak   = nav.expanding().max()
    dd_s   = (nav - peak) / peak
    mdd    = float(dd_s.min())
    rf     = 0.03
    sharpe = (ann - rf) / vol if vol > 0 else 0.0
    calmar = ann / abs(mdd) if mdd != 0 else 0.0
    monthly = nav.resample("ME").last().pct_change().dropna()
    win_m   = float((monthly > 0).mean())
    avg_w   = float(weights.mean()) if weights is not None else 1.0
    return dict(ann=ann, total=total, mdd=mdd, vol=vol,
                sharpe=sharpe, calmar=calmar, win_m=win_m, avg_w=avg_w)


# ── Window runner ─────────────────────────────────────────────────────────────

def run_window(full_df: pd.DataFrame,
               label: str, start: str, end: str | None,
               bear_type: str,
               feature_override: list | None = None) -> tuple[dict, dict | None]:
    """Returns (results_dict, feature_importance_dict | None)."""
    start_dt = pd.Timestamp(start)
    end_dt   = pd.Timestamp(end) if end else full_df.index.max()

    window = full_df.loc[start_dt:end_dt]
    if len(window) < 100:
        print(f"  [{label}] Insufficient data ({len(window)} rows) - skipping")
        return {}, None

    results = {}
    for strat_name, weight_fn in STRATEGIES.items():
        w   = weight_fn(window)
        nav = backtest(window, w)
        results[strat_name] = compute_metrics(nav, w)

    # ML strategy: leave-test-out CV (train on everything except this window)
    ml_result = train_ml_for_window(full_df, start_dt, end_dt,
                                     feature_override=feature_override)
    feat_imp = None
    if ml_result is not None:
        bull_clf, bear_clf, feat_cols = ml_result
        w_ml = ml_weight_from_model(window, bull_clf, bear_clf, feat_cols)
        nav_ml = backtest(window, w_ml)
        results["ML-MTF"] = compute_metrics(nav_ml, w_ml)

        # Average bull + bear importances (native xgb gain-based)
        bull_imp_raw = bull_clf.get_score(importance_type="gain")
        bear_imp_raw = bear_clf.get_score(importance_type="gain")
        bull_imp = pd.Series({f: bull_imp_raw.get(f, 0.0) for f in feat_cols})
        bear_imp = pd.Series({f: bear_imp_raw.get(f, 0.0) for f in feat_cols})
        feat_imp = (0.5 * bull_imp + 0.5 * bear_imp)
        total    = feat_imp.sum()
        feat_imp = feat_imp / total if total > 0 else feat_imp
    else:
        results["ML-MTF"] = dict(ann=0, total=0, mdd=0, vol=0,
                                  sharpe=0, calmar=0, win_m=0, avg_w=0)

    return results, feat_imp


# ── Tearsheet ─────────────────────────────────────────────────────────────────

def print_window_table(label: str, bear_type: str, start: str, end: str | None,
                       results: dict, full_df: pd.DataFrame):
    if not results:
        return

    end_str = end if end else full_df.index.max().strftime("%Y-%m-%d")
    bah  = results.get("QLD B&H", {})

    # Column order: SMA200 first, then others, ML, B&H last
    strats = ["SMA200", "SMA200+VIX", "Rules-5lvl", "VolTarget", "ML-MTF", "QLD B&H"]
    strats = [s for s in strats if s in results]

    col_w = 13
    hdr   = f"  {'Metric':<22}" + "".join(f" {s[:col_w]:>{col_w}}" for s in strats)
    w     = len(hdr)

    print()
    print("=" * w)
    print(f"  {label} ({bear_type}) - {start} to {end_str}")
    print("=" * w)
    print(hdr)
    print("-" * w)

    rows = [
        ("Ann. Return",  "ann",    ".1%"),
        ("Total Return", "total",  ".0%"),
        ("Max Drawdown", "mdd",    ".1%"),
        ("Sharpe Ratio", "sharpe", ".3f"),
        ("Calmar Ratio", "calmar", ".3f"),
        ("Win Rate(Mo)", "win_m",  ".0%"),
        ("Avg Weight",   "avg_w",  ".0%"),
    ]

    for metric_name, key, fmt in rows:
        row = f"  {metric_name:<22}"
        for s in strats:
            val = results[s].get(key, float("nan"))
            row += f" {format(val, fmt):>{col_w}}"
        print(row)

    # Beat B&H row
    print("-" * w)
    beat_row = f"  {'Beats B&H(R+MDD)':<22}"
    for s in strats:
        if s == "QLD B&H":
            beat_row += f" {'---':>{col_w}}"
            continue
        m   = results[s]
        win = (m.get("ann", 0) >= bah.get("ann", 0)) and \
              (m.get("mdd", -1) >= bah.get("mdd", -1))
        beat_row += f" {'YES' if win else 'no':>{col_w}}"
    print(beat_row)
    print("=" * w)


# ── Summary table across all windows ─────────────────────────────────────────

def print_summary(all_results: dict):
    """Cross-fold summary: Ann / MDD / Calmar for SMA200, ML-MTF, QLD B&H."""
    W = 118
    print()
    print("=" * W)
    print("  CROSS-FOLD SUMMARY")
    print("=" * W)
    print(f"  {'':22}  {'--- SMA200 ---':^27}  {'--- ML-MTF ---':^27}  {'--- QLD B&H ---':^27}")
    print(f"  {'Window':<22}  {'Ann':>7} {'MDD':>8} {'Cal':>8}  "
          f"{'Ann':>7} {'MDD':>8} {'Cal':>8}  "
          f"{'Ann':>7} {'MDD':>8} {'Cal':>8}  {'Beats B&H?':^11}")
    print("-" * W)

    calmar_sma_all = []
    calmar_ml_all  = []
    calmar_bah_all = []
    ann_sma_all, ann_ml_all, ann_bah_all = [], [], []
    mdd_sma_all, mdd_ml_all, mdd_bah_all = [], [], []
    sma_beat = ml_beat = 0
    total_count = 0

    for label, res in all_results.items():
        if not res:
            continue
        sma = res.get("SMA200", {})
        ml  = res.get("ML-MTF", {})
        bah = res.get("QLD B&H", {})
        if not sma or not bah:
            continue

        sma_wins = (sma.get("ann", 0) >= bah.get("ann", 0)) and \
                   (sma.get("mdd", -1) >= bah.get("mdd", -1))
        ml_wins  = (ml.get("ann", 0) >= bah.get("ann", 0)) and \
                   (ml.get("mdd", -1) >= bah.get("mdd", -1))
        if sma_wins: sma_beat += 1
        if ml_wins:  ml_beat  += 1
        total_count += 1

        calmar_sma_all.append(sma.get("calmar", 0))
        calmar_ml_all.append(ml.get("calmar", 0))
        calmar_bah_all.append(bah.get("calmar", 0))
        ann_sma_all.append(sma.get("ann", 0))
        ann_ml_all.append(ml.get("ann", 0))
        ann_bah_all.append(bah.get("ann", 0))
        mdd_sma_all.append(sma.get("mdd", 0))
        mdd_ml_all.append(ml.get("mdd", 0))
        mdd_bah_all.append(bah.get("mdd", 0))

        beat_str = f"SMA={'YES' if sma_wins else 'no'} ML={'YES' if ml_wins else 'no'}"
        print(f"  {label:<22}"
              f"  {sma.get('ann',0):>7.1%} {sma.get('mdd',0):>8.1%} {sma.get('calmar',0):>8.3f}"
              f"  {ml.get('ann',0):>7.1%} {ml.get('mdd',0):>8.1%} {ml.get('calmar',0):>8.3f}"
              f"  {bah.get('ann',0):>7.1%} {bah.get('mdd',0):>8.1%} {bah.get('calmar',0):>8.3f}"
              f"  {beat_str}")

    print("-" * W)
    if calmar_sma_all:
        print(f"  {'AVERAGE':<22}"
              f"  {np.mean(ann_sma_all):>7.1%} {np.mean(mdd_sma_all):>8.1%} {np.mean(calmar_sma_all):>8.3f}"
              f"  {np.mean(ann_ml_all):>7.1%} {np.mean(mdd_ml_all):>8.1%} {np.mean(calmar_ml_all):>8.3f}"
              f"  {np.mean(ann_bah_all):>7.1%} {np.mean(mdd_bah_all):>8.1%} {np.mean(calmar_bah_all):>8.3f}")
        print(f"\n  SMA200 beats B&H: {sma_beat}/{total_count}    ML-MTF beats B&H: {ml_beat}/{total_count}")
    print("=" * W)


def print_feature_importance(all_importances: list[pd.Series]):
    """Print mean feature importance across all folds that had ML models."""
    if not all_importances:
        return
    combined = pd.concat(all_importances, axis=1).fillna(0)
    mean_imp = combined.mean(axis=1).sort_values(ascending=False)

    print()
    print("=" * 60)
    print("  ML-MTF FEATURE IMPORTANCE (avg across folds, top 20)")
    print("=" * 60)
    print(f"  {'Rank':<5} {'Feature':<30} {'Importance':>10}")
    print("-" * 60)

    # Group by feature family
    per_period = {"return", "volatility", "volume_ratio", "vwap", "sma",
                  "hlrange", "rsi", "bb_upper", "bb_lower"}
    global_inds = {"macd", "macd_signal", "macd_hist", "dmi_plus", "dmi_minus", "adx"}
    vix_feats   = {"vix_close", "vix_sma20", "vix_percentile", "vix_change_5d"}

    def feat_family(f):
        prefix = f.split("_")[0]
        if f in global_inds:
            return "Global"
        if f in vix_feats or f.startswith("vix"):
            return "VIX"
        if f in {"qqq_drawdown", "days_in_drawdown"}:
            return "Drawdown"
        if prefix in per_period:
            return "Per-period"
        return "Other"

    for rank, (feat, imp) in enumerate(mean_imp.head(20).items(), 1):
        fam = feat_family(feat)
        print(f"  {rank:<5} {feat:<30} {imp:>9.3%}  [{fam}]")
    print("=" * 60)

    # Family breakdown
    fam_totals = {}
    for feat, imp in mean_imp.items():
        fam = feat_family(feat)
        fam_totals[fam] = fam_totals.get(fam, 0) + imp
    print("\n  Importance by feature family:")
    for fam, total in sorted(fam_totals.items(), key=lambda x: -x[1]):
        print(f"    {fam:<12} {total:.1%}")
    print()

    return mean_imp


# ── Threshold sweep ───────────────────────────────────────────────────────────

def weights_from_spread(spread: pd.Series, thresh: float) -> pd.Series:
    """Stateful hold logic applied to a pre-computed spread series (symmetric)."""
    return weights_from_spread_asym(spread, buy_thresh=thresh, sell_thresh=-thresh)


def weights_from_spread_asym(spread: pd.Series,
                              buy_thresh: float,
                              sell_thresh: float) -> pd.Series:
    """
    Asymmetric stateful hold logic:
      spread >= buy_thresh   -> weight = 1.0 (enter)
      spread <= sell_thresh  -> weight = 0.0 (exit)
      otherwise              -> hold previous weight

    sell_thresh should be negative (e.g. -0.25 means exit only on strong bear signal).
    buy_thresh  should be positive  (e.g.  0.10 means enter on mild bull signal).
    """
    s = spread.values
    w = 1.0 if s[0] >= buy_thresh else 0.0
    weights = np.empty(len(s))
    for i, v in enumerate(s):
        if v >= buy_thresh:
            w = 1.0
        elif v <= sell_thresh:
            w = 0.0
        weights[i] = w
    return pd.Series(weights, index=spread.index)


def tune_thresholds_asym(full_df: pd.DataFrame,
                         window_spreads: dict,
                         buy_thresholds: np.ndarray,
                         sell_thresholds: np.ndarray):
    """
    2D sweep: buy_thresh × sell_thresh combinations.
    sell_thresh is the MAGNITUDE (applied as -sell_thresh internally).
    Skips combos where sell_thresh < buy_thresh (incoherent: sell before buy).

    Prints top-15 combos sorted by avg Calmar (excluding artifact high-thresh runs).
    """
    # Pre-compute SMA200 stats
    sma_calmars = []
    for label, start, end, _ in WINDOWS:
        if label not in window_spreads:
            continue
        window, _ = window_spreads[label]
        nav = backtest(window, w_sma200(window))
        sma_calmars.append(compute_metrics(nav)["calmar"])
    sma_avg = np.mean(sma_calmars) if sma_calmars else 0.0

    rows = []
    for bt in buy_thresholds:
        for st in sell_thresholds:
            if st < bt:          # sell threshold must be >= buy (in magnitude)
                continue
            sell_neg = -st       # convert to negative for the exit condition

            calmars, mdds, anns, avg_ws, beats, trades = [], [], [], [], 0, []
            for label, start, end, _ in WINDOWS:
                if label not in window_spreads:
                    continue
                window, spread = window_spreads[label]
                w_ml  = weights_from_spread_asym(spread, bt, sell_neg)
                nav   = backtest(window, w_ml)
                m     = compute_metrics(nav, w_ml)
                bah_nav = backtest(window, pd.Series(1.0, index=window.index))
                bah_m   = compute_metrics(bah_nav)

                calmars.append(m["calmar"])
                mdds.append(m["mdd"])
                anns.append(m["ann"])
                avg_ws.append(m["avg_w"])
                if m["ann"] >= bah_m["ann"] and m["mdd"] >= bah_m["mdd"]:
                    beats += 1

                # Count transitions
                s = spread.values
                n_tr = sum(
                    1 for i in range(1, len(s))
                    if (s[i] >= bt and s[i-1] < bt) or (s[i] <= sell_neg and s[i-1] > sell_neg)
                )
                trades.append(n_tr)

            avg_c = np.mean(calmars)
            rows.append(dict(
                buy=bt, sell=st,
                avg_calmar=avg_c,
                avg_mdd=np.mean(mdds),
                avg_ann=np.mean(anns),
                avg_w=np.mean(avg_ws),
                beats=beats,
                n_win=len(calmars),
                avg_trades=np.mean(trades),
            ))

    # Filter out Calmar-explosion rows (avg_trades < 5 across windows)
    valid = [r for r in rows if r["avg_trades"] >= 5]
    valid.sort(key=lambda r: -r["avg_calmar"])

    print()
    print("=" * 105)
    print("  ASYMMETRIC THRESHOLD SWEEP  (buy=enter, sell=exit magnitude; sell >= buy)")
    print("  Filtered: avg_trades >= 5 per window")
    print("=" * 105)
    hdr = (f"  {'Buy':>6} {'Sell':>6}  {'AvgCalmar':>10}  {'AvgMDD':>8}  "
           f"{'AvgAnn':>8}  {'AvgW':>6}  {'BeatBaH':>8}  {'AvgTrades':>10}")
    print(hdr)
    print("-" * 105)

    best = valid[0] if valid else None
    for r in valid[:20]:
        marker = " <-- BEST" if r is best else ""
        print(f"  {r['buy']:>6.2f} {r['sell']:>6.2f}  {r['avg_calmar']:>10.3f}  "
              f"{r['avg_mdd']:>8.1%}  {r['avg_ann']:>8.1%}  {r['avg_w']:>6.1%}  "
              f"{r['beats']:>4}/{r['n_win']}   {r['avg_trades']:>8.1f}{marker}")

    print("-" * 105)
    print(f"  {'SMA200':>6}        {sma_avg:>10.3f}  (baseline)")
    print("=" * 105)

    return best


def tune_thresholds(full_df: pd.DataFrame,
                    window_spreads: dict,   # label -> (window_df, spread Series)
                    thresholds: np.ndarray):
    """
    For each threshold sweep, re-run backtest on cached spreads.
    Print a table: threshold | avg_calmar | avg_mdd | beat_bah_count | avg_ann
    Also show SMA200 baseline row for comparison.
    """
    # Pre-compute SMA200 stats per window for reference
    sma_calmars = []
    for label, start, end, _ in WINDOWS:
        if label not in window_spreads:
            continue
        window, _ = window_spreads[label]
        w_sma = w_sma200(window)
        nav   = backtest(window, w_sma)
        m     = compute_metrics(nav, w_sma)
        sma_calmars.append(m["calmar"])
    sma_avg_calmar = np.mean(sma_calmars) if sma_calmars else 0.0

    print()
    print("=" * 90)
    print("  THRESHOLD SWEEP  (symmetric: buy >= t, sell <= -t, hold otherwise)")
    print("=" * 90)
    hdr = (f"  {'Thresh':>7}  {'AvgCalmar':>10}  {'AvgMDD':>8}  "
           f"{'AvgAnn':>8}  {'AvgWeight':>10}  {'BeatBaH':>8}  {'AvgTrades':>10}")
    print(hdr)
    print("-" * 90)

    best_row = None
    best_calmar = -np.inf

    for t in thresholds:
        calmars, mdds, anns, avg_ws, beats = [], [], [], [], 0

        for label, start, end, _ in WINDOWS:
            if label not in window_spreads:
                continue
            window, spread = window_spreads[label]

            w_ml  = weights_from_spread(spread, t)
            nav   = backtest(window, w_ml)
            m     = compute_metrics(nav, w_ml)

            # B&H baseline for this window
            bah_nav = backtest(window, pd.Series(1.0, index=window.index))
            bah_m   = compute_metrics(bah_nav)

            calmars.append(m["calmar"])
            mdds.append(m["mdd"])
            anns.append(m["ann"])
            avg_ws.append(m["avg_w"])
            if m["ann"] >= bah_m["ann"] and m["mdd"] >= bah_m["mdd"]:
                beats += 1

        avg_c = np.mean(calmars)
        avg_m = np.mean(mdds)
        avg_a = np.mean(anns)
        avg_w = np.mean(avg_ws)
        n_win = len(calmars)

        # Average trades per window
        avg_trades = np.mean([
            int(sum(
                1 for i in range(1, len(window_spreads[lbl][1]))
                if (window_spreads[lbl][1].iloc[i] >= t and window_spreads[lbl][1].iloc[i-1] < t) or
                   (window_spreads[lbl][1].iloc[i] <= -t and window_spreads[lbl][1].iloc[i-1] > -t)
            ))
            for lbl, *_ in WINDOWS if lbl in window_spreads
        ])

        marker = " <--" if avg_c > best_calmar else ""
        if avg_c > best_calmar:
            best_calmar = avg_c
            best_row = t

        print(f"  {t:>7.2f}  {avg_c:>10.3f}  {avg_m:>8.1%}  "
              f"{avg_a:>8.1%}  {avg_w:>10.1%}  {beats:>5}/{n_win}   "
              f"{avg_trades:>8.1f}{marker}")

    print("-" * 90)
    print(f"  {'SMA200':>7}  {sma_avg_calmar:>10.3f}  (baseline)")
    print("=" * 90)
    print(f"\n  Best threshold: {best_row:.2f}  (avg Calmar {best_calmar:.3f})")
    return best_row


# ── Main ──────────────────────────────────────────────────────────────────────

def _run_config(full_df: pd.DataFrame,
                feat_list: list,
                config_name: str) -> dict:
    """Run all 6 windows with a given feature list. Returns all_results dict."""
    all_results     = {}
    all_importances = []
    for label, start, end, bear_type in WINDOWS:
        res, feat_imp = run_window(full_df, label, start, end, bear_type,
                                   feature_override=feat_list)
        all_results[label] = res
        if feat_imp is not None:
            feat_imp.name = label
            all_importances.append(feat_imp)
    return all_results, all_importances


def print_config_comparison(config_results: dict):
    """One-row-per-window comparison table across all configs."""
    configs = list(config_results.keys())
    W = 30 + 27 * len(configs)
    print()
    print("=" * W)
    print("  FEATURE GROUP COMPARISON  (avg Calmar at threshold 0.15)")
    print("=" * W)

    # Header
    hdr = f"  {'Window':<22}"
    for cfg in configs:
        hdr += f"  {cfg[:23]:^25}"
    print(hdr)
    sub = f"  {'':22}"
    for _ in configs:
        sub += f"  {'Ann':>7} {'MDD':>8} {'Cal':>7}"
    print(sub)
    print("-" * W)

    calmar_avgs = {cfg: [] for cfg in configs}
    for label, _, _, _ in WINDOWS:
        row = f"  {label:<22}"
        for cfg in configs:
            res = config_results[cfg].get(label, {})
            ml  = res.get("ML-MTF", {})
            ann = ml.get("ann", 0)
            mdd = ml.get("mdd", 0)
            cal = ml.get("calmar", 0)
            calmar_avgs[cfg].append(cal)
            row += f"  {ann:>7.1%} {mdd:>8.1%} {cal:>7.3f}"
        print(row)

    print("-" * W)
    row = f"  {'AVERAGE':<22}"
    for cfg in configs:
        vals = calmar_avgs[cfg]
        row += f"  {'':>7}  {'':>8} {np.mean(vals):>7.3f}"
    print(row)
    print("=" * W)

    # Also show SMA200 baseline
    first_cfg = configs[0]
    sma_cals = []
    for label, _, _, _ in WINDOWS:
        res = config_results[first_cfg].get(label, {})
        sma_cals.append(res.get("SMA200", {}).get("calmar", 0))
    print(f"\n  SMA200 baseline avg Calmar: {np.mean(sma_cals):.3f}")


def main():
    parser = argparse.ArgumentParser(description="Cross-fold validation across market regimes")
    parser.add_argument("--download", action="store_true",
                        help="Force re-download QQQ history")
    args = parser.parse_args()

    full_df = build_full_dataset(force_download=args.download)

    # ── Feature configs to compare ────────────────────────────────────────────
    CONFIGS = {
        "Base + Cross-Asset":              ML_FEATURES_BASE + CROSS_ASSET_ADDITIONS,
        "Base + Regime":                   ML_FEATURES_BASE + REGIME_ADDITIONS,
        "Base + Cross + Regime":           ML_FEATURES_BASE + CROSS_ASSET_ADDITIONS + REGIME_ADDITIONS,
        "Base + Cross + Regime + Macro":   ML_FEATURES_BASE + CROSS_ASSET_ADDITIONS + REGIME_ADDITIONS + MACRO_ADDITIONS,
    }

    config_results     = {}
    config_importances = {}

    for config_name, feat_list in CONFIGS.items():
        n_feat = len([f for f in feat_list if f in full_df.columns])
        print(f"\n{'='*70}")
        print(f"  CONFIG: {config_name}  ({n_feat} features)")
        print(f"{'='*70}")
        all_results, all_importances = _run_config(full_df, feat_list, config_name)
        config_results[config_name] = all_results
        config_importances[config_name] = all_importances

        # Print per-window tables for this config
        for label, start, end, bear_type in WINDOWS:
            print_window_table(label, bear_type, start, end,
                               all_results.get(label, {}), full_df)
        print_summary(all_results)
        print_feature_importance(all_importances)

    # ── Cross-config comparison ────────────────────────────────────────────────
    print_config_comparison(config_results)

    # ── Threshold sweep on best config (determined from comparison) ───────────
    # Default to full combo; user can adjust after seeing comparison table
    best_feat_list = ML_FEATURES_BASE + CROSS_ASSET_ADDITIONS + REGIME_ADDITIONS + MACRO_ADDITIONS
    print(f"\n\n{'='*70}")
    print(f"  THRESHOLD SWEEP: Base + Cross + Regime + Macro ({len([f for f in best_feat_list if f in full_df.columns])} features)")
    print(f"{'='*70}")
    print("Building spread cache (retraining per window)...")

    window_spreads = {}
    for label, start, end, _ in WINDOWS:
        start_dt = pd.Timestamp(start)
        end_dt   = pd.Timestamp(end) if end else full_df.index.max()
        window   = full_df.loc[start_dt:end_dt]
        if len(window) < 100:
            continue
        ml_result = train_ml_for_window(full_df, start_dt, end_dt,
                                         feature_override=best_feat_list)
        if ml_result is None:
            continue
        bull_clf, bear_clf, feat_cols = ml_result
        X = window[[c for c in feat_cols if c in window.columns]].copy()
        for c in feat_cols:
            if c not in X.columns:
                X[c] = 0.0
        X = X[feat_cols].ffill().fillna(0.0)
        dmat      = xgb.DMatrix(X, feature_names=feat_cols)
        bull_prob = pd.Series(bull_clf.predict(dmat), index=window.index)
        bear_prob = pd.Series(bear_clf.predict(dmat), index=window.index)
        window_spreads[label] = (window, bull_prob - bear_prob)

    thresholds  = np.round(np.arange(0.05, 0.65, 0.05), 2)
    best_thresh = tune_thresholds(full_df, window_spreads, thresholds)

    # ── Final window-by-window results at best threshold ──────────────────────
    # ── Asymmetric threshold sweep ────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print(f"  ASYMMETRIC THRESHOLD SWEEP: Base + Cross + Regime")
    print(f"{'='*70}")
    buy_thresholds  = np.round(np.arange(0.05, 0.40, 0.05), 2)
    sell_thresholds = np.round(np.arange(0.05, 0.55, 0.05), 2)
    best_asym = tune_thresholds_asym(full_df, window_spreads,
                                     buy_thresholds, sell_thresholds)

    if best_asym:
        bt = best_asym["buy"]
        st = -best_asym["sell"]
        print(f"\nFINAL: Base + Cross + Regime  buy={bt:.2f}  sell={-st:.2f}")
        print("=" * 118)
        print(f"  {'Window':<22}  {'ML Ann':>7} {'ML MDD':>8} {'ML Cal':>8}   "
              f"{'SMA Ann':>7} {'SMA MDD':>8} {'SMA Cal':>8}   {'Beat SMA?':>9}")
        print("-" * 118)
        for label, start, end, _ in WINDOWS:
            if label not in window_spreads:
                continue
            window, spread = window_spreads[label]
            w_ml  = weights_from_spread_asym(spread, bt, st)
            nav   = backtest(window, w_ml)
            m     = compute_metrics(nav, w_ml)
            w_sma = w_sma200(window)
            nav_s = backtest(window, w_sma)
            ms    = compute_metrics(nav_s, w_sma)
            beat  = "YES" if m["ann"] >= ms["ann"] and m["mdd"] >= ms["mdd"] else "no"
            print(f"  {label:<22}  {m['ann']:>7.1%} {m['mdd']:>8.1%} {m['calmar']:>8.3f}   "
                  f"{ms['ann']:>7.1%} {ms['mdd']:>8.1%} {ms['calmar']:>8.3f}   {beat:>9}")
        print("=" * 118)


if __name__ == "__main__":
    main()
