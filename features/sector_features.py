"""
Category 9: Sector & Cross-Sectional Features

Requires:
  - data/features_phase1.parquet  (output of build_features.py)
  - data/ticker_sectors.csv       (output of download_sectors.py)

Adds per-row:
  - sector_code, industry (integer-encoded)
  - return_vs_market        stock return minus market average
  - return_vs_sector        stock return minus sector average
  - sector_momentum_20d     average 20d return of sector peers
  - sector_rotation_rank    sector rank by 20d return (1=best)
  - cross_sectional_rank    percentile rank vs all stocks (by 1d return)
  - relative_strength_vs_mkt  rolling ratio of stock to market return
  - sector_vol_rank         stock vol rank within sector
  - sector_dispersion_20d   cross-sectional std of sector returns
"""

import numpy as np
import pandas as pd


def add_sector_features(df: pd.DataFrame, sectors: pd.DataFrame) -> pd.DataFrame:
    """
    df      : full feature dataframe with columns [ticker, date, Close, return_1d, ...]
    sectors : dataframe with columns [ticker, sector, industry, sector_code]
    """
    # Merge sector labels
    df = df.merge(
        sectors[["ticker", "sector", "sector_code"]].drop_duplicates("ticker"),
        on="ticker", how="left"
    )

    # Ensure return columns exist
    if "return_1d" not in df.columns:
        df["return_1d"] = df.groupby("ticker")["Close"].pct_change(1)
    if "return_20d" not in df.columns:
        df["return_20d"] = df.groupby("ticker")["Close"].pct_change(20)
    if "realized_vol_20d" not in df.columns:
        log_ret = df.groupby("ticker")["Close"].transform(lambda x: np.log(x / x.shift(1)))
        df["realized_vol_20d"] = log_ret.groupby(df["ticker"]).transform(
            lambda x: x.rolling(20).std() * np.sqrt(252)
        )

    # ── Per-date cross-sectional computations ────────────────────────────────
    dates = df[["date", "ticker", "return_1d", "return_20d",
                "realized_vol_20d", "sector_code"]].copy()

    # Market average return (all stocks)
    mkt_avg_1d = dates.groupby("date")["return_1d"].transform("mean")
    mkt_avg_20d = dates.groupby("date")["return_20d"].transform("mean")
    df["return_vs_market_1d"] = df["return_1d"] - mkt_avg_1d.values
    df["return_vs_market_20d"] = df["return_20d"] - mkt_avg_20d.values

    # Sector average return
    sec_avg_1d = dates.groupby(["date", "sector_code"])["return_1d"].transform("mean")
    sec_avg_20d = dates.groupby(["date", "sector_code"])["return_20d"].transform("mean")
    df["return_vs_sector_1d"] = df["return_1d"] - sec_avg_1d.values
    df["return_vs_sector_20d"] = df["return_20d"] - sec_avg_20d.values

    # Sector momentum: average 20d return of sector on this date
    df["sector_momentum_20d"] = sec_avg_20d.values

    # Sector rotation rank (1 = sector with highest 20d return, per date)
    sec_rank = (
        dates.groupby(["date", "sector_code"])["return_20d"]
        .mean()
        .reset_index()
        .assign(sector_rotation_rank=lambda x: x.groupby("date")["return_20d"]
                .rank(ascending=False, method="min"))
    )[["date", "sector_code", "sector_rotation_rank"]]
    df = df.merge(sec_rank, on=["date", "sector_code"], how="left")

    # Cross-sectional percentile rank of 1d return (per date)
    df["cross_sectional_rank"] = dates.groupby("date")["return_1d"].rank(pct=True).values

    # Cross-sectional std dev of 1d returns (market dispersion)
    df["market_dispersion_1d"] = dates.groupby("date")["return_1d"].transform("std").values

    # Sector dispersion: std of 20d returns within sector
    df["sector_dispersion_20d"] = (
        dates.groupby(["date", "sector_code"])["return_20d"].transform("std").values
    )

    # Sector vol rank: percentile of stock vol within sector (per date)
    df["sector_vol_rank"] = (
        dates.groupby(["date", "sector_code"])["realized_vol_20d"]
        .rank(pct=True)
        .values
    )

    # Relative strength vs market: rolling 20d cumulative return ratio
    mkt_cum = df.groupby("date")["return_1d"].transform("mean")
    mkt_cum_20 = mkt_cum.groupby(df["ticker"]).transform(
        lambda x: (1 + x).rolling(20).apply(np.prod, raw=True) - 1
    )
    stock_cum_20 = df.groupby("ticker")["return_1d"].transform(
        lambda x: (1 + x).rolling(20).apply(np.prod, raw=True) - 1
    )
    df["relative_strength_vs_mkt"] = stock_cum_20 - mkt_cum_20

    return df
