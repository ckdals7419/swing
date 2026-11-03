"""
Phase 1 Feature Builder
Reads sp500_clean_ohlcv.parquet, applies all OHLCV-based feature modules,
and saves data/features_phase1.parquet.

Usage:
    python build_features.py
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from features.price_action import add_price_action
from features.trend import add_trend
from features.momentum import add_momentum
from features.volatility import add_volatility
from features.volume import add_volume
from features.multi_timeframe import add_multi_timeframe
from features.calendar_features import add_calendar_features
from features.statistical import add_statistical
from features.targets import add_targets

INPUT = "data/sp500_clean_ohlcv.parquet"
OUTPUT = "data/features_phase1.parquet"


def build_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all Phase 1 feature modules to a single ticker's DataFrame."""
    df = df.sort_values("date").reset_index(drop=True)
    df = add_price_action(df)
    df = add_trend(df)
    df = add_momentum(df)
    df = add_volatility(df)
    df = add_volume(df)
    df = add_multi_timeframe(df)
    df = add_statistical(df)
    df = add_targets(df)        # forward-looking — safe to add here
    return df


def main():
    print(f"Loading {INPUT}...")
    raw = pd.read_parquet(INPUT)
    raw["date"] = pd.to_datetime(raw["date"])

    # Use Adj Close for all price calculations to account for splits/dividends
    raw["Close"] = raw["Adj Close"]
    # Adjust OHLC proportionally
    ratio = raw["Adj Close"] / raw["Close"].where(raw["Close"] != 0, np.nan)
    raw["Open"]  = raw["Open"]  * ratio
    raw["High"]  = raw["High"]  * ratio
    raw["Low"]   = raw["Low"]   * ratio

    tickers = raw["ticker"].unique()
    print(f"Processing {len(tickers)} tickers...")

    results = []
    t0 = time.time()

    for i, ticker in enumerate(tickers, 1):
        sub = raw[raw["ticker"] == ticker].copy()
        try:
            sub = build_ticker(sub)
            results.append(sub)
        except Exception as e:
            print(f"  [WARN] {ticker}: {e}", file=sys.stderr)

        if i % 50 == 0 or i == len(tickers):
            elapsed = time.time() - t0
            print(f"  {i}/{len(tickers)} tickers done ({elapsed:.0f}s)")

    print("Concatenating results...")
    out = pd.concat(results, ignore_index=True)
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Drop raw price columns (keep only Adj Close as "Close")
    drop_cols = ["Adj Close", "Dividends", "Stock Splits"]
    out = out.drop(columns=[c for c in drop_cols if c in out.columns])

    print(f"Final shape: {out.shape}")
    print(f"Features: {out.shape[1] - 3} (excl. ticker, date, OHLCV)")

    # Null summary
    null_pct = out.isnull().mean()
    high_null = null_pct[null_pct > 0.5]
    if len(high_null):
        print(f"\n[WARN] {len(high_null)} columns with >50% nulls:")
        print(high_null.to_string())

    print(f"\nSaving to {OUTPUT}...")
    out.to_parquet(OUTPUT, index=False)
    print("Done.")

    # Feature summary
    feature_cols = [c for c in out.columns if c not in
                    ["ticker", "date", "Open", "High", "Low", "Close", "Volume"]]
    target_cols = [c for c in feature_cols if c.startswith("fwd_")]
    feature_cols = [c for c in feature_cols if not c.startswith("fwd_")]
    print(f"\nFeature columns  : {len(feature_cols)}")
    print(f"Target columns   : {len(target_cols)}")
    print(f"Total rows       : {len(out):,}")
    print(f"Date range       : {out['date'].min().date()} → {out['date'].max().date()}")


if __name__ == "__main__":
    main()
