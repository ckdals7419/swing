"""
Phase 2 Feature Builder
Loads features_phase1.parquet + ticker_sectors.csv,
adds sector & cross-sectional features,
saves data/features_phase2.parquet.

Usage:
    python build_features_phase2.py
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from features.sector_features import add_sector_features

PHASE1 = "data/features_phase1.parquet"
SECTORS = "data/ticker_sectors.csv"
OUTPUT = "data/features_phase2.parquet"


def main():
    print(f"Loading {PHASE1}...")
    df = pd.read_parquet(PHASE1)
    df["date"] = pd.to_datetime(df["date"])
    print(f"  Shape: {df.shape}")

    print(f"Loading {SECTORS}...")
    sectors = pd.read_csv(SECTORS)
    print(f"  {len(sectors)} tickers, {sectors['sector'].notna().sum()} with sector")

    print("Adding sector & cross-sectional features...")
    df = add_sector_features(df, sectors)

    new_cols = [c for c in df.columns if c not in pd.read_parquet(PHASE1).columns]
    print(f"  Added {len(new_cols)} new columns: {new_cols}")

    print(f"\nSaving to {OUTPUT}...")
    df.to_parquet(OUTPUT, index=False)

    print(f"Done. Shape: {df.shape}")
    print(f"Total feature columns: {df.shape[1] - 3}")  # excl ticker, date, Close


if __name__ == "__main__":
    main()
