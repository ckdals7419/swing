"""
Phase 3 Feature Builder
Loads features_phase2.parquet, adds market regime features (SPY + VIX),
saves data/features_phase3.parquet.

Usage:
    python build_features_phase3.py
"""

import warnings
import pandas as pd
warnings.filterwarnings("ignore")

from features.market_regime import add_market_regime

PHASE2  = "data/features_phase2.parquet"
SPY     = "data/spy_daily.csv"
VIX     = "data/vix_daily.csv"
OUTPUT  = "data/features_phase3.parquet"


def main():
    print(f"Loading {PHASE2}...")
    df = pd.read_parquet(PHASE2)
    df["date"] = pd.to_datetime(df["date"])
    print(f"  Shape: {df.shape}")

    print("Adding market regime features (SPY + VIX + beta)...")
    df = add_market_regime(df, SPY, VIX)

    phase2_cols = set(pd.read_parquet(PHASE2).columns)
    new_cols = [c for c in df.columns if c not in phase2_cols]
    print(f"  Added {len(new_cols)} new columns: {new_cols}")

    print(f"\nSaving to {OUTPUT}...")
    df.to_parquet(OUTPUT, index=False)
    print(f"Done. Shape: {df.shape}")
    print(f"Total columns: {df.shape[1]}")


if __name__ == "__main__":
    main()
