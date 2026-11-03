"""
Phase 4 Feature Builder
Loads features_phase3.parquet, adds macro features,
saves data/features_phase4.parquet.

Usage:
    python build_features_phase4.py
"""

import warnings
import pandas as pd
warnings.filterwarnings("ignore")

from features.macro_features import add_macro_features

PHASE3 = "data/features_phase3.parquet"
MACRO  = "data/macro_daily.csv"
OUTPUT = "data/features_phase4.parquet"


def main():
    print(f"Loading {PHASE3}...")
    df = pd.read_parquet(PHASE3)
    df["date"] = pd.to_datetime(df["date"])
    print(f"  Shape: {df.shape}")

    print("Adding macro features...")
    df = add_macro_features(df, MACRO)

    phase3_cols = set(pd.read_parquet(PHASE3).columns)
    new_cols = [c for c in df.columns if c not in phase3_cols]
    print(f"  Added {len(new_cols)} new columns: {new_cols}")

    print(f"\nSaving to {OUTPUT}...")
    df.to_parquet(OUTPUT, index=False)
    print(f"Done. Shape: {df.shape}")
    print(f"Total columns: {df.shape[1]}")

    # Feature/target summary
    target_cols = [c for c in df.columns if c.startswith("fwd_")]
    feat_cols   = [c for c in df.columns if c not in target_cols
                   and c not in ["ticker", "date", "Open", "High", "Low", "Close", "Volume"]]
    print(f"\nFeature columns : {len(feat_cols)}")
    print(f"Target columns  : {len(target_cols)}")
    print(f"Rows            : {len(df):,}")


if __name__ == "__main__":
    main()
