"""
Target variable generation — forward returns and binary labels.
Must be applied AFTER all features (uses future prices → only safe at training time).
"""
import numpy as np
import pandas as pd

FORWARD_WINDOWS = [5, 10, 20]
THRESHOLDS = {5: 0.015, 10: 0.02, 20: 0.03}  # binary label thresholds


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    cl = df["Close"]

    for w in FORWARD_WINDOWS:
        fwd = cl.shift(-w) / cl - 1
        df[f"fwd_return_{w}d"] = fwd

        thresh = THRESHOLDS[w]
        df[f"fwd_label_{w}d"] = np.where(
            fwd > thresh, 1,
            np.where(fwd < -thresh, -1, 0)
        )
        # Binary up/down (ignore neutral zone)
        df[f"fwd_up_{w}d"] = (fwd > thresh).astype(float)
        df[f"fwd_down_{w}d"] = (fwd < -thresh).astype(float)

    # Risk-adjusted return (fwd / realized vol)
    if "realized_vol_10d" in df.columns:
        df["fwd_sharpe_10d"] = (
            df["fwd_return_10d"] / df["realized_vol_10d"].replace(0, np.nan)
        )

    return df
