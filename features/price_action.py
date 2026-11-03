"""
Category 1: Price Action & Candle Pattern Features
"""
import numpy as np
import pandas as pd


def add_price_action(df: pd.DataFrame) -> pd.DataFrame:
    hi = df["High"]
    lo = df["Low"]
    op = df["Open"]
    cl = df["Close"]

    hl_range = (hi - lo).replace(0, np.nan)

    df["body_ratio"] = (cl - op) / hl_range
    df["upper_shadow"] = (hi - np.maximum(op, cl)) / hl_range
    df["lower_shadow"] = (np.minimum(op, cl) - lo) / hl_range
    df["gap_pct"] = (op - cl.shift(1)) / cl.shift(1)
    df["intraday_range"] = hl_range / cl
    df["close_position"] = (cl - lo) / hl_range

    return df
