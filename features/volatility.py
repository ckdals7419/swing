"""
Category 4: Volatility Indicators — ATR, Bollinger, Keltner, Realized Vol, Parkinson, Garman-Klass
"""
import numpy as np
import pandas as pd


def add_volatility(df: pd.DataFrame) -> pd.DataFrame:
    cl = df["Close"]
    hi = df["High"]
    lo = df["Low"]
    op = df["Open"]
    log_ret = np.log(cl / cl.shift(1))

    # ATR (14)
    tr = pd.concat([
        hi - lo,
        (hi - cl.shift(1)).abs(),
        (lo - cl.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.ewm(alpha=1 / 14, adjust=False).mean()
    df["atr_14"] = atr14
    df["atr_pct_14"] = atr14 / cl

    # Bollinger Bands (20, 2σ)
    sma20 = cl.rolling(20).mean()
    std20 = cl.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_width = bb_upper - bb_lower
    df["bb_upper"] = bb_upper
    df["bb_lower"] = bb_lower
    df["bb_width"] = bb_width / cl           # normalized
    df["bb_pct"] = (cl - bb_lower) / bb_width.replace(0, np.nan)

    # Keltner Channel (20, 2×ATR)
    ema20 = cl.ewm(span=20, adjust=False).mean()
    atr20 = tr.ewm(alpha=1 / 20, adjust=False).mean()
    df["keltner_upper"] = ema20 + 2 * atr20
    df["keltner_lower"] = ema20 - 2 * atr20
    df["keltner_pct"] = (cl - df["keltner_lower"]) / (
        df["keltner_upper"] - df["keltner_lower"]
    ).replace(0, np.nan)

    # Realized Volatility (annualized)
    for w in [5, 10, 20, 60]:
        df[f"realized_vol_{w}d"] = log_ret.rolling(w).std() * np.sqrt(252)

    # Vol regime: short-term vs long-term
    df["vol_ratio_10_60"] = df["realized_vol_10d"] / df["realized_vol_60d"].replace(0, np.nan)

    # Parkinson Volatility (20d)
    df["parkinson_vol_20d"] = np.sqrt(
        (1 / (4 * np.log(2))) * (np.log(hi / lo) ** 2).rolling(20).mean()
    ) * np.sqrt(252)

    # Garman-Klass Volatility (20d)
    gk = 0.5 * (np.log(hi / lo) ** 2) - (2 * np.log(2) - 1) * (np.log(cl / op) ** 2)
    df["garman_klass_vol_20d"] = np.sqrt(gk.rolling(20).mean()) * np.sqrt(252)

    return df
