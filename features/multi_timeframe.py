"""
Category 6: Multi-Timeframe Features — Returns, Drawdowns, Rolling RSI/Vol/Volume at multiple windows
"""
import numpy as np
import pandas as pd


def add_multi_timeframe(df: pd.DataFrame) -> pd.DataFrame:
    cl = df["Close"]
    vol = df["Volume"]
    log_ret = np.log(cl / cl.shift(1))

    # Multi-window returns (already in momentum but repeated here for completeness)
    for w in [1, 3, 5, 10, 20, 60]:
        col = f"return_{w}d"
        if col not in df.columns:
            df[col] = cl.pct_change(w)

    # Rolling drawdown from trailing high
    for w in [20, 60, 252]:
        rolling_max = cl.rolling(w).max()
        df[f"drawdown_from_high_{w}d"] = (cl - rolling_max) / rolling_max

    # 52-week metrics
    high_252 = cl.rolling(252).max()
    low_252 = cl.rolling(252).min()
    df["pct_from_52w_high"] = (cl - high_252) / high_252
    df["pct_from_52w_low"] = (cl - low_252) / low_252.replace(0, np.nan)
    df["days_since_52w_high"] = cl.rolling(252).apply(
        lambda x: len(x) - 1 - np.argmax(x), raw=True
    )

    # Rolling volume ratios at different windows (supplement volume.py)
    for w in [5, 10, 20]:
        col = f"volume_ratio_{w}"
        if col not in df.columns:
            vol_ma = vol.rolling(w).mean()
            df[col] = vol / vol_ma.replace(0, np.nan)

    # Volatility at multiple windows (normalized, annualized)
    for w in [5, 10, 20, 60]:
        col = f"realized_vol_{w}d"
        if col not in df.columns:
            df[col] = log_ret.rolling(w).std() * np.sqrt(252)

    # Rolling RSI at multiple windows
    from features.momentum import _rsi
    for w in [7, 14, 21]:
        col = f"rsi_{w}"
        if col not in df.columns:
            df[col] = _rsi(cl, w)

    return df
