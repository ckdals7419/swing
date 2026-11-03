"""
Category 3: Momentum Indicators — RSI, MACD, Stochastic, CCI, TSI, ROC, Williams %R
"""
import numpy as np
import pandas as pd


def add_momentum(df: pd.DataFrame) -> pd.DataFrame:
    cl = df["Close"]
    hi = df["High"]
    lo = df["Low"]

    # RSI
    for w in [7, 14, 21]:
        df[f"rsi_{w}"] = _rsi(cl, w)

    # MACD (12/26/9)
    ema12 = cl.ewm(span=12, adjust=False).mean()
    ema26 = cl.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df["macd"] = macd
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd - macd_signal
    df["macd_hist_slope"] = df["macd_hist"].diff(3)

    # Stochastic (14, 3)
    low_min = lo.rolling(14).min()
    high_max = hi.rolling(14).max()
    stoch_k = 100 * (cl - low_min) / (high_max - low_min).replace(0, np.nan)
    df["stoch_k_14"] = stoch_k
    df["stoch_d_14"] = stoch_k.rolling(3).mean()

    # Rate of Change
    for w in [5, 10, 20]:
        df[f"roc_{w}"] = cl.pct_change(w)

    # Williams %R
    df["williams_r_14"] = -100 * (hi.rolling(14).max() - cl) / (
        hi.rolling(14).max() - lo.rolling(14).min()
    ).replace(0, np.nan)

    # CCI (20)
    tp = (hi + lo + cl) / 3
    tp_sma = tp.rolling(20).mean()
    tp_mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    df["cci_20"] = (tp - tp_sma) / (0.015 * tp_mad.replace(0, np.nan))

    # TSI (25, 13)
    df["tsi"] = _tsi(cl, r=25, s=13)

    # Simple momentum returns
    for w in [1, 3, 5, 10, 20]:
        df[f"momentum_{w}d"] = cl.pct_change(w)

    return df


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _tsi(series: pd.Series, r: int = 25, s: int = 13) -> pd.Series:
    diff = series.diff()
    abs_diff = diff.abs()
    smooth = lambda x, n: x.ewm(span=n, adjust=False).mean()
    num = smooth(smooth(diff, r), s)
    den = smooth(smooth(abs_diff, r), s)
    return 100 * num / den.replace(0, np.nan)
