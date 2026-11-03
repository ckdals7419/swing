"""
Category 2: Trend Indicators — SMA, EMA, ADX, Aroon, Ichimoku
"""
import numpy as np
import pandas as pd


def add_trend(df: pd.DataFrame) -> pd.DataFrame:
    cl = df["Close"]
    hi = df["High"]
    lo = df["Low"]

    # SMA
    for w in [5, 10, 20, 50, 200]:
        sma = cl.rolling(w).mean()
        df[f"sma_{w}"] = sma
        df[f"price_vs_sma_{w}"] = (cl - sma) / sma
        df[f"sma_slope_{w}"] = sma.diff(5) / sma.shift(5)

    # EMA
    for w in [5, 10, 20, 50]:
        ema = cl.ewm(span=w, adjust=False).mean()
        df[f"ema_{w}"] = ema
        df[f"price_vs_ema_{w}"] = (cl - ema) / ema

    # MA cross signals
    df["sma_cross_5_20"] = (df["sma_5"] > df["sma_20"]).astype(int)
    df["sma_cross_50_200"] = (df["sma_50"] > df["sma_200"]).astype(int)

    # ADX (14)
    df = _add_adx(df, period=14)

    # Aroon (25)
    df = _add_aroon(df, period=25)

    # Ichimoku
    df = _add_ichimoku(df)

    return df


def _add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    hi = df["High"]
    lo = df["Low"]
    cl = df["Close"]

    tr = pd.concat([
        hi - lo,
        (hi - cl.shift(1)).abs(),
        (lo - cl.shift(1)).abs(),
    ], axis=1).max(axis=1)

    dm_plus = hi.diff().clip(lower=0)
    dm_minus = (-lo.diff()).clip(lower=0)

    # Mask where both DM are equal (no trend)
    mask = dm_plus == dm_minus
    dm_plus[mask] = 0
    dm_minus[mask] = 0
    # Keep only dominant direction
    dm_plus[dm_plus < dm_minus] = 0
    dm_minus[dm_minus < dm_plus] = 0

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    di_plus = 100 * dm_plus.ewm(alpha=1 / period, adjust=False).mean() / atr
    di_minus = 100 * dm_minus.ewm(alpha=1 / period, adjust=False).mean() / atr

    dx = (100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan))
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    df["adx_14"] = adx
    df["di_plus_14"] = di_plus
    df["di_minus_14"] = di_minus

    return df


def _add_aroon(df: pd.DataFrame, period: int = 25) -> pd.DataFrame:
    hi = df["High"]
    lo = df["Low"]

    aroon_up = hi.rolling(period + 1).apply(
        lambda x: (period - (period - np.argmax(x))) / period * 100, raw=True
    )
    aroon_down = lo.rolling(period + 1).apply(
        lambda x: (period - (period - np.argmin(x))) / period * 100, raw=True
    )

    df["aroon_up_25"] = aroon_up
    df["aroon_down_25"] = aroon_down
    df["aroon_osc_25"] = aroon_up - aroon_down

    return df


def _add_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    hi = df["High"]
    lo = df["Low"]

    tenkan = (hi.rolling(9).max() + lo.rolling(9).min()) / 2
    kijun = (hi.rolling(26).max() + lo.rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((hi.rolling(52).max() + lo.rolling(52).min()) / 2).shift(26)

    cl = df["Close"]
    df["ichimoku_tenkan"] = tenkan
    df["ichimoku_kijun"] = kijun
    df["ichimoku_cloud_top"] = np.maximum(senkou_a, senkou_b)
    df["ichimoku_cloud_bot"] = np.minimum(senkou_a, senkou_b)
    df["ichimoku_above_cloud"] = (cl > df["ichimoku_cloud_top"]).astype(int)
    df["ichimoku_below_cloud"] = (cl < df["ichimoku_cloud_bot"]).astype(int)
    df["ichimoku_tk_cross"] = (tenkan > kijun).astype(int)

    return df
