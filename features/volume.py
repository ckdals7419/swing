"""
Category 5: Volume Features — OBV, MFI, Chaikin, VWAP, VPT, A/D
"""
import numpy as np
import pandas as pd


def add_volume(df: pd.DataFrame) -> pd.DataFrame:
    cl = df["Close"]
    hi = df["High"]
    lo = df["Low"]
    op = df["Open"]
    vol = df["Volume"]

    # Basic volume features
    vol_sma20 = vol.rolling(20).mean()
    df["volume_sma_20"] = vol_sma20
    df["volume_ratio_20"] = vol / vol_sma20.replace(0, np.nan)
    df["volume_ratio_5"] = vol / vol.rolling(5).mean().replace(0, np.nan)
    df["volume_ratio_10"] = vol / vol.rolling(10).mean().replace(0, np.nan)
    df["volume_volatility_20"] = vol.rolling(20).std() / vol_sma20.replace(0, np.nan)

    # On-Balance Volume
    obv = (np.sign(cl.diff()) * vol).cumsum()
    df["obv"] = obv
    df["obv_slope_5"] = obv.diff(5) / (obv.abs().rolling(5).mean().replace(0, np.nan))

    # Accumulation / Distribution Line
    clv = ((cl - lo) - (hi - cl)) / (hi - lo).replace(0, np.nan)
    ad = (clv * vol).cumsum()
    df["ad_line"] = ad
    df["ad_slope_5"] = ad.diff(5) / (ad.abs().rolling(5).mean().replace(0, np.nan))

    # Chaikin Money Flow (20)
    mf_vol = clv * vol
    df["chaikin_mf_20"] = mf_vol.rolling(20).sum() / vol.rolling(20).sum().replace(0, np.nan)

    # Volume Price Trend
    vpt = (cl.pct_change() * vol).cumsum()
    df["volume_price_trend"] = vpt
    df["vpt_slope_5"] = vpt.diff(5)

    # Money Flow Index (14) — volume-weighted RSI
    df["mfi_14"] = _mfi(hi, lo, cl, vol, period=14)

    # VWAP deviation (rolling 20d approximation)
    typical = (hi + lo + cl) / 3
    vwap = (typical * vol).rolling(20).sum() / vol.rolling(20).sum().replace(0, np.nan)
    df["vwap_deviation"] = (cl - vwap) / vwap.replace(0, np.nan)

    return df


def _mfi(hi, lo, cl, vol, period=14):
    tp = (hi + lo + cl) / 3
    mf = tp * vol
    pos_mf = mf.where(tp > tp.shift(1), 0)
    neg_mf = mf.where(tp < tp.shift(1), 0)
    mfr = pos_mf.rolling(period).sum() / neg_mf.rolling(period).sum().replace(0, np.nan)
    return 100 - 100 / (1 + mfr)
