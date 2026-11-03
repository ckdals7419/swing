"""
Category 12: Statistical / Derived Features
— Hurst exponent, autocorrelation, skewness, kurtosis, entropy,
  mean-reversion score, breakout signals, run-length, 52-week metrics
"""
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis


def add_statistical(df: pd.DataFrame) -> pd.DataFrame:
    cl = df["Close"]
    log_ret = np.log(cl / cl.shift(1))

    # Rolling skewness & kurtosis (20d)
    df["skewness_20d"] = log_ret.rolling(20).apply(lambda x: skew(x, nan_policy="omit"), raw=True)
    df["kurtosis_20d"] = log_ret.rolling(20).apply(lambda x: kurtosis(x, nan_policy="omit"), raw=True)

    # Autocorrelation lag-1 (20d rolling)
    df["autocorr_lag1_20d"] = log_ret.rolling(20).apply(
        lambda x: pd.Series(x).autocorr(lag=1), raw=True
    )

    # Shannon entropy of return signs (20d)
    df["entropy_20d"] = log_ret.rolling(20).apply(_sign_entropy, raw=True)

    # Mean reversion z-score
    for w in [20, 60]:
        mu = cl.rolling(w).mean()
        sigma = cl.rolling(w).std()
        df[f"mean_reversion_zscore_{w}d"] = (cl - mu) / sigma.replace(0, np.nan)

    # Hurst exponent (63d rolling, simplified R/S)
    df["hurst_63d"] = cl.rolling(63).apply(_hurst_rs, raw=True)

    # Breakout signals
    for w in [20, 50]:
        rolling_high = cl.rolling(w).max()
        rolling_low = cl.rolling(w).min()
        rng = (rolling_high - rolling_low).replace(0, np.nan)
        df[f"breakout_high_{w}d"] = (cl >= rolling_high.shift(1)).astype(int)
        df[f"breakout_low_{w}d"] = (cl <= rolling_low.shift(1)).astype(int)
        df[f"channel_position_{w}d"] = (cl - rolling_low) / rng

    # Consecutive up/down days
    direction = np.sign(log_ret)
    df["consecutive_up_days"] = _run_length(direction, target=1)
    df["consecutive_down_days"] = _run_length(direction, target=-1)

    # 52-week metrics (if not already added by multi_timeframe)
    if "pct_from_52w_high" not in df.columns:
        high_252 = cl.rolling(252).max()
        low_252 = cl.rolling(252).min()
        df["pct_from_52w_high"] = (cl - high_252) / high_252
        df["pct_from_52w_low"] = (cl - low_252) / low_252.replace(0, np.nan)
        df["days_since_52w_high"] = cl.rolling(252).apply(
            lambda x: len(x) - 1 - np.argmax(x), raw=True
        )

    return df


def _sign_entropy(x: np.ndarray) -> float:
    """Shannon entropy of return sign distribution."""
    signs = np.sign(x[~np.isnan(x)])
    if len(signs) == 0:
        return np.nan
    p_up = np.mean(signs > 0)
    p_dn = np.mean(signs < 0)
    result = 0.0
    for p in [p_up, p_dn]:
        if p > 0:
            result -= p * np.log2(p)
    return result


def _hurst_rs(prices: np.ndarray) -> float:
    """Simplified R/S Hurst exponent estimate."""
    prices = prices[~np.isnan(prices)]
    n = len(prices)
    if n < 20:
        return np.nan
    log_ret = np.diff(np.log(prices))
    mean_r = np.mean(log_ret)
    deviations = np.cumsum(log_ret - mean_r)
    r = np.max(deviations) - np.min(deviations)
    s = np.std(log_ret, ddof=1)
    if s == 0:
        return np.nan
    return np.log(r / s) / np.log(n)


def _run_length(direction: pd.Series, target: int) -> pd.Series:
    """Count consecutive days where direction == target."""
    is_target = (direction == target).astype(int)
    result = []
    count = 0
    for v in is_target:
        if v == 1:
            count += 1
        else:
            count = 0
        result.append(count)
    return pd.Series(result, index=direction.index)
