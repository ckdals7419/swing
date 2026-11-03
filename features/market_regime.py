"""
Category 10: Market Regime Features
— SPY trend/drawdown, VIX levels, market breadth, beta/correlation to market

Requires: data/spy_daily.csv, data/vix_daily.csv
"""

import numpy as np
import pandas as pd


def build_market_features(spy_path: str, vix_path: str) -> pd.DataFrame:
    """Build a date-indexed DataFrame of market-wide features."""
    spy = pd.read_csv(spy_path, parse_dates=["Date"])
    spy = spy.sort_values("Date").reset_index(drop=True)
    spy["spy_close"] = spy["Adj Close"] if "Adj Close" in spy.columns else spy["Close"]
    spy["spy_log_ret"] = np.log(spy["spy_close"] / spy["spy_close"].shift(1))

    vix = pd.read_csv(vix_path, parse_dates=["Date"])
    vix = vix.sort_values("Date").reset_index(drop=True)

    mkt = spy[["Date"]].copy()

    # SPY returns
    for w in [1, 5, 20, 60]:
        mkt[f"spy_return_{w}d"] = spy["spy_close"].pct_change(w).values

    # SPY above/below key MAs
    for w in [50, 200]:
        sma = spy["spy_close"].rolling(w).mean()
        mkt[f"spy_above_sma{w}"] = (spy["spy_close"] > sma).astype(int).values
        mkt[f"spy_vs_sma{w}"] = ((spy["spy_close"] - sma) / sma).values

    # SPY drawdown from 252d high
    high_252 = spy["spy_close"].rolling(252).max()
    mkt["spy_drawdown_252d"] = ((spy["spy_close"] - high_252) / high_252).values

    # SPY realized volatility
    for w in [10, 20, 60]:
        mkt[f"spy_vol_{w}d"] = (spy["spy_log_ret"].rolling(w).std() * np.sqrt(252)).values

    # Market regime: trend strength (ADX-like slope of SMA50)
    sma50 = spy["spy_close"].rolling(50).mean()
    mkt["spy_sma50_slope"] = (sma50.diff(10) / sma50.shift(10)).values

    # VIX features
    mkt = mkt.merge(vix.rename(columns={"Date": "Date"}), on="Date", how="left")
    mkt["vix_close"] = mkt["vix_close"].ffill()

    mkt["vix_sma_20"] = mkt["vix_close"].rolling(20).mean()
    mkt["vix_change_5d"] = mkt["vix_close"].pct_change(5)

    # VIX percentile rank (rolling 252d)
    mkt["vix_percentile_252d"] = mkt["vix_close"].rolling(252).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

    # VIX regime: low (<15), medium (15-25), high (25-35), extreme (>35)
    mkt["vix_regime"] = pd.cut(
        mkt["vix_close"],
        bins=[0, 15, 25, 35, np.inf],
        labels=[0, 1, 2, 3]
    ).astype(float)

    # VIX term structure proxy: deviation from 20d SMA (contango/backwardation)
    mkt["vix_term_structure"] = mkt["vix_close"] - mkt["vix_sma_20"]

    mkt = mkt.rename(columns={"Date": "date"})
    return mkt.sort_values("date").reset_index(drop=True)


def add_market_regime(df: pd.DataFrame, spy_path: str, vix_path: str) -> pd.DataFrame:
    """Merge market-wide features into the stock feature dataframe."""
    mkt = build_market_features(spy_path, vix_path)

    df = df.merge(mkt, on="date", how="left")

    # Per-stock beta and correlation to SPY (rolling 60d)
    spy = pd.read_csv(spy_path, parse_dates=["Date"])
    spy["spy_ret"] = spy["Adj Close"].pct_change() if "Adj Close" in spy.columns else spy["Close"].pct_change()
    spy_ret_map = spy.set_index("Date")["spy_ret"]

    df["spy_ret_aligned"] = df["date"].map(spy_ret_map)

    if "return_1d" not in df.columns:
        df["return_1d"] = df.groupby("ticker")["Close"].pct_change(1)

    def rolling_beta_corr(grp):
        stock = grp["return_1d"]
        mkt_r = grp["spy_ret_aligned"]
        roll_cov = stock.rolling(60).cov(mkt_r)
        roll_var = mkt_r.rolling(60).var()
        grp["beta_60d"] = roll_cov / roll_var.replace(0, np.nan)
        grp["corr_to_spy_60d"] = stock.rolling(60).corr(mkt_r)
        return grp

    df = df.groupby("ticker", group_keys=False).apply(rolling_beta_corr)
    df = df.drop(columns=["spy_ret_aligned"])

    return df
