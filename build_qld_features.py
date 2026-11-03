"""
Phase B — Feature Engineering for QLD Weight Strategy.

Loads QLD/QQQ/SPY prices + VIX + macro data and builds ~30 features
for the regime classifier. All signal features are computed on QQQ
(clean 1x series). QLD is used only for the target variable and for
the qld_drawdown feature.

Output: data/qld_features.parquet
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

UNIVERSE_PATH = "data/qld_universe.parquet"
VIX_PATH      = "data/vix_daily.csv"
MACRO_PATH    = "data/macro_daily.csv"
OUTPUT_PATH   = "data/qld_features.parquet"


# ── Load price data ────────────────────────────────────────────────────────────

def load_prices():
    univ = pd.read_parquet(UNIVERSE_PATH)
    univ["date"] = pd.to_datetime(univ["date"])

    def get(ticker):
        df = univ[univ["ticker"] == ticker].sort_values("date").reset_index(drop=True)
        return df.set_index("date")

    qqq = get("QQQ")
    qld = get("QLD")
    spy = get("SPY")
    bil = get("BIL") if "BIL" in univ["ticker"].values else None
    vxn = get("^VXN") if "^VXN" in univ["ticker"].values else None

    print(f"QQQ: {len(qqq)} rows  {qqq.index.min().date()} -> {qqq.index.max().date()}")
    print(f"QLD: {len(qld)} rows  {qld.index.min().date()} -> {qld.index.max().date()}")

    return qqq, qld, spy, bil, vxn


def load_vix():
    vix = pd.read_csv(VIX_PATH, parse_dates=["Date"])
    vix = vix.rename(columns={"Date": "date", "vix_close": "vix_raw"})
    return vix.set_index("date")["vix_raw"]


def load_macro():
    macro = pd.read_csv(MACRO_PATH, parse_dates=["date"])
    return macro.set_index("date")


# ── ADX ────────────────────────────────────────────────────────────────────────

def compute_adx(hi, lo, cl, period=14):
    tr = pd.concat([
        hi - lo,
        (hi - cl.shift(1)).abs(),
        (lo - cl.shift(1)).abs(),
    ], axis=1).max(axis=1)

    dm_plus  = hi.diff().clip(lower=0)
    dm_minus = (-lo.diff()).clip(lower=0)
    mask = dm_plus == dm_minus
    dm_plus[mask]  = 0
    dm_minus[mask] = 0
    dm_plus[dm_plus < dm_minus]   = 0
    dm_minus[dm_minus < dm_plus]  = 0

    atr      = tr.ewm(alpha=1/period, adjust=False).mean()
    di_plus  = 100 * dm_plus.ewm(alpha=1/period, adjust=False).mean() / atr
    di_minus = 100 * dm_minus.ewm(alpha=1/period, adjust=False).mean() / atr
    dx       = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    adx      = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx


# ── MACD ───────────────────────────────────────────────────────────────────────

def compute_macd(close, fast=12, slow=26, signal=9):
    ema_fast   = close.ewm(span=fast, adjust=False).mean()
    ema_slow   = close.ewm(span=slow, adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist       = macd_line - signal_line
    return hist


# ── RSI ────────────────────────────────────────────────────────────────────────

def compute_rsi(close, period=14):
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


# ── Feature builder ────────────────────────────────────────────────────────────

def build_features(qqq, qld, spy, bil, vxn, vix_raw, macro):
    # Work on a common daily index (QQQ trading days)
    idx = qqq.index

    close = qqq["Close"]
    hi    = qqq["High"]
    lo    = qqq["Low"]

    df = pd.DataFrame(index=idx)
    df.index.name = "date"

    # ── B1. Trend ──────────────────────────────────────────────────────────────
    sma50       = close.rolling(50, min_periods=50).mean()
    sma200      = close.rolling(200, min_periods=200).mean()
    df["qqq_vs_sma50"]     = close / sma50 - 1
    df["qqq_vs_sma200"]    = close / sma200 - 1
    df["sma50_slope"]      = sma50.pct_change(5)
    df["sma200_slope"]     = sma200.pct_change(20)
    df["sma_cross"]        = (sma50 > sma200).astype(float)
    df["price_vs_20d_high"] = close / close.rolling(20).max() - 1
    df["price_vs_60d_high"] = close / close.rolling(60).max() - 1
    df["adx_14"]           = compute_adx(hi, lo, close, 14)

    # ── B2. Momentum ───────────────────────────────────────────────────────────
    df["return_1d"]   = close.pct_change(1)
    df["return_5d"]   = close.pct_change(5)
    df["return_20d"]  = close.pct_change(20)
    df["return_60d"]  = close.pct_change(60)
    df["rsi_14"]      = compute_rsi(close, 14)
    df["macd_hist"]   = compute_macd(close)
    df["roc_20"]      = close.pct_change(20)   # same as return_20d; keep for naming consistency

    # ── B3. Volatility ─────────────────────────────────────────────────────────
    log_ret = np.log(close / close.shift(1))
    rv10    = log_ret.rolling(10).std()  * np.sqrt(252)
    rv20    = log_ret.rolling(20).std()  * np.sqrt(252)
    rv60    = log_ret.rolling(60).std()  * np.sqrt(252)
    df["realized_vol_10d"] = rv10
    df["realized_vol_20d"] = rv20
    df["vol_ratio_10_60"]  = rv10 / rv60.replace(0, np.nan)

    # VIX (reindex to QQQ trading days)
    vix = vix_raw.reindex(idx, method="ffill")
    df["vix_close"]       = vix
    df["vix_sma20"]       = vix.rolling(20, min_periods=5).mean()
    df["vix_percentile"]  = vix.rolling(252, min_periods=50).rank(pct=True)
    df["vix_change_5d"]   = vix.pct_change(5)   # proxy for term structure if VXN unavailable

    # VXN if available
    if vxn is not None and len(vxn) > 100:
        vxn_s = vxn["Close"].reindex(idx, method="ffill")
        df["vxn_close"]      = vxn_s
        df["vix_vs_vxn"]     = vix / vxn_s.replace(0, np.nan) - 1
    else:
        df["vxn_close"]  = np.nan
        df["vix_vs_vxn"] = np.nan

    # ── B4. Drawdown ───────────────────────────────────────────────────────────
    qqq_peak              = close.rolling(252, min_periods=1).max()
    df["qqq_drawdown"]    = close / qqq_peak - 1

    qld_close             = qld["Close"].reindex(idx, method="ffill")
    qld_peak              = qld_close.rolling(252, min_periods=1).max()
    df["qld_drawdown"]    = qld_close / qld_peak - 1

    # Days since last all-time high (rolling 252d)
    at_high  = (close >= close.rolling(252, min_periods=1).max()).astype(float)
    days_dd  = at_high.groupby((at_high != at_high.shift()).cumsum()).cumcount()
    df["days_in_drawdown"] = days_dd * (1 - at_high)   # 0 on new ATH days

    # ── B5. Breadth / Macro ────────────────────────────────────────────────────
    macro_r = macro.reindex(idx, method="ffill")

    df["yield_curve_10y2y"] = macro_r["yield_curve_10y2y"] if "yield_curve_10y2y" in macro_r else np.nan
    df["fed_funds_rate"]    = macro_r["fed_funds_rate"]     if "fed_funds_rate"    in macro_r else np.nan
    df["credit_spread"]     = macro_r["hy_credit_spread"]   if "hy_credit_spread"  in macro_r else np.nan
    df["dxy_change_20d"]    = macro_r["dxy"].pct_change(20) if "dxy" in macro_r.columns else np.nan
    df["gold_change_20d"]   = macro_r["gold_change_20d"]    if "gold_change_20d"   in macro_r else np.nan

    spy_close   = spy["Close"].reindex(idx, method="ffill")
    spy_ret_20d = spy_close.pct_change(20)
    qqq_ret_20d = close.pct_change(20)
    df["spy_vs_qqq_20d"]  = spy_ret_20d - qqq_ret_20d

    # ── B6. Cash proxy return ─────────────────────────────────────────────────
    if bil is not None and len(bil) > 100:
        bil_close   = bil["Close"].reindex(idx, method="ffill")
        df["bil_return_1d"] = bil_close.pct_change(1)
    else:
        # Approximate: fed_funds_rate / 252
        df["bil_return_1d"] = macro_r["fed_funds_rate"].fillna(1.0) / 100.0 / 252.0

    # ── Targets ────────────────────────────────────────────────────────────────
    # Use QLD returns as targets (what we'll trade)
    qld_ret = qld_close.pct_change(1)

    # Forward returns: shift(-N) means the return STARTING next day for N days
    # We compute on QLD adjusted close
    def fwd_return(n):
        return qld_close.shift(-n) / qld_close - 1

    df["fwd_return_5d"]  = fwd_return(5)
    df["fwd_return_10d"] = fwd_return(10)
    df["fwd_return_20d"] = fwd_return(20)
    df["fwd_up_5d"]      = (df["fwd_return_5d"]  > 0).astype(float)
    df["fwd_up_10d"]     = (df["fwd_return_10d"] > 0).astype(float)
    df["fwd_up_20d"]     = (df["fwd_return_20d"] > 0).astype(float)

    # Keep raw prices for the backtest engine
    df["qld_close"]      = qld_close
    df["qld_open"]       = qld["Open"].reindex(idx, method="ffill")
    df["qqq_close"]      = close
    df["qqq_open"]       = qqq["Open"].reindex(idx, method="ffill")

    df = df.reset_index()
    print(f"Features: {len(df):,} rows, {len(df.columns)} columns")
    print(f"Date range: {df['date'].min().date()} -> {df['date'].max().date()}")
    null_pct = df.isnull().mean().sort_values(ascending=False)
    high_null = null_pct[null_pct > 0.1]
    if len(high_null) > 0:
        print(f"Columns with >10% nulls:")
        for c, v in high_null.items():
            print(f"  {c}: {v:.1%}")

    return df


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading price data...")
    qqq, qld, spy, bil, vxn = load_prices()

    print("Loading VIX...")
    vix_raw = load_vix()

    print("Loading macro...")
    macro = load_macro()

    print("Building features...")
    df = build_features(qqq, qld, spy, bil, vxn, vix_raw, macro)

    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
