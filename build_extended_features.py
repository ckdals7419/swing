"""
Build extended features file from 1999 using QQQ + synthetic LETF data.

Uses QQQ full history + synthetic QLD/TQQQ prices to extend the
qld_features.parquet back to 1999 (dot-com era).

Only builds features needed for SMA200 DCA backtests:
  - qqq_vs_sma200, qqq_vs_sma50, sma_cross, sma50_slope, sma200_slope
  - vix_close, vix_sma20, vix_percentile, vix_change_5d
  - qqq_drawdown, qld_drawdown
  - realized_vol_20d
  - bil_return_1d (from fed funds)
  - qld/tqqq open/close (synthetic before real data, real after)

Output: data/qld_features_extended.parquet
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def main():
    # ── Load data sources ────────────────────────────────────────────────────
    print("Loading QQQ full history...")
    qqq = pd.read_parquet("data/qqq_full.parquet")
    qqq["date"] = pd.to_datetime(qqq["date"])
    qqq = qqq.sort_values("date").set_index("date")
    print(f"  QQQ: {qqq.index[0].date()} -> {qqq.index[-1].date()} ({len(qqq):,} days)")

    print("Loading synthetic LETF...")
    synth = pd.read_parquet("data/synthetic_letf.parquet")
    synth["date"] = pd.to_datetime(synth["date"])
    synth = synth.sort_values("date").set_index("date")

    print("Loading VIX...")
    vix_df = pd.read_csv("data/vix_daily.csv", parse_dates=["Date"])
    vix_raw = vix_df.set_index("Date")["vix_close"]

    print("Loading macro (fed funds)...")
    macro = pd.read_csv("data/macro_daily.csv", parse_dates=["date"])
    macro = macro.sort_values("date").set_index("date")

    # Load real prices to use where available
    print("Loading real QLD/TQQQ prices...")
    univ = pd.read_parquet("data/qld_universe.parquet")
    univ["date"] = pd.to_datetime(univ["date"])

    def get_real(ticker):
        t = univ[univ["ticker"] == ticker].sort_values("date").set_index("date")
        return t[["Open", "Close"]]

    real_qld  = get_real("QLD")
    real_tqqq = get_real("TQQQ")
    real_bil  = get_real("BIL")

    # ── Build on QQQ index ───────────────────────────────────────────────────
    idx = qqq.index
    close = qqq["Close"]
    hi    = qqq["High"]
    lo    = qqq["Low"]

    df = pd.DataFrame(index=idx)
    df.index.name = "date"

    # ── Trend features ───────────────────────────────────────────────────────
    sma50  = close.rolling(50, min_periods=50).mean()
    sma200 = close.rolling(200, min_periods=200).mean()
    df["qqq_vs_sma50"]     = close / sma50 - 1
    df["qqq_vs_sma200"]    = close / sma200 - 1
    df["sma50_slope"]      = sma50.pct_change(5)
    df["sma200_slope"]     = sma200.pct_change(20)
    df["sma_cross"]        = (sma50 > sma200).astype(float)
    df["price_vs_20d_high"] = close / close.rolling(20).max() - 1
    df["price_vs_60d_high"] = close / close.rolling(60).max() - 1

    # ── Momentum ─────────────────────────────────────────────────────────────
    df["return_1d"]  = close.pct_change(1)
    df["return_5d"]  = close.pct_change(5)
    df["return_20d"] = close.pct_change(20)
    df["return_60d"] = close.pct_change(60)

    # RSI
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - 100 / (1 + rs)

    # ── Volatility ───────────────────────────────────────────────────────────
    log_ret = np.log(close / close.shift(1))
    df["realized_vol_10d"] = log_ret.rolling(10).std() * np.sqrt(252)
    df["realized_vol_20d"] = log_ret.rolling(20).std() * np.sqrt(252)
    rv60 = log_ret.rolling(60).std() * np.sqrt(252)
    df["vol_ratio_10_60"]  = df["realized_vol_10d"] / rv60.replace(0, np.nan)

    # VIX
    vix = vix_raw.reindex(idx, method="ffill")
    df["vix_close"]      = vix
    df["vix_sma20"]      = vix.rolling(20, min_periods=5).mean()
    df["vix_percentile"] = vix.rolling(252, min_periods=50).rank(pct=True)
    df["vix_change_5d"]  = vix.pct_change(5)

    # ── Drawdown ─────────────────────────────────────────────────────────────
    qqq_peak           = close.rolling(252, min_periods=1).max()
    df["qqq_drawdown"] = close / qqq_peak - 1

    # QLD drawdown from synthetic (will be overwritten where real exists)
    synth_qld = synth["synth_qld_close"].reindex(idx, method="ffill")
    qld_peak  = synth_qld.rolling(252, min_periods=1).max()
    df["qld_drawdown"] = synth_qld / qld_peak - 1

    # ── Macro ────────────────────────────────────────────────────────────────
    macro_r = macro.reindex(idx, method="ffill")
    df["yield_curve_10y2y"] = macro_r.get("yield_curve_10y2y", np.nan)
    df["fed_funds_rate"]    = macro_r.get("fed_funds_rate", np.nan)

    # ── Cash proxy (BIL where available, else fed funds / 252) ───────────────
    bil_close = real_bil["Close"].reindex(idx, method="ffill")
    bil_ret   = bil_close.pct_change(1)
    ff_ret    = macro_r["fed_funds_rate"].fillna(1.0) / 100.0 / 252.0
    df["bil_return_1d"] = bil_ret.fillna(ff_ret)

    # ── QLD prices: real where available, synthetic before ───────────────────
    # Rescale synthetic to match real at first overlap date (handles split adjustments)
    qld_close_real = real_qld["Close"].reindex(idx)
    qld_open_real  = real_qld["Open"].reindex(idx)
    synth_qld_c    = synth["synth_qld_close"].reindex(idx)

    first_real_qld = qld_close_real.first_valid_index()
    qld_scale = qld_close_real.loc[first_real_qld] / synth_qld_c.loc[first_real_qld]
    synth_qld_c_scaled = synth_qld_c * qld_scale
    print(f"  QLD scale factor: {qld_scale:.6f} (splice at {first_real_qld.date()})")

    qqq_oc_ratio = qqq["Open"] / qqq["Close"].shift(1)
    synth_qld_open = synth_qld_c_scaled.shift(1) * (1 + 2 * (qqq_oc_ratio - 1))

    df["qld_close"] = qld_close_real.fillna(synth_qld_c_scaled)
    df["qld_open"]  = qld_open_real.fillna(synth_qld_open)

    # ── TQQQ prices: real where available, synthetic before ──────────────────
    tqqq_close_real = real_tqqq["Close"].reindex(idx)
    tqqq_open_real  = real_tqqq["Open"].reindex(idx)
    synth_tqqq_c    = synth["synth_tqqq_close"].reindex(idx)

    first_real_tqqq = tqqq_close_real.first_valid_index()
    tqqq_scale = tqqq_close_real.loc[first_real_tqqq] / synth_tqqq_c.loc[first_real_tqqq]
    synth_tqqq_c_scaled = synth_tqqq_c * tqqq_scale
    print(f"  TQQQ scale factor: {tqqq_scale:.6f} (splice at {first_real_tqqq.date()})")

    synth_tqqq_open = synth_tqqq_c_scaled.shift(1) * (1 + 3 * (qqq_oc_ratio - 1))

    df["tqqq_close"] = tqqq_close_real.fillna(synth_tqqq_c_scaled)
    df["tqqq_open"]  = tqqq_open_real.fillna(synth_tqqq_open)

    # QQQ prices
    df["qqq_close"] = close
    df["qqq_open"]  = qqq["Open"]

    # ── Save ─────────────────────────────────────────────────────────────────
    df = df.reset_index()
    out = "data/qld_features_extended.parquet"
    df.to_parquet(out, index=False)
    print(f"\nSaved: {out}")
    print(f"  {len(df):,} rows, {len(df.columns)} columns")
    print(f"  {df['date'].min().date()} -> {df['date'].max().date()}")

    # Verify overlap with original features
    orig = pd.read_parquet("data/qld_features.parquet")
    orig["date"] = pd.to_datetime(orig["date"])
    print(f"\n  Original features: {orig['date'].min().date()} -> {orig['date'].max().date()} ({len(orig):,} rows)")
    print(f"  Extended features: {df['date'].min().date()} -> {df['date'].max().date()} ({len(df):,} rows)")
    print(f"  New rows added:    {len(df) - len(orig):,} (pre-2006 history)")


if __name__ == "__main__":
    main()
