"""
Phase 3: Download SPY and VIX data (1990-present).

Outputs:
  data/spy_daily.csv   - SPY OHLCV + Adj Close
  data/vix_daily.csv   - VIX daily close
"""

import pandas as pd
import yfinance as yf

def download(ticker, start="1990-01-01"):
    df = yf.download(ticker, start=start, auto_adjust=False, progress=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.reset_index()
    df.columns = [str(c).strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"])
    return df

print("Downloading SPY...")
spy = download("SPY")
spy.to_csv("data/spy_daily.csv", index=False)
print(f"  SPY: {len(spy)} rows, {spy['Date'].min().date()} -> {spy['Date'].max().date()}")

print("Downloading ^VIX...")
vix = download("^VIX")
vix = vix[["Date", "Close"]].rename(columns={"Close": "vix_close"})
vix.to_csv("data/vix_daily.csv", index=False)
print(f"  VIX: {len(vix)} rows, {vix['Date'].min().date()} -> {vix['Date'].max().date()}")

print("Done.")
