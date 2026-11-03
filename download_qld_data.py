"""
Phase A — Download QLD/QQQ universe data.

Downloads daily OHLCV (adjusted) for:
  QLD   - 2x leveraged NASDAQ-100 (trade target)
  QQQ   - 1x NASDAQ-100 (signal source)
  TQQQ  - 3x NASDAQ-100 (optional comparison)
  SPY   - S&P 500 (benchmark)
  BIL   - SPDR Bloomberg 1-3 Month T-Bill (cash proxy, for interest on idle cash)
  ^VXN  - CBOE NASDAQ Volatility Index (if available)

Period: 2006-07-01 -> today (QLD inception: 2006-06-19)

Output: data/qld_universe.parquet
        Columns: date, ticker, Open, High, Low, Close, Adj_Close, Volume
"""

import warnings
import yfinance as yf
import pandas as pd
from datetime import date

warnings.filterwarnings("ignore")

START   = "2006-07-01"
END     = date.today().strftime("%Y-%m-%d")
TICKERS = ["QLD", "QQQ", "TQQQ", "SPY", "BIL", "^VXN"]
OUT     = "data/qld_universe.parquet"


def download_ticker(ticker: str) -> pd.DataFrame:
    raw = yf.download(ticker, start=START, end=END,
                      progress=False, auto_adjust=True)
    if raw.empty:
        print(f"  [WARN] {ticker}: no data returned")
        return pd.DataFrame()

    raw = raw.copy()
    raw.index = pd.to_datetime(raw.index)
    # Flatten MultiIndex columns if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]
    # Normalize column names
    raw = raw.rename(columns={"Adj Close": "Adj_Close"})
    if "Adj_Close" not in raw.columns:
        raw["Adj_Close"] = raw["Close"]

    raw = raw[["Open", "High", "Low", "Close", "Adj_Close", "Volume"]].copy()
    raw.index.name = "date"
    raw = raw.reset_index()
    raw["ticker"] = ticker
    if hasattr(raw["date"].dt, "tz") and raw["date"].dt.tz is not None:
        raw["date"] = raw["date"].dt.tz_localize(None)

    return raw


frames = []
for ticker in TICKERS:
    df = download_ticker(ticker)
    if not df.empty:
        frames.append(df)
        print(f"  {ticker:6s}: {len(df):,} rows  "
              f"{df['date'].min().date()} -> {df['date'].max().date()}")

universe = pd.concat(frames, ignore_index=True)
universe.to_parquet(OUT, index=False)
print(f"\nSaved: {OUT}  ({len(universe):,} rows, "
      f"{universe['ticker'].nunique()} tickers)")
