"""
Filter S&P 500 OHLCV data to tickers with clean (null-free) data
within their membership period.

Outputs:
  data/clean_tickers.csv          - list of clean tickers with metadata
  data/sp500_clean_ohlcv.parquet  - filtered OHLCV for clean tickers only
"""

import pandas as pd
import numpy as np

# ── Load ──────────────────────────────────────────────────────────────────────
ohlcv = pd.read_parquet("data/sp500_daily_ohlcv.parquet")
ohlcv["date"] = pd.to_datetime(ohlcv["date"])

membership = pd.read_csv("data/sp500_membership_intervals.csv")
membership["start_date"] = pd.to_datetime(membership["start_date"])
membership["end_date"] = pd.to_datetime(membership["end_date"])

constituents = pd.read_csv("data/sp500_current_constituents.csv")

max_date = ohlcv["date"].max()

# ── Evaluate each ticker within its membership window ─────────────────────────
results = []
for _, row in membership.iterrows():
    ticker = row["ticker"]
    start = row["start_date"]
    end = row["end_date"] if pd.notna(row["end_date"]) else max_date

    mask = (ohlcv["ticker"] == ticker) & (ohlcv["date"] >= start) & (ohlcv["date"] <= end)
    sub = ohlcv[mask]

    if len(sub) == 0:
        continue

    null_pct = sub[["Open", "High", "Low", "Close", "Volume"]].isnull().mean().mean()
    results.append({
        "ticker": ticker,
        "start_date": start,
        "end_date": end,
        "trading_days": len(sub),
        "null_pct": round(null_pct, 6),
        "is_current": ticker in constituents["ticker"].values,
    })

summary = pd.DataFrame(results)

# ── Split: clean = 0% nulls within membership period ─────────────────────────
clean = summary[summary["null_pct"] == 0].copy()
dirty = summary[summary["null_pct"] > 0].copy()

print(f"Total tickers evaluated : {len(summary)}")
print(f"Clean (0% null)         : {len(clean)}")
print(f"  - current constituents: {clean['is_current'].sum()}")
print(f"  - historical only     : (~clean['is_current']).sum() = {(~clean['is_current']).sum()}")
print(f"Dirty (>0% null)        : {len(dirty)}")

# ── Save clean ticker list ────────────────────────────────────────────────────
clean_sorted = clean.sort_values(["is_current", "ticker"], ascending=[False, True])
clean_sorted.to_csv("data/clean_tickers.csv", index=False)
print(f"\nSaved: data/clean_tickers.csv ({len(clean_sorted)} tickers)")

# ── Save filtered OHLCV ───────────────────────────────────────────────────────
clean_ticker_set = set(clean["ticker"])

# Keep only rows within each ticker's membership window
keep_rows = []
for _, row in clean.iterrows():
    mask = (
        (ohlcv["ticker"] == row["ticker"]) &
        (ohlcv["date"] >= row["start_date"]) &
        (ohlcv["date"] <= row["end_date"])
    )
    keep_rows.append(ohlcv[mask])

clean_ohlcv = pd.concat(keep_rows, ignore_index=True)
clean_ohlcv = clean_ohlcv.sort_values(["ticker", "date"]).reset_index(drop=True)
clean_ohlcv.to_parquet("data/sp500_clean_ohlcv.parquet", index=False)
print(f"Saved: data/sp500_clean_ohlcv.parquet ({len(clean_ohlcv):,} rows, {clean_ohlcv['ticker'].nunique()} tickers)")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Clean ticker breakdown ===")
print(f"Current S&P 500 members : {clean['is_current'].sum()}")
print(f"Historical members only : {(~clean['is_current']).sum()}")
print(f"Date range              : {clean_ohlcv['date'].min().date()} → {clean_ohlcv['date'].max().date()}")
