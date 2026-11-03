"""
Phase 4: Download macroeconomic data from FRED (no API key needed for public series).

Series downloaded:
  FEDFUNDS    - Federal Funds Rate (monthly)
  T10Y2Y      - 10Y-2Y Treasury spread (daily)
  T10Y3M      - 10Y-3M Treasury spread (daily)
  CPIAUCSL    - CPI (monthly, SA)
  UNRATE      - Unemployment rate (monthly)
  NAPM        - ISM Manufacturing PMI (monthly)
  BAMLH0A0HYM2EY - HY credit spread (daily)
  DTWEXBGS    - USD index (daily)
  DCOILWTICO  - WTI Crude Oil (daily)
  GOLDAMGBD228NLBM - Gold price (daily)

Outputs:
  data/macro_daily.csv  - all series forward-filled to daily frequency
"""

import requests
import pandas as pd
from datetime import date

START = "1990-01-01"
END   = date.today().strftime("%Y-%m-%d")

FRED_SERIES = {
    "fed_funds_rate":    "FEDFUNDS",
    "yield_curve_10y2y": "T10Y2Y",
    "yield_curve_10y3m": "T10Y3M",
    "cpi":               "CPIAUCSL",
    "unemployment_rate": "UNRATE",
    "ism_pmi":           "MANEMP",
    "hy_credit_spread":  "BAMLH0A0HYM2EY",
    "dxy":               "DTWEXBGS",
    "oil_price":         "DCOILWTICO",
}
# Gold fetched separately from yfinance

FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&vintage_date={end}"

def fetch_fred(series_id, name):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    from io import StringIO
    df = pd.read_csv(StringIO(r.text))
    df.columns = ["date", name]
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= START]
    df[name] = pd.to_numeric(df[name], errors="coerce")
    return df.set_index("date")[name]

frames = []
for name, series_id in FRED_SERIES.items():
    try:
        s = fetch_fred(series_id, name)
        frames.append(s)
        print(f"  {name} ({series_id}): {len(s)} obs, {s.index.min().date()} -> {s.index.max().date()}")
    except Exception as e:
        print(f"  [WARN] {name} ({series_id}): {e}")

# Gold via yfinance (GLD ETF as proxy)
try:
    import yfinance as yf
    gld = yf.download("GLD", start=START, progress=False, auto_adjust=True)
    gld = gld[["Close"]].reset_index()
    gld.columns = ["date", "gold_price"]
    gld["date"] = pd.to_datetime(gld["date"])
    if hasattr(gld["date"].dt, "tz") and gld["date"].dt.tz is not None:
        gld["date"] = gld["date"].dt.tz_localize(None)
    gld = gld.set_index("date")["gold_price"]
    frames.append(gld)
    print(f"  gold_price (GLD/yfinance): {len(gld)} obs, {gld.index.min().date()} -> {gld.index.max().date()}")
except Exception as e:
    print(f"  [WARN] gold_price: {e}")

macro = pd.concat(frames, axis=1)
macro.index.name = "date"
macro = macro.reset_index()
macro["date"] = pd.to_datetime(macro["date"])

# Build a full daily date range and forward-fill monthly/weekly series
all_dates = pd.date_range(macro["date"].min(), macro["date"].max(), freq="B")
macro = macro.set_index("date").reindex(all_dates).ffill().reset_index()
macro = macro.rename(columns={"index": "date"})

# Derived macro features
macro["cpi_yoy"] = macro["cpi"].pct_change(12)           # 12-month CPI change
macro["oil_change_20d"] = macro["oil_price"].pct_change(20) if "oil_price" in macro.columns else None
macro["gold_change_20d"] = macro["gold_price"].pct_change(20) if "gold_price" in macro.columns else None
macro["rate_change_direction"] = macro["fed_funds_rate"].diff(1).apply(
    lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
)
macro["yield_curve_inverted"] = (macro["yield_curve_10y2y"] < 0).astype(int)

macro.to_csv("data/macro_daily.csv", index=False)
print(f"\nSaved: data/macro_daily.csv  ({len(macro)} rows, {macro.shape[1]} cols)")
print(macro.tail(3).to_string())
