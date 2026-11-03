"""
Phase 2: Download GICS sector/industry data for all clean tickers.
Uses yfinance for sector info + Wikipedia S&P 500 list as fallback.

Outputs:
  data/ticker_sectors.csv  — ticker, sector, industry, market_cap
"""

import time
import pandas as pd
import yfinance as yf

TICKERS_FILE = "data/clean_tickers.csv"
OUTPUT = "data/ticker_sectors.csv"

GICS_SECTORS = {
    # yfinance names
    "Technology": 0,
    "Healthcare": 1,
    "Financial Services": 2,
    "Consumer Cyclical": 3,
    "Communication Services": 4,
    "Industrials": 5,
    "Consumer Defensive": 6,
    "Energy": 7,
    "Utilities": 8,
    "Real Estate": 9,
    "Basic Materials": 10,
    # standard GICS aliases
    "Health Care": 1,
    "Financials": 2,
    "Consumer Discretionary": 3,
    "Consumer Staples": 6,
    "Materials": 10,
}


def fetch_yfinance_sectors(tickers: list) -> pd.DataFrame:
    records = []
    for i, ticker in enumerate(tickers, 1):
        try:
            info = yf.Ticker(ticker).info
            records.append({
                "ticker": ticker,
                "sector": info.get("sector", None),
                "industry": info.get("industry", None),
                "market_cap": info.get("marketCap", None),
            })
        except Exception as e:
            records.append({"ticker": ticker, "sector": None, "industry": None, "market_cap": None})

        if i % 50 == 0 or i == len(tickers):
            print(f"  {i}/{len(tickers)} fetched")
            time.sleep(0.5)   # be polite to yfinance

    return pd.DataFrame(records)


def fetch_wikipedia_sectors() -> pd.DataFrame:
    """Fallback: scrape current S&P 500 list from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0][["Symbol", "GICS Sector", "GICS Sub-Industry"]].copy()
    df.columns = ["ticker", "sector", "industry"]
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
    df["source"] = "wikipedia"
    return df


def main():
    tickers_df = pd.read_csv(TICKERS_FILE)
    tickers = tickers_df["ticker"].tolist()
    print(f"Fetching sector data for {len(tickers)} tickers via yfinance...")

    yf_df = fetch_yfinance_sectors(tickers)

    # Fill missing with Wikipedia data
    missing = yf_df[yf_df["sector"].isna()]["ticker"].tolist()
    print(f"\n{len(missing)} tickers missing sector - trying Wikipedia fallback...")
    try:
        wiki_df = fetch_wikipedia_sectors()
        wiki_map = wiki_df.set_index("ticker")[["sector", "industry"]]
        for ticker in missing:
            if ticker in wiki_map.index:
                idx = yf_df[yf_df["ticker"] == ticker].index
                yf_df.loc[idx, "sector"] = wiki_map.loc[ticker, "sector"]
                yf_df.loc[idx, "industry"] = wiki_map.loc[ticker, "industry"]
    except Exception as e:
        print(f"  Wikipedia fallback failed: {e}")

    # Encode sector as integer
    yf_df["sector_code"] = yf_df["sector"].map(GICS_SECTORS)

    still_missing = yf_df["sector"].isna().sum()
    print(f"\nSector coverage: {len(yf_df) - still_missing}/{len(yf_df)} tickers")
    print(yf_df["sector"].value_counts(dropna=False).to_string())

    yf_df.to_csv(OUTPUT, index=False)
    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
