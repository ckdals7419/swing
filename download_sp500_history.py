from __future__ import annotations

import argparse
import io
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
import yfinance as yf

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


@dataclass
class MembershipInterval:
    ticker: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp | None


def normalize_ticker(ticker: object) -> str:
    if ticker is None or (isinstance(ticker, float) and pd.isna(ticker)):
        return ""
    # Wikipedia uses dots for share classes (e.g., BRK.B), yfinance expects dashes.
    return str(ticker).strip().upper().replace(".", "-")


def load_wikipedia_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    resp = requests.get(
        WIKI_URL,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        },
        timeout=30,
    )
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    if len(tables) < 2:
        raise RuntimeError("Wikipedia table format changed: expected at least 2 tables.")
    current_df = tables[0].copy()
    changes_df = tables[1].copy()
    return current_df, changes_df


def parse_changes_table(changes_df: pd.DataFrame) -> pd.DataFrame:
    df = changes_df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        flat_cols = []
        for a, b in df.columns:
            flat_cols.append(f"{str(a).strip().lower()}_{str(b).strip().lower()}")
        df.columns = flat_cols
    else:
        df.columns = [str(c).strip().lower() for c in df.columns]

    date_col = next((c for c in df.columns if "effective date" in c or c == "date"), None)
    added_col = next((c for c in df.columns if "added_ticker" in c or c == "added"), None)
    removed_col = next((c for c in df.columns if "removed_ticker" in c or c == "removed"), None)
    if not date_col or not added_col or not removed_col:
        raise RuntimeError("Could not identify date/added/removed columns from changes table.")

    out = df[[date_col, added_col, removed_col]].copy()
    out.columns = ["date", "added", "removed"]
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])
    out["added"] = out["added"].astype(str).str.strip()
    out["removed"] = out["removed"].astype(str).str.strip()
    out = out[out["added"].ne("") | out["removed"].ne("")]
    out["added"] = out["added"].replace({"nan": ""})
    out["removed"] = out["removed"].replace({"nan": ""})
    return out.sort_values("date")


def build_membership_intervals(
    current_symbols: Iterable[str],
    changes_df: pd.DataFrame,
) -> list[MembershipInterval]:
    today = pd.Timestamp(datetime.now(UTC).date())
    # Intervals tracked as [start, end], inclusive dates.
    intervals: dict[str, list[list[pd.Timestamp | None]]] = {
        normalize_ticker(sym): [[pd.NaT, None]] for sym in current_symbols
    }

    # Work backward: if ticker was added on D, before D it was not in index.
    # If ticker was removed on D, before D it was in index.
    for _, row in changes_df.sort_values("date", ascending=False).iterrows():
        d = pd.Timestamp(row["date"]).normalize()

        added = normalize_ticker(row["added"]) if row["added"] else ""
        removed = normalize_ticker(row["removed"]) if row["removed"] else ""

        if added:
            current = intervals.get(added)
            if current:
                # Current open interval must start at added date.
                if pd.isna(current[-1][0]):
                    current[-1][0] = d
                # If there were weird multiple add/remove events, clip safely.
                elif current[-1][0] > d:
                    current[-1][0] = d

        if removed:
            arr = intervals.setdefault(removed, [])
            # Before removal date, ticker must have an interval ending at d - 1.
            arr.append([pd.NaT, d - pd.Timedelta(days=1)])

    # Finalize unknown starts with earliest change date or 1957-03-04 (index expansion date).
    earliest_known = changes_df["date"].min()
    default_start = pd.Timestamp("1957-03-04")
    earliest_start = min(default_start, earliest_known) if pd.notna(earliest_known) else default_start

    result: list[MembershipInterval] = []
    for ticker, raw_intervals in intervals.items():
        for start, end in raw_intervals:
            s = earliest_start if pd.isna(start) else pd.Timestamp(start).normalize()
            e = None if end is None else pd.Timestamp(end).normalize()
            if e is not None and s > e:
                continue
            # Clamp future ends.
            if e is not None and e > today:
                e = today
            result.append(MembershipInterval(ticker=ticker, start_date=s, end_date=e))

    return result


def intervals_to_dataframe(intervals: list[MembershipInterval]) -> pd.DataFrame:
    rows = []
    for itv in intervals:
        rows.append(
            {
                "ticker": itv.ticker,
                "start_date": itv.start_date.date().isoformat(),
                "end_date": "" if itv.end_date is None else itv.end_date.date().isoformat(),
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values(["ticker", "start_date"]).reset_index(drop=True)


def fetch_daily_ohlcv(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        actions=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    if data.empty:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []

    if isinstance(data.columns, pd.MultiIndex):
        level0 = data.columns.get_level_values(0).unique().tolist()
        single_symbol_mode = not any(sym in level0 for sym in tickers)
        if single_symbol_mode:
            # yfinance can return single-ticker frame even when list has one item.
            symbol = tickers[0]
            one = data.copy()
            one.columns = [str(c) for c in one.columns]
            one = one.reset_index().rename(columns={"Date": "date"})
            one["ticker"] = symbol
            frames.append(one)
        else:
            for symbol in tickers:
                if symbol not in data.columns.get_level_values(0):
                    continue
                one = data[symbol].copy()
                one.columns = [str(c) for c in one.columns]
                one = one.reset_index().rename(columns={"Date": "date"})
                one["ticker"] = symbol
                frames.append(one)
    else:
        symbol = tickers[0]
        one = data.copy()
        one.columns = [str(c) for c in one.columns]
        one = one.reset_index().rename(columns={"Date": "date"})
        one["ticker"] = symbol
        frames.append(one)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    ordered_cols = [
        "ticker",
        "date",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
        "Dividends",
        "Stock Splits",
    ]
    for c in ordered_cols:
        if c not in out.columns:
            out[c] = pd.NA
    out = out[ordered_cols].sort_values(["ticker", "date"]).reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download historical S&P 500 constituents and daily OHLCV."
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory to store outputs (default: data)",
    )
    parser.add_argument(
        "--start-date",
        default="1990-01-01",
        help="Daily data start date YYYY-MM-DD (default: 1990-01-01)",
    )
    parser.add_argument(
        "--end-date",
        default=datetime.now(UTC).date().isoformat(),
        help="Daily data end date YYYY-MM-DD, exclusive for yfinance (default: today)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    current_df, changes_df_raw = load_wikipedia_tables()
    current_symbols = (
        current_df["Symbol"].astype(str).map(normalize_ticker).dropna().unique().tolist()
    )
    changes_df = parse_changes_table(changes_df_raw)
    intervals = build_membership_intervals(current_symbols, changes_df)

    intervals_df = intervals_to_dataframe(intervals)
    intervals_path = out_dir / "sp500_membership_intervals.csv"
    intervals_df.to_csv(intervals_path, index=False, encoding="utf-8-sig")

    current_list_path = out_dir / "sp500_current_constituents.csv"
    current_df_out = pd.DataFrame({"ticker": sorted(current_symbols)})
    current_df_out.to_csv(current_list_path, index=False, encoding="utf-8-sig")

    all_tickers = sorted(intervals_df["ticker"].unique().tolist())
    prices_df = fetch_daily_ohlcv(
        tickers=all_tickers,
        start=args.start_date,
        end=args.end_date,
    )
    prices_csv_path = out_dir / "sp500_daily_ohlcv.csv"
    prices_parquet_path = out_dir / "sp500_daily_ohlcv.parquet"
    prices_df.to_csv(prices_csv_path, index=False, encoding="utf-8-sig")
    prices_df.to_parquet(prices_parquet_path, index=False)

    print(f"[OK] Saved current constituents: {current_list_path}")
    print(f"[OK] Saved membership intervals: {intervals_path}")
    print(f"[OK] Saved daily OHLCV CSV: {prices_csv_path}")
    print(f"[OK] Saved daily OHLCV Parquet: {prices_parquet_path}")
    print(f"[INFO] Tickers attempted: {len(all_tickers)}")
    print(f"[INFO] Price rows downloaded: {len(prices_df)}")


if __name__ == "__main__":
    main()
