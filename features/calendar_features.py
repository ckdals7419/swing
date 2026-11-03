"""
Category 7 & 8: Calendar / Seasonality + Presidential Cycle Features
"""
import numpy as np
import pandas as pd

# US presidential election years 1988–2028
ELECTION_YEARS = set(range(1988, 2029, 4))  # 1988,1992,...,2024,2028

# Incumbent party mapping (0 = Democrat, 1 = Republican) by year of inauguration
# (party that won the election in Nov of election year)
PARTY_WINS = {
    1988: 1,   # Bush Sr (R)
    1992: 0,   # Clinton (D)
    1996: 0,   # Clinton (D)
    2000: 1,   # Bush Jr (R)
    2004: 1,   # Bush Jr (R)
    2008: 0,   # Obama (D)
    2012: 0,   # Obama (D)
    2016: 1,   # Trump (R)
    2020: 0,   # Biden (D)
    2024: 1,   # Trump (R)
}


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    date = df["date"]

    df["day_of_week"] = date.dt.dayofweek          # 0=Mon, 4=Fri
    df["month"] = date.dt.month
    df["quarter"] = date.dt.quarter
    df["week_of_year"] = date.dt.isocalendar().week.astype(int)
    df["year"] = date.dt.year

    # Trading day position within month
    df["trading_day_of_month"] = df.groupby(
        [df["year"], df["month"]]
    ).cumcount() + 1

    # Month-end / month-start flags (trading-day based)
    month_size = df.groupby([df["year"], df["month"]])["trading_day_of_month"].transform("max")
    df["is_month_start"] = (df["trading_day_of_month"] <= 3).astype(int)
    df["is_month_end"] = (df["trading_day_of_month"] >= month_size - 2).astype(int)

    # Quarter-end flag (last 5 trading days of quarter)
    quarter_size = df.groupby([df["year"], df["quarter"]])["trading_day_of_month"].transform("max")
    df["trading_day_of_quarter"] = df.groupby(
        [df["year"], df["quarter"]]
    ).cumcount() + 1
    df["is_quarter_end"] = (
        df["trading_day_of_quarter"] >= quarter_size - 4
    ).astype(int)

    # Days to quarter end
    df["days_to_quarter_end"] = quarter_size - df["trading_day_of_quarter"]

    # Year-end flag (December)
    df["is_year_end"] = (df["month"] == 12).astype(int)
    df["is_january"] = (df["month"] == 1).astype(int)

    # Options expiration week: 3rd Friday of month
    df["is_opex_week"] = _opex_week_flag(date)

    # Sell-in-May signal: May–Oct vs Nov–Apr
    df["sell_in_may"] = date.dt.month.isin([5, 6, 7, 8, 9, 10]).astype(int)

    # ── Presidential Cycle (Category 8) ──────────────────────────────────────
    year = date.dt.year

    # Cycle phase: 0=election yr, 1=post-election, 2=midterm, 3=pre-election
    df["election_cycle_phase"] = year % 4  # maps naturally: election yrs ≡ 0 mod 4 from 1988

    df["is_election_year"] = year.isin(ELECTION_YEARS).astype(int)
    df["is_midterm_year"] = year.isin({y + 2 for y in ELECTION_YEARS}).astype(int)
    df["is_pre_election_year"] = year.isin({y - 1 for y in ELECTION_YEARS}).astype(int)
    df["is_post_election_year"] = year.isin({y + 1 for y in ELECTION_YEARS}).astype(int)

    # Months to next November election
    df["months_to_election"] = _months_to_election(date)

    # Inauguration quarter: Q1 of post-election year
    df["is_inauguration_quarter"] = (
        df["is_post_election_year"] & (df["quarter"] == 1)
    ).astype(int)

    # Lame duck: Nov–Jan of election → post-election transition
    df["is_lame_duck"] = (
        (df["is_election_year"] & (df["month"] >= 11)) |
        (df["is_post_election_year"] & (df["month"] == 1))
    ).astype(int)

    # Party in power during trading day (party that holds presidency)
    df["party_in_power"] = _party_in_power(date)

    return df


def _opex_week_flag(date: pd.Series) -> pd.Series:
    """1 if the week contains the 3rd Friday of the month."""
    # 3rd Friday = first day of month + offset to first Friday + 14 days
    first_of_month = date.dt.to_period("M").dt.to_timestamp()
    dow_first = first_of_month.dt.dayofweek  # 0=Mon
    days_to_first_fri = (4 - dow_first) % 7
    third_friday = first_of_month + pd.to_timedelta(days_to_first_fri + 14, unit="D")
    # Flag: current date is within Mon–Fri of the week containing 3rd Friday
    week_start = third_friday - pd.to_timedelta(third_friday.dt.dayofweek, unit="D")
    week_end = week_start + pd.Timedelta(days=4)
    return ((date >= week_start) & (date <= week_end)).astype(int)


def _months_to_election(date: pd.Series) -> pd.Series:
    """Months until the next presidential election (November of election year)."""
    def compute(d):
        y = d.year
        # Find the next election year >= current year
        next_ey = next((ey for ey in sorted(ELECTION_YEARS) if ey > y or (ey == y and d.month <= 11)), None)
        if next_ey is None:
            return np.nan
        election_date = pd.Timestamp(next_ey, 11, 1)
        return max(0, (election_date.year - d.year) * 12 + (election_date.month - d.month))
    return date.map(compute)


def _party_in_power(date: pd.Series) -> pd.Series:
    """0=Democrat, 1=Republican holding presidency on this date."""
    # Inaugurations are Jan 20 of post-election year
    transitions = {}
    for ey, party in PARTY_WINS.items():
        inaug = pd.Timestamp(ey + 1, 1, 20)
        transitions[inaug] = party

    sorted_dates = sorted(transitions.keys())

    def lookup(d):
        # Last inauguration before or on date d
        relevant = [t for t in sorted_dates if t <= d]
        if not relevant:
            return np.nan
        return transitions[max(relevant)]

    return date.map(lookup)
