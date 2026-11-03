"""
Category 11: Macroeconomic Features
Merges macro_daily.csv into the stock feature dataframe by date.
"""

import numpy as np
import pandas as pd


def add_macro_features(df: pd.DataFrame, macro_path: str) -> pd.DataFrame:
    macro = pd.read_csv(macro_path, parse_dates=["date"])
    macro = macro.sort_values("date").reset_index(drop=True)

    # Rate regime: encode as integer (0=low<2, 1=medium 2-5, 2=high>5)
    macro["rate_regime"] = pd.cut(
        macro["fed_funds_rate"],
        bins=[-np.inf, 2, 5, np.inf],
        labels=[0, 1, 2]
    ).astype(float)

    # Rate momentum (3-month change)
    macro["rate_momentum_3m"] = macro["fed_funds_rate"].diff(3)

    # CPI acceleration (change in YoY)
    macro["cpi_acceleration"] = macro["cpi_yoy"].diff(3)

    # Yield curve slope change
    macro["yield_curve_change_20d"] = macro["yield_curve_10y2y"].diff(20)

    # HY spread regime (data only from 2023, so will be NaN for earlier)
    macro["credit_spread_change_20d"] = macro["hy_credit_spread"].diff(20) if "hy_credit_spread" in macro.columns else np.nan

    # DXY momentum
    macro["dxy_change_20d"] = macro["dxy"].pct_change(20) if "dxy" in macro.columns else np.nan

    # Select columns to merge
    merge_cols = [
        "date", "fed_funds_rate", "rate_regime", "rate_momentum_3m",
        "rate_change_direction", "yield_curve_10y2y", "yield_curve_10y3m",
        "yield_curve_inverted", "yield_curve_change_20d",
        "cpi_yoy", "cpi_acceleration", "unemployment_rate",
        "hy_credit_spread", "credit_spread_change_20d",
        "dxy", "dxy_change_20d",
        "oil_price", "oil_change_20d",
        "gold_price", "gold_change_20d",
    ]
    merge_cols = [c for c in merge_cols if c in macro.columns]

    df = df.merge(macro[merge_cols], on="date", how="left")

    return df
