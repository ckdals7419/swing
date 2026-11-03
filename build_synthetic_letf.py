"""
Build synthetic QLD (2x) and TQQQ (3x) from QQQ daily returns.

Leveraged ETF daily return model:
  r_letf = L × r_qqq − daily_cost

Where daily_cost accounts for:
  1. Expense ratio (ER): ~0.95% annualized for both QLD/TQQQ
  2. Leverage borrowing cost: (L-1) × short-term rate (fed funds)
  3. Swap/financing spread: ~0.40% annualized on borrowed notional

  daily_cost = (ER + (L-1) × (fed_funds + swap_spread)) / 252

This captures the daily compounding, volatility decay, and financing
drag that make leveraged ETFs diverge from simple L× of the index.

Outputs:
  data/synthetic_letf.parquet
    Columns: date, qqq_close, synth_qld_close, synth_tqqq_close,
             real_qld_close, real_tqqq_close
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Parameters ───────────────────────────────────────────────────────────────
QLD_LEVERAGE  = 2
TQQQ_LEVERAGE = 3
EXPENSE_RATIO = 0.0095       # 0.95% annualized
SWAP_SPREAD   = 0.0049       # 0.49% annualized on borrowed notional (calibrated)


def load_qqq() -> pd.DataFrame:
    df = pd.read_parquet("data/qqq_full.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    return df[["Open", "Close"]].rename(columns={"Open": "qqq_open", "Close": "qqq_close"})


def load_fed_funds() -> pd.Series:
    macro = pd.read_csv("data/macro_daily.csv", parse_dates=["date"])
    macro = macro.sort_values("date").set_index("date")
    ff = macro["fed_funds_rate"].ffill().fillna(5.0) / 100.0  # convert % to decimal
    return ff


def load_real_prices() -> pd.DataFrame:
    univ = pd.read_parquet("data/qld_universe.parquet")
    univ["date"] = pd.to_datetime(univ["date"])

    qld = (univ[univ["ticker"] == "QLD"].sort_values("date")
           .set_index("date")[["Close"]].rename(columns={"Close": "real_qld_close"}))
    tqqq = (univ[univ["ticker"] == "TQQQ"].sort_values("date")
            .set_index("date")[["Close"]].rename(columns={"Close": "real_tqqq_close"}))

    return qld.join(tqqq, how="outer")


def build_synthetic(qqq: pd.DataFrame, ff: pd.Series,
                    leverage: int, er: float, spread: float) -> pd.Series:
    """Build synthetic leveraged ETF price series from QQQ returns."""
    qqq_close = qqq["qqq_close"]
    daily_ret = qqq_close.pct_change()

    # Align fed funds to QQQ dates
    ff_aligned = ff.reindex(qqq.index).ffill().fillna(0.05)

    # Daily cost: expense ratio + borrowing cost on leveraged portion
    daily_cost = (er + (leverage - 1) * (ff_aligned + spread)) / 252.0

    # Leveraged daily return
    letf_ret = leverage * daily_ret - daily_cost

    # Build price series (start at 100)
    letf_price = (1 + letf_ret).cumprod() * 100
    letf_price.iloc[0] = 100.0

    return letf_price


def main():
    print("Loading QQQ prices...")
    qqq = load_qqq()
    print(f"  QQQ: {qqq.index[0].date()} -> {qqq.index[-1].date()} ({len(qqq):,} days)")

    print("Loading fed funds rate...")
    ff = load_fed_funds()
    print(f"  Fed funds: {ff.index[0].date()} -> {ff.index[-1].date()}")

    print("Building synthetic QLD (2x)...")
    synth_qld = build_synthetic(qqq, ff, QLD_LEVERAGE, EXPENSE_RATIO, SWAP_SPREAD)

    print("Building synthetic TQQQ (3x)...")
    synth_tqqq = build_synthetic(qqq, ff, TQQQ_LEVERAGE, EXPENSE_RATIO, SWAP_SPREAD)

    # Load real prices for validation
    print("Loading real QLD/TQQQ prices...")
    real = load_real_prices()

    # Combine into single dataframe
    result = pd.DataFrame({
        "qqq_close":         qqq["qqq_close"],
        "qqq_open":          qqq["qqq_open"],
        "synth_qld_close":   synth_qld,
        "synth_tqqq_close":  synth_tqqq,
    }).join(real, how="left")

    result.index.name = "date"
    result = result.reset_index()

    out = "data/synthetic_letf.parquet"
    result.to_parquet(out, index=False)
    print(f"\nSaved: {out}  ({len(result):,} rows)")
    print(f"  Date range: {result['date'].min().date()} -> {result['date'].max().date()}")

    # ── Validation: compare synthetic vs real on overlap period ───────────
    print("\n" + "=" * 80)
    print("  VALIDATION: Synthetic vs Real (overlap period)")
    print("=" * 80)

    for name, synth_col, real_col, lev in [
        ("QLD  (2x)", "synth_qld_close",  "real_qld_close",  2),
        ("TQQQ (3x)", "synth_tqqq_close", "real_tqqq_close", 3),
    ]:
        overlap = result.dropna(subset=[synth_col, real_col]).copy()
        if len(overlap) == 0:
            print(f"\n  {name}: no overlapping data")
            continue

        # Normalize both to start at 100 on first overlap date
        s = overlap[synth_col].values
        r = overlap[real_col].values
        s_norm = s / s[0] * 100
        r_norm = r / r[0] * 100

        # Compute daily return correlation
        s_ret = np.diff(s_norm) / s_norm[:-1]
        r_ret = np.diff(r_norm) / r_norm[:-1]
        corr = np.corrcoef(s_ret, r_ret)[0, 1]

        # Terminal value comparison
        s_final = s_norm[-1]
        r_final = r_norm[-1]
        drift = s_final / r_final - 1

        # Tracking error (annualized)
        diff_ret = s_ret - r_ret
        te = np.std(diff_ret) * np.sqrt(252) * 100  # in bps

        start_d = overlap["date"].iloc[0].date()
        end_d   = overlap["date"].iloc[-1].date()

        print(f"\n  {name}  {start_d} -> {end_d}  ({len(overlap):,} days)")
        print(f"    Daily return correlation:  {corr:.6f}")
        print(f"    Synthetic final NAV:       {s_final:.1f}")
        print(f"    Real final NAV:            {r_final:.1f}")
        print(f"    Terminal drift:            {drift:+.2%}")
        print(f"    Tracking error (ann.):     {te:.1f} bps")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
