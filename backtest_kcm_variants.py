"""
KCM variant backtest with:
  - Synthetic TQQQ (NDX 3x) pre-2010 spliced to real TQQQ
  - Transaction cost 9 bps/side on |Δw|
  - KR overseas capital-gains tax 22% on annual net realized gains
  - Variants: BASE, H3, SMA ensemble (#3), VIX term (#4), MOC fill (#5)

Usage:
  python backtest_kcm_variants.py
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

from notify_kcm_telegram import BUF, RSI_DIP, kcm_weights, rsi_sma, vt_scale

warnings.filterwarnings("ignore")

COST_BPS = 9.0
TAX_RATE = 0.22
TQQQ_LEV = 3.0
DAILY_DRAG = 0.0001  # ~2.5%/yr expense+financing on synth
START = "2000-01-01"  # dot-com era onwards (need SMA200 warm-up before)


# ── Data ─────────────────────────────────────────────────────────────────────

def _dl(ticker: str, start: str = "1990-01-01") -> pd.DataFrame:
    d = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    return d


def build_dataset() -> pd.DataFrame:
    print("Downloading market data...")
    ndx = _dl("^NDX")
    qqq = _dl("QQQ")
    tqqq = _dl("TQQQ")
    vix = _dl("^VIX")
    vix3m = _dl("^VIX3M")

    # Signal underlying: QQQ from inception, else NDX scaled
    sig_close = ndx["Close"].copy().astype(float)
    sig_open = ndx["Open"].copy().astype(float)
    # splice QQQ returns onto NDX level from QQQ start (keep NDX path continuity for SMA)
    # Better: use NDX for entire signal history (cleaner for pre-QQQ), QQQ optional.
    # KCM live uses QQQ; NDX≈QQQ for regime. Use NDX full history for consistency.
    underlying_c = sig_close
    underlying_o = sig_open

    # Synthetic TQQQ from NDX daily returns
    ndx_ret = underlying_c.pct_change().fillna(0.0)
    ndx_oc = (underlying_o / underlying_c.shift(1)).fillna(1.0)

    synth_c = pd.Series(index=underlying_c.index, dtype=float)
    synth_c.iloc[0] = 100.0
    for i in range(1, len(synth_c)):
        synth_c.iloc[i] = synth_c.iloc[i - 1] * (1 + TQQQ_LEV * float(ndx_ret.iloc[i]) - DAILY_DRAG)

    synth_o = synth_c.shift(1) * (1 + TQQQ_LEV * (ndx_oc - 1.0))
    synth_o.iloc[0] = synth_c.iloc[0]

    # Splice real TQQQ from first available date (scale to match synth level)
    real_c = tqqq["Close"].astype(float)
    real_o = tqqq["Open"].astype(float)
    splice = real_c.index[0]
    # scale real series so first real close == synth close that day
    scale = float(synth_c.loc[splice]) / float(real_c.loc[splice])
    real_c_s = real_c * scale
    real_o_s = real_o * scale

    veh_c = synth_c.copy()
    veh_o = synth_o.copy()
    common = real_c_s.index.intersection(veh_c.index)
    veh_c.loc[common] = real_c_s.loc[common]
    veh_o.loc[common] = real_o_s.loc[common]

    vx = vix["Close"].astype(float).reindex(underlying_c.index).ffill()
    vx3 = vix3m["Close"].astype(float).reindex(underlying_c.index).ffill()

    df = pd.DataFrame({
        "sig_close": underlying_c,
        "sig_open": underlying_o,
        "veh_close": veh_c,
        "veh_open": veh_o,
        "vix": vx,
        "vix3m": vx3,
        "qqq_close": qqq["Close"].astype(float).reindex(underlying_c.index),
    })
    df = df.loc[START:].dropna(subset=["sig_close", "veh_close", "veh_open", "vix"])
    print(f"  Signal: NDX  |  Vehicle: synth TQQQ → real from {splice.date()}")
    print(f"  Range: {df.index[0].date()} → {df.index[-1].date()}  ({len(df)} days)")
    print(f"  VIX3M available from: {vx3.first_valid_index().date() if vx3.notna().any() else 'N/A'}")
    return df


# ── Signal variants ──────────────────────────────────────────────────────────

def w_base(df: pd.DataFrame, *, vt_target: float | None = None) -> pd.Series:
    return kcm_weights(df["sig_close"], df["vix"], vt_target=vt_target)[0]


def w_h_confirm(df: pd.DataFrame, days: int = 3) -> pd.Series:
    close, vix = df["sig_close"], df["vix"]
    vs200 = (close / close.rolling(200).mean() - 1).fillna(0)
    vs50 = (close / close.rolling(50).mean() - 1).fillna(0)
    rsi = rsi_sma(close).ffill().fillna(50)
    vx = vix.reindex(close.index, method="ffill")
    vx10 = vx.rolling(10).mean()
    n = len(close)
    w = np.zeros(n)
    inp = bool(vs200.iloc[0] >= 0)
    streak = 0
    w[0] = 1.0 if inp else 0.0
    for i in range(1, n):
        has = pd.notna(vx.iloc[i]) and pd.notna(vx10.iloc[i])
        vr = bool(has and vx.iloc[i] > 35 and vx.iloc[i] < vx10.iloc[i])
        dip = float(rsi.iloc[i]) < RSI_DIP
        entry = vs200.iloc[i] >= BUF or vs50.iloc[i] >= 0
        if inp:
            if vs200.iloc[i] < -BUF and vs50.iloc[i] < 0:
                inp = False
                streak = 0
                w[i] = 1.0 if (vr or dip) else 0.0
            else:
                w[i] = 1.0
        else:
            streak = streak + 1 if entry else 0
            if streak >= days:
                inp = True
                w[i] = 1.0
            elif vr or dip:
                w[i] = 1.0
            else:
                w[i] = 0.0
    return pd.Series(w, index=close.index, name="w")


def w_sma_ensemble(df: pd.DataFrame, windows=(100, 150, 200, 250)) -> pd.Series:
    """#3 continuous weight = fraction of SMAs price is above."""
    close = df["sig_close"]
    votes = []
    for n in windows:
        sma = close.rolling(n).mean()
        votes.append((close >= sma).astype(float))
    w = sum(votes) / len(votes)
    return w.fillna(0.0).clip(0, 1)


def w_vix_term(df: pd.DataFrame, base: pd.Series, cut: float = 1.0,
               reduce_to: float = 0.0) -> pd.Series:
    """#4: when VIX/VIX3M >= cut (backwardation), scale weight to reduce_to.
    Before VIX3M exists, leave base unchanged.
    """
    ratio = (df["vix"] / df["vix3m"]).replace([np.inf, -np.inf], np.nan)
    out = base.copy().astype(float)
    mask = ratio.notna() & (ratio >= cut)
    out.loc[mask] = reduce_to
    return out.clip(0, 1)


def w_vix_term_scale(df: pd.DataFrame, base: pd.Series, cut: float = 1.0) -> pd.Series:
    """Softer #4: in backwardation, halve exposure instead of flat cash."""
    ratio = (df["vix"] / df["vix3m"]).replace([np.inf, -np.inf], np.nan)
    out = base.copy().astype(float)
    mask = ratio.notna() & (ratio >= cut)
    out.loc[mask] = out.loc[mask] * 0.5
    return out.clip(0, 1)


# ── Backtest engine ──────────────────────────────────────────────────────────

@dataclass
class BTResult:
    label: str
    nav: pd.Series
    cagr: float
    total: float
    mdd: float
    peak: pd.Timestamp
    trough: pd.Timestamp
    calmar: float
    n_trades: int
    tax_paid: float
    fill: str


def backtest(
    df: pd.DataFrame,
    weights: pd.Series,
    label: str,
    fill: str = "next_open",  # or "same_close"
    cost_bps: float = COST_BPS,
    tax_rate: float = TAX_RATE,
    initial: float = 1.0,
) -> BTResult:
    """
    fill=next_open: signal T close → apply from T+1 open (open→next open / last close)
    fill=same_close: signal T close → apply at T close (close→close with today's w)
    Tax: track lot cost basis; on weight decrease realize PnL; settle 22% on
         positive net realized gains at each calendar year-end (exemption ignored).
    """
    w = weights.reindex(df.index).ffill().fillna(0.0)
    close = df["veh_close"]
    open_ = df["veh_open"]
    cost_rate = cost_bps / 10_000.0

    dates = df.index
    nav = initial
    navs = []
    prev_w = 0.0
    basis = 0.0          # cost basis of current position (absolute $)
    position_val = 0.0   # mark of current equity sleeve
    year_realized = 0.0
    tax_paid_total = 0.0
    n_trades = 0
    cur_year = dates[0].year

    def settle_tax(y_realized: float) -> float:
        if y_realized > 0:
            return y_realized * tax_rate
        return 0.0

    for i, d in enumerate(dates):
        # year-end tax settlement at first bar of new year (on prior year)
        if d.year != cur_year:
            tax = settle_tax(year_realized)
            if tax > 0:
                nav -= tax
                tax_paid_total += tax
            year_realized = 0.0
            cur_year = d.year

        if fill == "next_open":
            # Signal at T close → trade T+1 open → earn open[T+1]→open[T+2]
            ww = float(w.iloc[i - 1]) if i > 0 else 0.0
            if i < len(dates) - 1:
                r = float(open_.iloc[i + 1] / open_.iloc[i] - 1)
            else:
                r = float(close.iloc[i] / open_.iloc[i] - 1) if float(open_.iloc[i]) else 0.0
        else:
            # MOC (no look-ahead): signal at T close → trade T close
            # → earn close[T]→close[T+1] with weight w[T].
            # On bar i we apply w[i-1] to return close[i-1]→close[i].
            ww = float(w.iloc[i - 1]) if i > 0 else 0.0
            if i == 0:
                r = 0.0
            else:
                r = float(close.iloc[i] / close.iloc[i - 1] - 1)

        # trade at start of bar
        dw = ww - prev_w
        if abs(dw) > 1e-12:
            n_trades += 1
            tc = abs(dw) * nav * cost_rate
            nav -= tc
            if dw > 0:
                # buy: add to basis at current NAV equity mark (~nav)
                # approximate entry price unit = 1 at trade time; track $ basis
                buy_notional = dw * nav  # after cost approx
                basis += buy_notional
            else:
                # sell: realize PnL on sold fraction of position
                if prev_w > 1e-12:
                    sold_frac = (-dw) / prev_w
                    sold_basis = basis * sold_frac
                    sold_value = (-dw) * nav
                    year_realized += sold_value - sold_basis
                    basis -= sold_basis
                if ww <= 1e-12:
                    basis = 0.0

        # mark-to-market
        nav = nav * (1.0 + ww * r)
        # scale basis with remaining position notional conceptually stays as cost;
        # basis is cost, not marked — OK
        if ww <= 1e-12:
            basis = 0.0
        prev_w = ww
        navs.append(nav)

    # final year tax
    tax = settle_tax(year_realized)
    if tax > 0 and navs:
        navs[-1] -= tax
        tax_paid_total += tax
        # also reduce if sold to flat at end — already in year_realized

    s = pd.Series(navs, index=dates, name="nav")
    # clean nonpositive
    s = s.clip(lower=1e-9)
    total = float(s.iloc[-1] / s.iloc[0] - 1)
    nyrs = max((s.index[-1] - s.index[0]).days / 365.25, 1e-9)
    cagr = float((s.iloc[-1] / s.iloc[0]) ** (1 / nyrs) - 1)
    peak = s.cummax()
    dd = s / peak - 1
    mdd = float(dd.min())
    trough = dd.idxmin()
    peak_d = s.loc[:trough][s.loc[:trough] == peak.loc[trough]].index[-1]
    calmar = cagr / abs(mdd) if mdd != 0 else 0.0
    return BTResult(label, s, cagr, total, mdd, peak_d, trough, calmar,
                    n_trades, tax_paid_total / initial, fill)


def period_metrics(nav: pd.Series, start: str, end: str | None = None) -> dict:
    s = nav.loc[start:end] if end else nav.loc[start:]
    if len(s) < 20:
        return {}
    total = float(s.iloc[-1] / s.iloc[0] - 1)
    nyrs = max((s.index[-1] - s.index[0]).days / 365.25, 1e-9)
    cagr = float((s.iloc[-1] / s.iloc[0]) ** (1 / nyrs) - 1)
    mdd = float((s / s.cummax() - 1).min())
    return dict(cagr=cagr, total=total, mdd=mdd, start=s.index[0].date(), end=s.index[-1].date())


def yearly_table(results: list[BTResult]) -> None:
    years = range(results[0].nav.index[0].year, results[0].nav.index[-1].year + 1)
    labels = [r.label for r in results]
    print(f"\n{'Year':<6}" + "".join(f"{l[:11]:>12}" for l in labels))
    for y in years:
        row = f"{y:<6}"
        for r in results:
            s = r.nav.loc[f"{y}-01-01":f"{y}-12-31"]
            if len(s) < 5:
                row += f"{'n/a':>12}"
            else:
                ret = s.iloc[-1] / s.iloc[0] - 1
                row += f"{ret:>+11.1%} "
        print(row)


def main() -> None:
    df = build_dataset()

    print("\nBuilding signals...")
    base = w_base(df)
    base_vt = (base * vt_scale(df["sig_close"])).clip(0, 1)  # optional overlay
    h3 = w_h_confirm(df, 3)
    ens = w_sma_ensemble(df)
    vix_flat = w_vix_term(df, base, cut=1.0, reduce_to=0.0)
    vix_half = w_vix_term_scale(df, base, cut=1.0)
    # also VIX term on H3
    h3_vix = w_vix_term(df, h3, cut=1.0, reduce_to=0.0)

    bah = pd.Series(1.0, index=df.index)

    configs = [
        ("BASE KCM", base, "next_open"),
        ("BASE+VT25 (옵션)", base_vt, "next_open"),
        ("H3", h3, "next_open"),
        ("#3 SMA ensemble", ens, "next_open"),
        ("#4 VIX term flat", vix_flat, "next_open"),
        ("#4 VIX term 50%", vix_half, "next_open"),
        ("#5 BASE MOC", base, "same_close"),
        ("#5 H3 MOC", h3, "same_close"),
        ("H3+#4 VIX flat", h3_vix, "next_open"),
        ("TQQQ B&H", bah, "next_open"),
    ]

    print("\nRunning backtests (cost=9bps, tax=22% annual realized)...")
    results: list[BTResult] = []
    for label, w, fill in configs:
        r = backtest(df, w, label, fill=fill)
        results.append(r)

    # Summary
    print("\n" + "=" * 110)
    print(f"{'Strategy':<20} {'Fill':<10} {'CAGR':>8} {'Total':>10} {'MDD':>8} {'Calmar':>7} "
          f"{'Trades':>7} {'Tax/Init':>9}  MDD period")
    print("-" * 110)
    for r in results:
        print(f"{r.label:<20} {r.fill:<10} {r.cagr:>+7.1%} {r.total:>+9.0%} {r.mdd:>7.1%} "
              f"{r.calmar:>6.2f} {r.n_trades:>7} {r.tax_paid:>8.1f}x  "
              f"{r.peak.date()} → {r.trough.date()}")

    # Subperiods
    periods = [
        ("닷컴 2000-2002", "2000-01-01", "2002-12-31"),
        ("회복 2003-2007", "2003-01-01", "2007-12-31"),
        ("GFC 2008-2009", "2008-01-01", "2009-12-31"),
        ("TQQQ실물 2010+", "2010-02-11", None),
        ("코로나 2020", "2020-01-01", "2020-12-31"),
        ("전체", START, None),
    ]
    print("\n" + "=" * 110)
    print("구간별 CAGR / MDD")
    for pname, ps, pe in periods:
        print(f"\n[{pname}]")
        print(f"  {'Strategy':<20} {'CAGR':>8} {'MDD':>8} {'Total':>10}")
        for r in results:
            m = period_metrics(r.nav, ps, pe)
            if not m:
                continue
            print(f"  {r.label:<20} {m['cagr']:>+7.1%} {m['mdd']:>7.1%} {m['total']:>+9.0%}")

    # Yearly returns for key set
    key = [r for r in results if r.label in {
        "BASE KCM", "H3", "#3 SMA ensemble", "#4 VIX term flat",
        "#5 BASE MOC", "TQQQ B&H",
    }]
    print("\n연도별 수익률 (세후·비용후)")
    yearly_table(key)

    # VIX3M coverage note for #4 in early years
    first3m = df["vix3m"].first_valid_index()
    print(f"\n참고: VIX3M은 {first3m.date() if first3m else 'N/A'}부터 → "
          f"#4는 그 이전에는 BASE와 동일")
    print(f"세금: 연말 순실현이익 × {TAX_RATE:.0%} (공제 무시, 큰 자본 가정)")
    print(f"비용: 편도 {COST_BPS:.0f}bps × |Δweight|")


if __name__ == "__main__":
    main()
