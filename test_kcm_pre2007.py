"""
KCM backtest on full synthetic NASDAQ data prior to 2007.
  - ^IXIC (1971-1985) + ^NDX (1985-2006) -> synthetic 2x QLD, 3x TQQQ
  - Gold: GC=F futures (1971+) + GLD splice (2004+)
  - 4 strategies: KCM QLD, KCM TQQQ, KCM QLD+GLD, KCM TQQQ+GLD
"""
import warnings, numpy as np, pandas as pd, yfinance as yf
warnings.filterwarnings("ignore")
from strategy_qld import backtest, _nav_metrics, COST_BPS

END_DATE = "2006-12-31"
DAILY_COST = 0.0001  # ~2.5% annual (expense + financing)

# ── Download NASDAQ indices ──────────────────────────────────────────────────
print("Downloading NASDAQ indices...")
ndx = yf.download("^NDX", start="1985-01-01", end="2007-01-01", progress=False)
ndx.columns = ndx.columns.get_level_values(0)
ixic = yf.download("^IXIC", start="1971-01-01", end="2007-01-01", progress=False)
ixic.columns = ixic.columns.get_level_values(0)
print(f"  ^NDX:  {ndx.index[0].date()} -> {ndx.index[-1].date()} ({len(ndx):,} days)")
print(f"  ^IXIC: {ixic.index[0].date()} -> {ixic.index[-1].date()} ({len(ixic):,} days)")

# ── Splice: IXIC before NDX start, NDX after ────────────────────────────────
ndx_close = ndx["Close"].squeeze()
ndx_open  = ndx["Open"].squeeze()
ixic_close = ixic["Close"].squeeze()
ixic_open  = ixic["Open"].squeeze()

ixic_ret = ixic_close.pct_change()
ndx_ret  = ndx_close.pct_change()
ndx_start = ndx.index[0]

combined_ret = ixic_ret.copy()
combined_ret.loc[ndx_start:] = ndx_ret.loc[ndx_start:]
combined_ret = combined_ret.fillna(0)

ixic_oc = ixic_open / ixic_close.shift(1)
ndx_oc  = ndx_open / ndx_close.shift(1)
combined_oc = ixic_oc.copy()
combined_oc.loc[ndx_start:] = ndx_oc.loc[ndx_start:]
combined_oc = combined_oc.fillna(1.0)

# ── Build underlying close (for SMA features) ───────────────────────────────
underlying = pd.Series(index=combined_ret.index, dtype=float)
underlying.iloc[0] = 100.0
for i in range(1, len(underlying)):
    underlying.iloc[i] = underlying.iloc[i-1] * (1 + combined_ret.iloc[i])

# ── Build synthetic QLD (2x) and TQQQ (3x) ──────────────────────────────────
qld_close = pd.Series(index=combined_ret.index, dtype=float)
qld_close.iloc[0] = 100.0
tqqq_close = pd.Series(index=combined_ret.index, dtype=float)
tqqq_close.iloc[0] = 100.0
for i in range(1, len(combined_ret)):
    r = combined_ret.iloc[i]
    qld_close.iloc[i] = qld_close.iloc[i-1] * (1 + 2.0 * r - DAILY_COST)
    tqqq_close.iloc[i] = tqqq_close.iloc[i-1] * (1 + 3.0 * r - DAILY_COST)

qld_open = qld_close.shift(1) * (1 + 2.0 * (combined_oc - 1))
qld_open.iloc[0] = qld_close.iloc[0]
tqqq_open = tqqq_close.shift(1) * (1 + 3.0 * (combined_oc - 1))
tqqq_open.iloc[0] = tqqq_close.iloc[0]

print(f"\nSynthetic QLD:  start={qld_close.iloc[0]:.0f}, end={qld_close.iloc[-1]:.0f}")
print(f"Synthetic TQQQ: start={tqqq_close.iloc[0]:.0f}, end={tqqq_close.iloc[-1]:.0f}")

# ── Download Gold ────────────────────────────────────────────────────────────
print("\nDownloading gold price (GC=F)...")
gold_raw = yf.download("GC=F", start="1970-01-01", end="2007-06-01", progress=False)
gold_raw.columns = gold_raw.columns.get_level_values(0)
gold_close_raw = gold_raw["Close"].squeeze()
gold_open_raw  = gold_raw["Open"].squeeze()
print(f"  GC=F: {gold_close_raw.index[0].date()} -> {gold_close_raw.index[-1].date()} ({len(gold_close_raw):,} days)")
print(f"  Price range: ${gold_close_raw.min():.0f} - ${gold_close_raw.max():.0f}")

# Align gold to underlying index
g_close = gold_close_raw.reindex(combined_ret.index, method="ffill")
g_open  = gold_open_raw.reindex(combined_ret.index, method="ffill")
# Fill early NaN with first available value
first_gold = g_close.first_valid_index()
if first_gold:
    g_close = g_close.ffill().bfill()
    g_open  = g_open.ffill().bfill()

# ── Build features ───────────────────────────────────────────────────────────
close = underlying
sma50  = close.rolling(50, min_periods=50).mean()
sma200 = close.rolling(200, min_periods=200).mean()

delta = close.diff()
gain  = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
loss  = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
rs    = gain / loss.replace(0, np.nan)
rsi14 = 100 - 100 / (1 + rs)

vix_df = pd.read_csv("data/vix_daily.csv", parse_dates=["Date"])
vix_raw = vix_df.set_index("Date")["vix_close"]
vix = vix_raw.reindex(close.index, method="ffill")  # NaN before 1990
vix_sma10 = vix.rolling(10).mean()

df = pd.DataFrame(index=close.index)
df.index.name = "date"
df["qqq_vs_sma200"] = close / sma200 - 1
df["qqq_vs_sma50"]  = close / sma50 - 1
df["rsi_14"]         = rsi14
df["vix_close"]      = vix
df["realized_vol_20d"] = np.log(close / close.shift(1)).rolling(20).std() * np.sqrt(252)
df["bil_return_1d"]  = 0.0325 / 252.0
df["qld_open"]       = qld_open
df["qld_close"]      = qld_close
df["tqqq_open"]      = tqqq_open
df["tqqq_close"]     = tqqq_close
df["gld_open"]       = g_open
df["gld_close"]      = g_close
df = df.dropna(subset=["qqq_vs_sma200"])
df = df[df.index <= END_DATE]

years = (df.index[-1] - df.index[0]).days / 365.25
print(f"\nDataset: {df.index[0].date()} -> {df.index[-1].date()} ({len(df):,} days, {years:.0f} years)")


# ── KCM weight function ─────────────────────────────────────────────────────
def w_kcm(df):
    vs200 = df["qqq_vs_sma200"].fillna(0)
    vs50  = df["qqq_vs_sma50"].fillna(0)
    vix   = df["vix_close"].ffill().fillna(20)
    vix_sma10 = vix.rolling(10).mean()
    rsi   = df["rsi_14"].ffill().fillna(50)

    n = len(df)
    w = np.zeros(n)
    in_position = vs200.iloc[0] >= 0
    w[0] = 1.0 if in_position else 0.0

    for i in range(1, n):
        has_vix = not np.isnan(vix.iloc[i]) and pd.notna(vix_sma10.iloc[i])
        vix_reentry = has_vix and vix.iloc[i] > 35 and vix.iloc[i] < vix_sma10.iloc[i]
        if in_position:
            if vs200.iloc[i] < -0.01 and vs50.iloc[i] < 0:
                in_position = False
                if vix_reentry or rsi.iloc[i] < 30: w[i] = 1.0
                else: w[i] = 0.0
            else:
                w[i] = 1.0
        else:
            if vs200.iloc[i] >= 0.01: in_position = True; w[i] = 1.0
            elif vs50.iloc[i] >= 0: in_position = True; w[i] = 1.0
            elif vix_reentry or rsi.iloc[i] < 30: w[i] = 1.0
            else: w[i] = 0.0

    return pd.Series(w, index=df.index)


# ── Backtest with GLD hedge ─────────────────────────────────────────────────
def backtest_with_gld(df, weights, vehicle_open, vehicle_close,
                      gld_open, gld_close, cost_bps=COST_BPS,
                      initial_cash=1_000_000.0):
    """
    weight portion -> leveraged ETF, (1 - weight) -> GLD (not cash).
    """
    cost_rate = cost_bps / 10_000.0

    weights = weights.reindex(df.index).ffill().fillna(0.0)
    v_close = vehicle_close.reindex(df.index).ffill()
    v_open  = vehicle_open.reindex(df.index).ffill()
    gc      = gld_close.reindex(df.index).ffill()
    go      = gld_open.reindex(df.index).ffill()

    cc_ret    = (v_close / v_close.shift(1) - 1).fillna(0.0)
    intra_ret = (v_close / v_open - 1).fillna(0.0)
    gld_cc    = (gc / gc.shift(1) - 1).fillna(0.0)
    gld_intra = (gc / go - 1).fillna(0.0)

    w_applied = weights.shift(1).fillna(0.0)
    w_prev    = w_applied.shift(1).fillna(0.0)
    w_delta   = w_applied - w_prev
    w_abs_chg = w_delta.abs()

    nav_vals = np.empty(len(df))
    nav_val  = float(initial_cash)

    for i in range(len(df)):
        w    = w_applied.iloc[i]
        wp   = w_prev.iloc[i]
        dw   = w_delta.iloc[i]
        adw  = w_abs_chg.iloc[i]

        # Cost: trade LETF + trade GLD (both sides)
        tc = adw * nav_val * cost_rate * 2

        # LETF portion
        unchanged = min(w, wp)
        bought    = max(0.0, dw)
        letf_ret  = unchanged * cc_ret.iloc[i] + bought * intra_ret.iloc[i]

        # GLD portion
        gld_w      = 1.0 - w
        gld_wp     = 1.0 - wp
        gld_unchanged = min(gld_w, gld_wp)
        gld_bought    = max(0.0, gld_w - gld_wp)
        gld_ret    = gld_unchanged * gld_cc.iloc[i] + gld_bought * gld_intra.iloc[i]

        daily_ret = letf_ret + gld_ret
        nav_val   = nav_val * (1 + daily_ret) - tc
        nav_vals[i] = nav_val

    return pd.Series(nav_vals, index=df.index, name="nav")


# ── Run all strategies ───────────────────────────────────────────────────────
print("\nRunning backtests...\n")
weights = w_kcm(df)
bh_w = pd.Series(np.ones(len(df)), index=df.index)

# QLD
nav_qld_bh   = backtest(df, bh_w, vehicle_open=df["qld_open"], vehicle_close=df["qld_close"])
nav_qld_kcm  = backtest(df, weights, vehicle_open=df["qld_open"], vehicle_close=df["qld_close"])
nav_qld_gld  = backtest_with_gld(df, weights, df["qld_open"], df["qld_close"],
                                  df["gld_open"], df["gld_close"])

# TQQQ
nav_tqqq_bh  = backtest(df, bh_w, vehicle_open=df["tqqq_open"], vehicle_close=df["tqqq_close"])
nav_tqqq_kcm = backtest(df, weights, vehicle_open=df["tqqq_open"], vehicle_close=df["tqqq_close"])
nav_tqqq_gld = backtest_with_gld(df, weights, df["tqqq_open"], df["tqqq_close"],
                                  df["gld_open"], df["gld_close"])

# GLD B&H
nav_gld_bh   = backtest(df, bh_w, vehicle_open=df["gld_open"], vehicle_close=df["gld_close"])

# ── Metrics ──────────────────────────────────────────────────────────────────
strategies = [
    ("QLD Buy & Hold",    nav_qld_bh,    bh_w),
    ("KCM QLD",           nav_qld_kcm,   weights),
    ("KCM QLD+GLD",       nav_qld_gld,   weights),
    ("TQQQ Buy & Hold",   nav_tqqq_bh,   bh_w),
    ("KCM TQQQ",          nav_tqqq_kcm,  weights),
    ("KCM TQQQ+GLD",      nav_tqqq_gld,  weights),
    ("GLD Buy & Hold",    nav_gld_bh,    bh_w),
]

results = []
for name, nav, w in strategies:
    m = _nav_metrics(nav, w)
    results.append({**m, "name": name})

# ── Print results ────────────────────────────────────────────────────────────
print(f"\n{'='*115}")
print(f"  KCM Strategy - Synthetic NASDAQ {df.index[0].date()} -> {df.index[-1].date()} ({years:.0f} years)")
print(f"  Source: ^IXIC (1971-1985) + ^NDX (1985+) | Daily cost: {DAILY_COST*252*100:.1f}%/yr | Commission: {COST_BPS:.0f} bps")
print(f"{'='*115}")
print(f"  {'Strategy':<22} {'Ann.Ret':>8} {'Total':>14} {'MDD':>9} {'Sharpe':>7} {'Calmar':>7} {'AvgWt':>6}")
print(f"  {'-'*105}")

groups = [
    ("-- QLD (2x) --", ["QLD Buy & Hold", "KCM QLD", "KCM QLD+GLD"]),
    ("-- TQQQ (3x) --", ["TQQQ Buy & Hold", "KCM TQQQ", "KCM TQQQ+GLD"]),
    ("-- Reference --", ["GLD Buy & Hold"]),
]

for header, names in groups:
    print(f"  {header}")
    for r in results:
        if r["name"] in names:
            print(f"  {r['name']:<22} {r['ann']:>8.2%} {r['total']:>14.1%}"
                  f" {r['mdd']:>9.2%} {r['sharpe']:>7.3f} {r['calmar']:>7.3f}"
                  f" {r['avg_w']:>6.1%}")

print(f"{'='*115}")

# ── KCM stats ────────────────────────────────────────────────────────────────
in_market_pct = (weights > 0).mean() * 100
full_pct = (weights >= 0.99).mean() * 100
trades = (weights.diff().abs() > 0.01).sum()
print(f"\n  KCM in-market: {in_market_pct:.1f}% | Full position: {full_pct:.1f}% | Trades: {trades}")

# ── GLD benefit ──────────────────────────────────────────────────────────────
print(f"\n  GLD hedge benefit (vs cash at 3.25%):")
for letf, base_name, gld_name in [
    ("QLD", "KCM QLD", "KCM QLD+GLD"),
    ("TQQQ", "KCM TQQQ", "KCM TQQQ+GLD"),
]:
    base = next(r for r in results if r["name"] == base_name)
    gld  = next(r for r in results if r["name"] == gld_name)
    print(f"    {letf}: Ann {base['ann']:.2%} -> {gld['ann']:.2%} ({gld['ann']-base['ann']:+.2%})"
          f"  |  MDD {base['mdd']:.2%} -> {gld['mdd']:.2%}"
          f"  |  Sharpe {base['sharpe']:.3f} -> {gld['sharpe']:.3f}")

# ── Period breakdown ─────────────────────────────────────────────────────────
print(f"\n{'='*115}")
print(f"  Period breakdown (annualized return)")
print(f"{'='*115}")
hdr = f"  {'Period':<25} {'QLD B&H':>8} {'KCM QLD':>8} {'QLD+GLD':>8} {'TQQQ B&H':>9} {'KCM TQQQ':>9} {'TQQQ+GLD':>9} {'GLD B&H':>8}"
print(hdr)
print(f"  {'-'*105}")

periods = [
    ("1972-1974 (oil crisis)",    "1972-01-01", "1974-12-31"),
    ("1975-1979 (recovery)",      "1975-01-01", "1979-12-31"),
    ("1980-1986 (early tech)",    "1980-01-01", "1986-12-31"),
    ("1987-1989 (crash+recover)", "1987-01-01", "1989-12-31"),
    ("1990-1994 (early VIX era)", "1990-01-01", "1994-12-31"),
    ("1995-1999 (dot-com bull)",  "1995-01-01", "1999-12-31"),
    ("2000-2002 (dot-com bust)",  "2000-01-01", "2002-12-31"),
    ("2003-2006 (recovery)",      "2003-01-01", "2006-12-31"),
    ("Pre-VIX (1972-1989)",       "1972-01-01", "1989-12-31"),
    ("VIX era (1990-2006)",       "1990-01-01", "2006-12-31"),
    ("Full period",               df.index[0].strftime("%Y-%m-%d"), END_DATE),
]

for label, s, e in periods:
    mask = (df.index >= s) & (df.index <= e)
    if mask.sum() < 100:
        continue
    sub = df.loc[mask]
    w_sub = weights.loc[mask]
    bh_sub = pd.Series(np.ones(mask.sum()), index=sub.index)

    navs = []
    for veh_o, veh_c, w_s, use_gld in [
        (sub["qld_open"], sub["qld_close"], bh_sub, False),
        (sub["qld_open"], sub["qld_close"], w_sub, False),
        (sub["qld_open"], sub["qld_close"], w_sub, True),
        (sub["tqqq_open"], sub["tqqq_close"], bh_sub, False),
        (sub["tqqq_open"], sub["tqqq_close"], w_sub, False),
        (sub["tqqq_open"], sub["tqqq_close"], w_sub, True),
        (sub["gld_open"], sub["gld_close"], bh_sub, False),
    ]:
        if use_gld:
            nav = backtest_with_gld(sub, w_s, veh_o, veh_c,
                                     sub["gld_open"], sub["gld_close"])
        else:
            nav = backtest(sub, w_s, vehicle_open=veh_o, vehicle_close=veh_c)
        navs.append(_nav_metrics(nav, w_s))

    print(f"  {label:<25}"
          f" {navs[0]['ann']:>8.2%} {navs[1]['ann']:>8.2%} {navs[2]['ann']:>8.2%}"
          f" {navs[3]['ann']:>9.2%} {navs[4]['ann']:>9.2%} {navs[5]['ann']:>9.2%}"
          f" {navs[6]['ann']:>8.2%}")

print(f"{'='*115}")
