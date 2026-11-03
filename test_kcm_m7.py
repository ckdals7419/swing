"""KCM M7 backtest — equal-weight Magnificent 7 index from 1981."""
import warnings, numpy as np, pandas as pd, yfinance as yf
warnings.filterwarnings("ignore")
from strategy_qld import backtest, _nav_metrics

# ── Download all M7 ──
tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'NVDA', 'META', 'TSLA']
data = {}
for t in tickers:
    d = yf.download(t, start='1980-01-01', end='2026-06-01', progress=False)
    d.columns = d.columns.get_level_values(0)
    data[t] = d[['Open', 'Close']].copy()
    print(f"  {t}: {d.index[0].date()} -> {d.index[-1].date()}")

# ── Build equal-weight M7 index ──
all_dates = sorted(set().union(*[d.index for d in data.values()]))
all_dates = pd.DatetimeIndex(all_dates)

m7_ret = pd.DataFrame(index=all_dates)
m7_open_ret = pd.DataFrame(index=all_dates)
for t in tickers:
    c = data[t]['Close'].squeeze()
    o = data[t]['Open'].squeeze()
    m7_ret[t] = c.pct_change()
    m7_open_ret[t] = o / c.shift(1) - 1

m7_daily = m7_ret.mean(axis=1).fillna(0)
m7_overnight = m7_open_ret.mean(axis=1).fillna(0)

m7_close = pd.Series(index=all_dates, dtype=float)
m7_close.iloc[0] = 100.0
for i in range(1, len(all_dates)):
    m7_close.iloc[i] = m7_close.iloc[i-1] * (1 + m7_daily.iloc[i])

m7_open = m7_close.shift(1) * (1 + m7_overnight)
m7_open.iloc[0] = m7_close.iloc[0]

stock_count = m7_ret.notna().sum(axis=1)

# ── Build features ──
close = m7_close
sma50  = close.rolling(50, min_periods=50).mean()
sma200 = close.rolling(200, min_periods=200).mean()

delta = close.diff()
gain  = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
loss  = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
rs    = gain / loss.replace(0, np.nan)
rsi14 = 100 - 100 / (1 + rs)

vix_df = pd.read_csv('data/vix_daily.csv', parse_dates=['Date'])
vix_raw = vix_df.set_index('Date')['vix_close']
vix = vix_raw.reindex(close.index, method='ffill')
vix_sma10 = vix.rolling(10).mean()

df = pd.DataFrame(index=close.index)
df.index.name = 'date'
df['qqq_vs_sma200'] = close / sma200 - 1
df['qqq_vs_sma50']  = close / sma50 - 1
df['rsi_14']         = rsi14
df['vix_close']      = vix
df['realized_vol_20d'] = np.log(close / close.shift(1)).rolling(20).std() * np.sqrt(252)
df['bil_return_1d']  = 0.0325 / 252.0
df['m7_open']  = m7_open
df['m7_close'] = m7_close
df['stock_count'] = stock_count
df = df.dropna(subset=['qqq_vs_sma200'])

print(f"\nM7 Index composition:")
for yr in [1981, 1987, 1998, 2005, 2011, 2013, 2020, 2025]:
    mask = df.index.year == yr
    if mask.sum() > 0:
        avg_n = df.loc[mask, 'stock_count'].mean()
        print(f"  {yr}: {avg_n:.1f} stocks")

years = (df.index[-1] - df.index[0]).days / 365.25
print(f"\nDataset: {df.index[0].date()} -> {df.index[-1].date()} ({len(df):,} days, {years:.0f} years)")

vs200 = df['qqq_vs_sma200'].fillna(0).values
vs50  = df['qqq_vs_sma50'].fillna(0).values
vix_v = df['vix_close'].values
vix_sma10_v = pd.Series(vix_v).rolling(10).mean().values
rsi_v = df['rsi_14'].ffill().fillna(50).values
n = len(df)
idx = df.index


def kcm_hold():
    w = np.zeros(n)
    inp = vs200[0] >= 0
    w[0] = 1.0 if inp else 0.0
    for i in range(1, n):
        has_vix = not np.isnan(vix_v[i]) and not np.isnan(vix_sma10_v[i])
        vr = has_vix and vix_v[i] > 35 and vix_v[i] < vix_sma10_v[i]
        if inp:
            if vs200[i] < -0.01 and vs50[i] < 0:
                inp = False
                if vr or rsi_v[i] < 30: w[i] = 1.0
                else: w[i] = 0.0
            else:
                w[i] = 1.0
        else:
            if vs200[i] >= 0.01: inp = True; w[i] = 1.0
            elif vs50[i] >= 0: inp = True; w[i] = 1.0
            elif vr or rsi_v[i] < 30: w[i] = 1.0
            else: w[i] = 0.0
    return pd.Series(w, index=idx)


def kcm_rsi_swing(sell_rsi, buy_rsi):
    w = np.zeros(n)
    in_regime = vs200[0] >= 0
    swing_out = False
    w[0] = 1.0 if in_regime else 0.0
    for i in range(1, n):
        has_vix = not np.isnan(vix_v[i]) and not np.isnan(vix_sma10_v[i])
        vr = has_vix and vix_v[i] > 35 and vix_v[i] < vix_sma10_v[i]
        if in_regime:
            if vs200[i] < -0.01 and vs50[i] < 0:
                in_regime = False; swing_out = False
                if vr or rsi_v[i] < 30: w[i] = 1.0
                else: w[i] = 0.0
            elif swing_out:
                if rsi_v[i] < buy_rsi: swing_out = False; w[i] = 1.0
                else: w[i] = 0.0
            else:
                if rsi_v[i] > sell_rsi: swing_out = True; w[i] = 0.0
                else: w[i] = 1.0
        else:
            if vs200[i] >= 0.01: in_regime = True; swing_out = False; w[i] = 1.0
            elif vs50[i] >= 0: in_regime = True; swing_out = False; w[i] = 1.0
            elif vr or rsi_v[i] < 30: w[i] = 1.0
            else: w[i] = 0.0
    return pd.Series(w, index=idx)


def run(name, ws):
    nav = backtest(df, ws, vehicle_open=df['m7_open'], vehicle_close=df['m7_close'])
    m = _nav_metrics(nav, ws)
    trades = (ws.diff().abs() > 0.01).sum()
    in_pct = (ws > 0).mean() * 100
    print(f"  {name:<50} {m['ann']:>7.2%} {m['total']:>14.1%} {m['mdd']:>8.2%} {m['sharpe']:>7.3f} {m['calmar']:>7.3f} {in_pct:>5.1f}% {trades:>5}")


bh = pd.Series(np.ones(n), index=idx)

print(f"\n{'Strategy':<52} {'Ann':>7} {'Total':>14} {'MDD':>8} {'Sharpe':>7} {'Calmar':>7} {'InMkt':>6} {'Trd':>5}")
print("=" * 118)

run("M7 Buy & Hold", bh)
run("KCM M7 (hold 100%)", kcm_hold())
run("KCM M7 + RSI>80/50 swing", kcm_rsi_swing(80, 50))
run("KCM M7 + RSI>80/55 swing", kcm_rsi_swing(80, 55))
run("KCM M7 + RSI>80/60 swing", kcm_rsi_swing(80, 60))
run("KCM M7 + RSI>75/50 swing", kcm_rsi_swing(75, 50))

# ── Period breakdown ──
print("\n=== Period breakdown: KCM M7 hold vs RSI>80/50 swing ===")
hdr = f"{'Period':<25} {'Stocks':>6} {'Hold Ann':>10} {'Hold MDD':>10} {'Swing Ann':>10} {'Swing MDD':>10} {'Delta':>8}"
print(hdr)
print("-" * len(hdr))

w_hold = kcm_hold()
w_swing = kcm_rsi_swing(80, 50)

periods = [
    ("1981-1986 (AAPL only)",    "1981-01-01", "1986-12-31"),
    ("1987-1996 (AAPL+MSFT)",    "1987-01-01", "1996-12-31"),
    ("1997-2003 (+AMZN,NVDA)",   "1997-01-01", "2003-12-31"),
    ("2004-2009 (+GOOG)",        "2004-01-01", "2009-12-31"),
    ("2010-2011 (+TSLA)",        "2010-01-01", "2011-12-31"),
    ("2012-2019 (full M7)",      "2012-01-01", "2019-12-31"),
    ("2020-2026 (full M7)",      "2020-01-01", "2026-12-31"),
]

for label, s, e in periods:
    mask = (idx >= s) & (idx <= e)
    if mask.sum() < 100:
        continue
    avg_n = df.loc[mask, 'stock_count'].mean()
    nav_h = backtest(df.loc[mask], w_hold.loc[mask],
                     vehicle_open=df.loc[mask, 'm7_open'],
                     vehicle_close=df.loc[mask, 'm7_close'])
    nav_s = backtest(df.loc[mask], w_swing.loc[mask],
                     vehicle_open=df.loc[mask, 'm7_open'],
                     vehicle_close=df.loc[mask, 'm7_close'])
    mh = _nav_metrics(nav_h, w_hold.loc[mask])
    ms = _nav_metrics(nav_s, w_swing.loc[mask])
    d = ms['ann'] - mh['ann']
    print(f"  {label:<23} {avg_n:>6.1f} {mh['ann']:>10.2%} {mh['mdd']:>10.2%} {ms['ann']:>10.2%} {ms['mdd']:>10.2%} {d:>+8.2%}")
